"""
This repo is forked from [Boyuan Chen](https://boyuan.space/)'s research 
template [repo](https://github.com/buoyancy99/research-template). 
By its MIT license, you must keep the above sentence in `README.md` 
and the `LICENSE` file to credit the author.
"""

from omegaconf import DictConfig
import numpy as np
from random import random, randint
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any
from einops import rearrange, repeat

from lightning.pytorch.utilities.types import STEP_OUTPUT

from algorithms.common.base_pytorch_algo import BasePytorchAlgo
from utils.logging_utils import get_validation_metrics_for_states
from .models.diffusion_transition import DiffusionCorrectionransitionModel
from .models.anneal_prob import CosineAnnealProb


class RNN_DiffusionCorrectionBase(BasePytorchAlgo):
    def __init__(self, original_algo: BasePytorchAlgo, cfg: DictConfig):
        self.cfg = cfg
        self.x_shape = cfg.x_shape
        self.z_shape = cfg.z_shape

        self.frame_stack = cfg.frame_stack
        self.context_frames = cfg.context_frames  # number of context frames at validation time
        self.external_cond_dim = cfg.external_cond_dim
        self.sampling_timesteps = cfg.diffusion.sampling_timesteps
        self.correction_size = cfg.correction_size
        self.finetune_org_model = cfg.finetune_org_model
        self.anneal_groundtruth_rollout_cfg = cfg.anneal_groundtruth_rollout
        self.posterior_rolling = cfg.get("posterior_rolling", False)

        self.cfg.diffusion.cum_snr_decay = self.cfg.diffusion.cum_snr_decay**self.frame_stack
        self.x_stacked_shape = list(cfg.x_shape)
        self.x_stacked_shape[0] *= cfg.frame_stack

        self.is_spatial = len(self.x_stacked_shape) == 3  # pixel
        self.calc_crps_sum = cfg.calc_crps_sum
        self.min_crps_sum = float("inf")
        del original_algo.metrics

        super().__init__(cfg)
        self.original_algo = original_algo

        if self.finetune_org_model:
            self.original_algo.train()
        else:
            self.original_algo.eval()
            for param in self.original_algo.parameters():
                param.requires_grad = False
            
    def _build_model(self):
        self.transition_model = DiffusionCorrectionransitionModel(
            self.x_stacked_shape, self.z_shape, self.external_cond_dim, self.cfg.diffusion, self.cfg.backbone
        )
        self.register_data_mean_std(self.cfg.data_mean, self.cfg.data_std)

        if self.anneal_groundtruth_rollout_cfg is not None:
            self.gt_rollout_prob = CosineAnnealProb(**self.anneal_groundtruth_rollout_cfg)
        else:
            self.gt_rollout_prob = None

    def configure_optimizers(self):
        transition_params = list(self.transition_model.parameters())

        configs = [
            {"params": transition_params, "lr": self.cfg.lr, "weight_decay": self.cfg.weight_decay, "betas": self.cfg.optimizer_beta},
        ]
        if self.finetune_org_model:
            configs.append(
                {"params": self.original_algo.parameters(), "lr": self.cfg.original_algo_lr, "weight_decay": self.cfg.weight_decay, "betas": self.cfg.optimizer_beta}
            )

        optimizer_dynamics = torch.optim.AdamW(configs)
        return optimizer_dynamics

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # update params
        optimizer.step(closure=optimizer_closure)

        # manually warm up lr without a scheduler
        if self.trainer.global_step < self.cfg.warmup_steps:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.cfg.warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.cfg.lr

    def _preprocess_batch(self, batch, get_zs=True, batch_first=False):
        """Preprocess batch data for training or validation.
        
        Args:
            batch: Input batch from dataloader
            get_zs: Whether to get latent variables from original model
            batch_first: Return tensors with batch dim first if True. Only applied for xs, pred_xs, residuals, zs
        Returns:
            Processed tensors for model input in shape (b, t, (fs c), h, w)
        """
        if get_zs:
            use_groundtruth = False
            if self.gt_rollout_prob is not None:
                use_groundtruth = random() < self.gt_rollout_prob(self.trainer.global_step)
                
            outputs = self.original_algo.validation_step(batch, 0, is_log_video=False, use_groundtruth=use_groundtruth, cal_metrics=False)
            xs = outputs["xs"]
            pred_xs = outputs["xs_pred"]
            zs = outputs["zs"]

            xs = rearrange(xs, "(t fs) b c ... -> b t (fs c) ...", fs=self.frame_stack).contiguous()
            pred_xs = rearrange(pred_xs, "(t fs) b c ... -> b t (fs c) ...", fs=self.frame_stack).contiguous()
            zs = rearrange(zs, "t b ... -> b t ...").contiguous()
            pred_xs = self._normalize_x(pred_xs)
            xs = self._normalize_x(xs)

            n_frames, batch_size = xs.shape[:2]
        else:
            xs = batch[0]
            xs = rearrange(xs, "b (t fs) c ... -> b t (fs c) ...", fs=self.frame_stack).contiguous()
            xs = self._normalize_x(xs)
            batch_size, n_frames = xs.shape[:2]
        
        if n_frames % self.frame_stack != 0:
            raise ValueError("Number of frames must be divisible by frame stack size")
        if self.context_frames % self.frame_stack != 0:
            raise ValueError("Number of context frames must be divisible by frame stack size")
        n_frames_stacked = n_frames // self.frame_stack

        nonterminals = batch[-1].bool().permute(1, 0)
        masks = torch.cumprod(nonterminals, dim=0).contiguous()
        conditions = [None] * n_frames_stacked

        # pred_xs already normalized ? and in the same shape as xs

        # For training
        if get_zs:
            n_context = self.context_frames // self.frame_stack
            max_start = max(n_frames_stacked - self.correction_size, n_context)
            # Randomly select a start frame and also remove the first frame
            if self.cfg.random_start is True:
                start = randint(n_context, max_start)
            elif self.cfg.random_start == 'context':
                start = n_context
            elif self.cfg.random_start == 'start':
                start = 1 # first frame is GT
            else:
                raise ValueError("Invalid random_start value")
            
            end = min(start + self.correction_size, n_frames_stacked)

            xs = xs[:, start:end]
            pred_xs = pred_xs[:, start:end]
            zs = zs[:, start:end]
            masks = masks[start*self.frame_stack:end*self.frame_stack]
            conditions = conditions[start:end]

            if self.cfg.target_type == "diff":
                target = xs - pred_xs
            elif self.cfg.target_type == "data":
                target = xs
            else:
                raise ValueError("Invalid target_type")
            
            pred_xs = pred_xs
            if not batch_first:
                xs = rearrange(xs, "b t (fs c) ... -> t b (fs c) ...", fs=self.frame_stack)
                pred_xs = rearrange(pred_xs, "b t (fs c) ... -> t b (fs c) ...", fs=self.frame_stack)
                target = rearrange(target, "b t (fs c) ... -> t b (fs c) ...", fs=self.frame_stack)
                zs = rearrange(zs, "b t ... -> t b ...")
            return xs, pred_xs, target, zs, conditions, masks
        
        # For validation
        else:
            if self.original_algo.learnable_init_z:
                init_z = self.original_algo.init_z[None].expand(batch_size, *self.z_shape)
            else:
                init_z = torch.zeros(batch_size, *self.z_shape)
                init_z = init_z.to(xs.device)

            if not batch_first:
                xs = rearrange(xs, "b t (fs c) ... -> t b (fs c) ...", fs=self.frame_stack)

            return xs, conditions, masks, init_z

    def reweigh_loss(self, loss, weight=None):
        loss = rearrange(loss, "t b (fs c) ... -> t b fs c ...", fs=self.frame_stack)
        if weight is not None:
            expand_dim = len(loss.shape) - len(weight.shape) - 1
            weight = rearrange(weight, "(t fs) b ... -> t b fs ..." + " 1" * expand_dim, fs=self.frame_stack)
            loss = loss * weight

        return loss.mean()

    def training_step(self, batch, batch_idx):
        # training step for dynamics
        xs, org_xs_pred, target, zs, conditions, masks  = self._preprocess_batch(batch, batch_first=True) # (b, t, (fs c), h, w)

        # batch first
        pred_target, loss = self.transition_model(
            zs, target, conditions, deterministic_t=None, cum_snr=None
        ) # (b, t, (fs c), h, w)

        if self.cfg.target_type == "diff":
            xs_pred = pred_target + org_xs_pred # (b, t, (fs c), h, w)
        elif self.cfg.target_type == "data":
            xs_pred = pred_target
        else:
            raise ValueError("Invalid target_type")

        loss = rearrange(loss, "b t ... -> t b ...")
        x_loss = self.reweigh_loss(loss, masks)
        loss = x_loss

        if batch_idx % 20 == 0:
            self.log_dict(
                {
                    "training/loss": loss,
                    "training/x_loss": x_loss,
                }
            )

        # Unstack, timestep first for visualization, metrics
        xs = rearrange(xs, "b t (fs c) ... -> (t fs) b c ...", fs=self.frame_stack)
        xs_pred = rearrange(xs_pred, "b t (fs c) ... -> (t fs) b c ...", fs=self.frame_stack)
        org_xs_pred = rearrange(org_xs_pred, "b t (fs c) ... -> (t fs) b c ...", fs=self.frame_stack)

        output_dict = {
            "loss": loss,
            "xs_pred": self._unnormalize_x(xs_pred),
            "org_xs_pred": self._unnormalize_x(org_xs_pred),
            "xs": self._unnormalize_x(xs),
            "zs": rearrange(zs, "b t ... -> t b ..."),
        }

        return output_dict

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, namespace="validation", return_all_timesteps=False):
        if self.calc_crps_sum:
            # repeat batch for crps sum for time series prediction
            batch = [d[None].expand(self.calc_crps_sum, *([-1] * len(d.shape))).flatten(0, 1) for d in batch]

        xs, conditions, masks, init_z = self._preprocess_batch(batch, get_zs=False, batch_first=False) 
        # xs: (t, b, (fs c), h, w)
        # conditions: (t, (fs d))
        # masks: (t fs, b)
        # init_z: (b, c_z, z_dim, z_dim)

        n_frames, batch_size, *_ = xs.shape
        xs_pred = [xs[0]] # (t, b, (fs c), h, w)
        xs_pred_wo_z = [xs[0]]
        all_xs_pred = [] # (t, b, (fs c), h, w)
        org_xs_pred = [xs[0]] # (t, b, (fs c), h, w)

        # TODO: check: do we need to do context first: because we need to correct these context frames as well
        n_context = self.context_frames // self.frame_stack
        n_context = 0
        z = init_z
        # for t in range(1, n_context):
        #     x_next_pred, z = self.original_algo.roll_1_step(
        #         x=xs[t],
        #         z=z,
        #         condition=conditions[t],
        #         t=t
        #     )
        #     org_xs_pred.append(x_next_pred)
        # z = init_z
        t = n_context
        # prediction
        while len(xs_pred) < n_frames - n_context:
            chunk_size = min(self.correction_size, n_frames - len(xs_pred))
            sampling_steps_matrix = self.create_sampling_noise_levels(self.sampling_timesteps, chunk_size)
            ts = [t + i for i in range(chunk_size)]

            # timestep first, not include the first input frame and init_z
            chunk_org_xs_pred, chunk_zs = self.original_algo.rollout(
                first_x=xs_pred[-1], # (b, (fs c), h, w)
                init_z=z, # (b, c_z, z_dim, z_dim)
                ts=ts,
                conditions=conditions[t:t + chunk_size],
            )

            # get last z and remove it from chunk_zs
            org_xs_pred.extend(chunk_org_xs_pred.unbind(dim=0))

            # batch first, reshape for diffusion model
            chunk_org_xs_pred = rearrange(chunk_org_xs_pred, "t b ... -> b t ...")
            # get residuals from diffusion
            chunk_x_t = torch.randn((batch_size, chunk_size) + tuple(self.x_stacked_shape)).to(xs.device)
            chunk_x_t_wo_z = chunk_x_t.clone()

            for m in range(sampling_steps_matrix.shape[0]):
                current_steps = sampling_steps_matrix[m]
                chunk_x_t = self.transition_model.ddim_sample_step(
                    chunk_x_t, # (b, t, (fs c), h, w)
                    z_cond=rearrange(chunk_zs, 't b ... -> b t ...'),
                    index=current_steps,
                    external_cond=conditions[t:t + chunk_size]
                )[0]
                # chunk_x_t_wo_z = self.transition_model.ddim_sample_step(
                #     chunk_x_t_wo_z, # (b, t, (fs c), h, w)
                #     z_cond=torch.zeros_like(rearrange(chunk_zs, 't b ... -> b t ...')),
                #     index=current_steps,
                #     external_cond=conditions[t:t + chunk_size]
                # )[0]


            
            if self.cfg.target_type == "diff":
                chunk_xs_pred = chunk_x_t + chunk_org_xs_pred # (b, t, (fs c), h, w)
                # chunk_xs_pred_wo_z = chunk_x_t_wo_z + chunk_org_xs_pred
            elif self.cfg.target_type == "data":
                chunk_xs_pred = chunk_x_t
                # chunk_xs_pred_wo_z = chunk_x_t_wo_z
            else:
                raise ValueError("Invalid target_type")
            
            # timestep first
            xs_pred.extend(chunk_xs_pred.unbind(dim=1))
            # xs_pred_wo_z.extend(chunk_xs_pred_wo_z.unbind(dim=1))
            # all_xs_pred += xs_pred


            if self.posterior_rolling:
                chunk_org_xs_pred, chunk_zs = self.original_algo.rollout(
                    first_x=None, # (b, (fs c), h, w)
                    init_z=z, # (b, c_z, z_dim, z_dim)
                    ts=ts,
                    conditions=conditions[t:t + chunk_size],
                    groundtruth_xs=rearrange(chunk_xs_pred, 'b t ... -> t b ...')
                )
            # TODO: do we need to feed new pred into transition model again to get posterior z_chunk? 
            
            t += chunk_size
            z = chunk_zs[-1]

        # timestep first
        xs_pred = torch.stack(xs_pred)
        # xs_pred_wo_z = torch.stack(xs_pred_wo_z)
        org_xs_pred = torch.stack(org_xs_pred)

        loss = F.mse_loss(xs_pred, xs, reduction="none")
        loss = self.reweigh_loss(loss, masks)

        org_loss = F.mse_loss(org_xs_pred, xs, reduction="none")
        org_loss = self.reweigh_loss(org_loss, masks)

        xs = rearrange(xs, "t b (fs c) ... -> (t fs) b c ...", fs=self.frame_stack)
        xs_pred = rearrange(xs_pred, "t b (fs c) ... -> (t fs) b c ...", fs=self.frame_stack)
        # xs_pred_wo_z = rearrange(xs_pred_wo_z, "t b (fs c) ... -> (t fs) b c ...", fs=self.frame_stack)
        org_xs_pred = rearrange(org_xs_pred, "t b (fs c) ... -> (t fs) b c ...", fs=self.frame_stack)

        xs = self._unnormalize_x(xs)
        xs_pred = self._unnormalize_x(xs_pred)
        # xs_pred_wo_z = self._unnormalize_x(xs_pred_wo_z)
        org_xs_pred = self._unnormalize_x(org_xs_pred)

        if not self.is_spatial:
            if self.transition_model.return_all_timesteps:
                xs_pred_all = [torch.stack(item) for item in xs_pred_all]

                limit = self.transition_model.sampling_timesteps
                for i in np.linspace(1, limit, 5, dtype=int):
                    xs_pred = xs_pred_all[i]
                    xs_pred = self._unnormalize_x(xs_pred)
                    metric_dict = get_validation_metrics_for_states(xs_pred, xs)

                    self.log_dict(
                        {f"{namespace}/{i}_sampling_steps_{k}": v for k, v in metric_dict.items()},
                        on_step=False,
                        on_epoch=True,
                        prog_bar=True,
                    )
            else:
                metric_dict = get_validation_metrics_for_states(xs_pred, xs)
                self.log_dict(
                    {f"{namespace}/{k}": v for k, v in metric_dict.items()},
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                )

                org_metric_dict = get_validation_metrics_for_states(org_xs_pred, xs)
                self.log_dict(
                    {f"{namespace}/org_{k}": v for k, v in org_metric_dict.items()},
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                )

        if return_all_timesteps:
            all_xs_pred = torch.stack(all_xs_pred)
            all_xs_pred = rearrange(all_xs_pred, "t b d (fs c) ... -> (t fs) b d c ...", fs=self.frame_stack)
            all_xs_pred = self._unnormalize_x(all_xs_pred)
            return {
                "loss": loss,
                "xs_pred": xs_pred,
                "org_xs_pred": org_xs_pred,
                "xs": xs,
                "all_xs_pred": all_xs_pred,
            }
        else:
            return {
                "loss": loss,
                "xs_pred": xs_pred,
                # 'xs_pred_wo_z': xs_pred_wo_z,
                "org_xs_pred": org_xs_pred,
                "xs": xs,
            }

    def _normalize_x(self, xs):
        shape = [1] * (xs.ndim - self.data_mean.ndim) + list(self.data_mean.shape)
        mean = self.data_mean.reshape(shape).to(xs.device)
        std = self.data_std.reshape(shape).to(xs.device)
        return (xs - mean) / std

    def _unnormalize_x(self, xs):
        shape = [1] * (xs.ndim - self.data_mean.ndim) + list(self.data_mean.shape)
        mean = self.data_mean.reshape(shape).to(xs.device)
        std = self.data_std.reshape(shape).to(xs.device)
        return xs * std + mean

    def create_sampling_noise_levels(self, sampling_timesteps, n_frames):  
        if self.cfg.diffusion.noise_level_sampling == "linear_increasing":
            pyramid_height = sampling_timesteps + n_frames - 1
            pyramid = np.zeros((pyramid_height, n_frames), dtype=int)
            for m in range(pyramid_height):
                for t in range(n_frames):
                    pyramid[m, t] = m - t
            pyramid = np.clip(pyramid, a_min=0, a_max=sampling_timesteps-1, dtype=int)
            return pyramid
        elif self.cfg.diffusion.noise_level_sampling in ['random', 'constant']:
            noise_levels = torch.arange(0, sampling_timesteps, dtype=torch.long)
            return repeat(noise_levels, 't -> t f', f=n_frames)
        else:
            raise ValueError("Invalid noise_level_sampling")

