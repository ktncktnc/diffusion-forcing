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
from einops import rearrange

from lightning.pytorch.utilities.types import STEP_OUTPUT

from algorithms.common.base_pytorch_algo import BasePytorchAlgo
from utils.logging_utils import get_validation_metrics_for_states
from .models.diffusion_transition import DiffusionCorrectionransitionModel


class RNN_DiffusionCorrectionBase(BasePytorchAlgo):
    def __init__(self, original_algo: BasePytorchAlgo, cfg: DictConfig):
        self.cfg = cfg
        self.x_shape = cfg.x_shape
        self.z_shape = cfg.z_shape
        self.frame_stack = cfg.frame_stack
        self.cfg.diffusion.cum_snr_decay = self.cfg.diffusion.cum_snr_decay**self.frame_stack
        self.x_stacked_shape = list(cfg.x_shape)
        self.x_stacked_shape[0] *= cfg.frame_stack
        self.is_spatial = len(self.x_stacked_shape) == 3  # pixel
        self.gt_cond_prob = cfg.gt_cond_prob  # probability to condition one-step diffusion o_t+1 on ground truth o_t
        self.gt_first_frame = cfg.gt_first_frame
        self.context_frames = cfg.context_frames  # number of context frames at validation time
        self.chunk_size = cfg.chunk_size
        self.calc_crps_sum = cfg.calc_crps_sum
        self.external_cond_dim = cfg.external_cond_dim
        self.uncertainty_scale = cfg.uncertainty_scale
        self.sampling_timesteps = cfg.diffusion.sampling_timesteps
        self.validation_step_outputs = []
        self.min_crps_sum = float("inf")
        self.correction_size = cfg.correction_size

        super().__init__(cfg)
        self.original_algo = original_algo
        self.original_algo.eval()
        for param in self.original_algo.parameters():
            param.requires_grad = False

    def _build_model(self):
        self.transition_model = DiffusionCorrectionransitionModel(
            self.x_stacked_shape, self.z_shape, self.external_cond_dim, self.cfg.diffusion, self.cfg.backbone
        )
        self.register_data_mean_std(self.cfg.data_mean, self.cfg.data_std)

    def configure_optimizers(self):
        transition_params = list(self.transition_model.parameters())
        optimizer_dynamics = torch.optim.AdamW(
            transition_params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay, betas=self.cfg.optimizer_beta
        )

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
            _, (pred_xs, xs, zs) = self.original_algo.validation_step(batch, 0, return_prediction=True, save=False, return_z=True)
            n_frames, batch_size = xs.shape[:2]
            xs = rearrange(xs, "(t fs) b c ... -> b t (fs c) ...", fs=self.frame_stack).contiguous()
            pred_xs = rearrange(pred_xs, "(t fs) b c ... -> b t (fs c) ...", fs=self.frame_stack).contiguous()
            zs = rearrange(zs, "t b ... -> b t ...").contiguous()

            pred_xs = self._normalize_x(pred_xs)
        else:
            xs = batch[0]
            xs = rearrange(xs, "b (t fs) c ... -> b t (fs c) ...", fs=self.frame_stack).contiguous()
            batch_size, n_frames = xs.shape[:2]
        
        xs = self._normalize_x(xs)

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
                start = 0
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
        }

        return output_dict

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, namespace="validation"):
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
        xs_pred_all = [] # (t, b, (fs c), h, w)
        org_xs_pred = [xs[0]] # (t, b, (fs c), h, w)

        # TODO: check: do we need to do context first
        n_context = self.context_frames // self.frame_stack
        n_context = 0
        z = init_z
        t = n_context
        
        # prediction
        while len(xs_pred) < n_frames - n_context:
            generation_chunk_size = min(self.correction_size, n_frames - len(xs_pred))
            ts = [t + i for i in range(generation_chunk_size)]

            # timestep first, not include the first input frame and init_z
            chunk_org_xs_pred, chunk_zs = self.original_algo.rollout(
                first_x=xs_pred[-1], # (b, (fs c), h, w)
                init_z=z, # (b, c_z, z_dim, z_dim)
                ts=ts,
                conditions=conditions[t:t + generation_chunk_size],
            )

            # get last z and remove it from chunk_zs
            z = chunk_zs[-1]
            org_xs_pred.extend(chunk_org_xs_pred.unbind(dim=0))

            # batch first, reshape for diffusion model
            chunk_org_xs_pred = rearrange(chunk_org_xs_pred, "t b ... -> b t ...")
            # get residuals from diffusion
            pred_target = self.transition_model.ddim_sample(
                chunk_org_xs_pred.shape, # (b, t, (fs c), h, w)
                z_cond=rearrange(chunk_zs, 't b ... -> b t ...'), # (b, t, c_z, z_dim, z_dim)
                external_cond=conditions[t:t + generation_chunk_size],
                return_all_timesteps=False
            ) 
            
            if self.cfg.target_type == "diff":
                chunk_xs_pred = pred_target + chunk_org_xs_pred # (b, t, (fs c), h, w)
            elif self.cfg.target_type == "data":
                chunk_xs_pred = pred_target
            else:
                raise ValueError("Invalid target_type")
            
            # timestep first
            xs_pred.extend(chunk_xs_pred.unbind(dim=1))
            xs_pred_all += xs_pred

            t += generation_chunk_size

            # TODO: do we need to feed new pred into transition model again to get posterior z_chunk? 

        # timestep first
        xs_pred = torch.stack(xs_pred)
        org_xs_pred = torch.stack(org_xs_pred)

        loss = F.mse_loss(xs_pred, xs, reduction="none")
        loss = self.reweigh_loss(loss, masks)

        org_loss = F.mse_loss(org_xs_pred, xs, reduction="none")
        org_loss = self.reweigh_loss(org_loss, masks)

        xs = rearrange(xs, "t b (fs c) ... -> (t fs) b c ...", fs=self.frame_stack)
        xs_pred = rearrange(xs_pred, "t b (fs c) ... -> (t fs) b c ...", fs=self.frame_stack)
        org_xs_pred = rearrange(org_xs_pred, "t b (fs c) ... -> (t fs) b c ...", fs=self.frame_stack)

        xs = self._unnormalize_x(xs)
        xs_pred = self._unnormalize_x(xs_pred)
        org_xs_pred = self._unnormalize_x(org_xs_pred)

        # if not self.is_spatial:
        #     if self.transition_model.return_all_timesteps:
        #         xs_pred_all = [torch.stack(item) for item in xs_pred_all]

        #         limit = self.transition_model.sampling_timesteps
        #         for i in np.linspace(1, limit, 5, dtype=int):
        #             xs_pred = xs_pred_all[i]
        #             xs_pred = self._unnormalize_x(xs_pred)
        #             metric_dict = get_validation_metrics_for_states(xs_pred, xs)

        #             self.log_dict(
        #                 {f"{namespace}/{i}_sampling_steps_{k}": v for k, v in metric_dict.items()},
        #                 on_step=False,
        #                 on_epoch=True,
        #                 prog_bar=True,
        #             )
        #     else:
        #         metric_dict = get_validation_metrics_for_states(xs_pred, xs)
        #         self.log_dict(
        #             {f"{namespace}/{k}": v for k, v in metric_dict.items()},
        #             on_step=False,
        #             on_epoch=True,
        #             prog_bar=True,
        #         )

        #         org_metric_dict = get_validation_metrics_for_states(org_xs_pred, xs)
        #         self.log_dict(
        #             {f"{namespace}/org_{k}": v for k, v in org_metric_dict.items()},
        #             on_step=False,
        #             on_epoch=True,
        #             prog_bar=True,
        #         )

        self.validation_step_outputs.append((xs_pred.detach().cpu(), org_xs_pred.detach().cpu(), xs.detach().cpu()))

        return loss

    def on_validation_epoch_end(self, namespace="validation"):
        if not self.validation_step_outputs:
            return

        self.validation_step_outputs.clear()

    def test_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        return self.validation_step(*args, **kwargs, namespace="test")

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
