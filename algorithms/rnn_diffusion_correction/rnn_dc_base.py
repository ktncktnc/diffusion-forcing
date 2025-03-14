"""
This repo is forked from [Boyuan Chen](https://boyuan.space/)'s research 
template [repo](https://github.com/buoyancy99/research-template). 
By its MIT license, you must keep the above sentence in `README.md` 
and the `LICENSE` file to credit the author.
"""

from omegaconf import DictConfig
import numpy as np
from random import random
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
            self.x_stacked_shape, self.z_shape, self.external_cond_dim, self.cfg.diffusion
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

    def _preprocess_batch(self, batch, get_zs=True):
        if get_zs:
            # # TODO: currently generate whole zs => change to fix frames, random start
            _, (pred_xs, _, zs) = self.original_algo.validation_step(batch, 0, return_prediction=True, save_z=True)

        xs = batch[0]
        batch_size, n_frames = xs.shape[:2]

        if n_frames % self.frame_stack != 0:
            raise ValueError("Number of frames must be divisible by frame stack size")
        if self.context_frames % self.frame_stack != 0:
            raise ValueError("Number of context frames must be divisible by frame stack size")

        nonterminals = batch[-1]
        nonterminals = nonterminals.bool().permute(1, 0)
        masks = torch.cumprod(nonterminals, dim=0).contiguous()
        n_frames = n_frames // self.frame_stack

        if self.external_cond_dim:
            conditions = batch[1]
            conditions = torch.cat([torch.zeros_like(conditions[:, :1]), conditions[:, 1:]], 1)
            conditions = rearrange(conditions, "b (t fs) d -> t b (fs d)", fs=self.frame_stack).contiguous()
        else:
            conditions = [None for _ in range(n_frames)]

        # pred_xs already normalized ? and in the same shape as xs
        xs = self._normalize_x(xs)    

        # Remove first frame since we use it as ground truth
        xs = rearrange(xs, "b (t fs) c ... -> t b (fs c) ...", fs=self.frame_stack).contiguous()
        if get_zs:
            xs = xs[1:]
            pred_xs = rearrange(pred_xs, "(t fs) b c ... -> t b (fs c) ...", fs=self.frame_stack).contiguous()
            masks = masks[self.frame_stack:]
            residuals = xs - pred_xs
            return xs, pred_xs, residuals, zs, conditions, masks

        if self.original_algo.learnable_init_z:
            init_z = self.original_algo.init_z[None].expand(batch_size, *self.z_shape)
        else:
            init_z = torch.zeros(batch_size, *self.z_shape)
            init_z = init_z.to(xs.device)
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
        xs, org_xs_pred, residuals, zs, conditions, masks  = self._preprocess_batch(batch)

        n_frames, batch_size, _, *_ = xs.shape

        loss = []
        # denoise all timesteps
        # TODO: edit cum_snr_next, check forward
        residual_pred, loss, cum_snr_next = self.transition_model(
            zs, residuals, conditions, deterministic_t=None, cum_snr=None
        )
        xs_pred = residual_pred + org_xs_pred
        
        x_loss = self.reweigh_loss(loss, masks)
        loss = x_loss

        if batch_idx % 20 == 0:
            self.log_dict(
                {
                    "training/loss": loss,
                    "training/x_loss": x_loss,
                }
            )

        xs = rearrange(xs, "t b (fs c) ... -> (t fs) b c ...", fs=self.frame_stack)
        xs_pred = rearrange(xs_pred, "t b (fs c) ... -> (t fs) b c ...", fs=self.frame_stack)
        org_xs_pred = rearrange(org_xs_pred, "t b (fs c) ... -> (t fs) b c ...", fs=self.frame_stack)

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

        xs, conditions, masks, init_z = self._preprocess_batch(batch, get_zs=False)

        n_frames, batch_size, *_ = xs.shape
        xs_pred = [xs[0]]
        xs_pred_all = []
        org_xs_pred = [xs[0]]

        # TODO: check: do we need to do context first
        n_context = self.context_frames // self.frame_stack
        n_context = 0
        z = init_z
        # for t in range(0, n_context):
        #     x_next_pred, z = self.original_algo.roll_1_step(
        #         x=xs[t],
        #         z=z,
        #         condition=conditions[t],
        #         t=t
        #     )
        #     org_xs_pred.append(x_next_pred)

        t = n_context
        # prediction

        while len(xs_pred) < n_frames - n_context:
            generation_chunk_size = min(self.correction_size, n_frames - len(xs_pred))
            ts = [t + i for i in range(generation_chunk_size)]

            chunk_org_xs_pred, chunk_zs = self.original_algo.rollout(
                first_x=xs_pred[-1], # (b, (fs c), h, w)
                init_z=z, # (b, c_z, z_dim, z_dim)
                ts=ts,
                conditions=conditions[t:t + generation_chunk_size],
            )

            # get last z and remove it from chunk_zs
            z = chunk_zs[-1]
            chunk_zs = chunk_zs[:-1]

            org_xs_pred += chunk_org_xs_pred

            # get residuals from diffusion
            residual_pred = self.transition_model.ddim_sample(
                chunk_org_xs_pred.shape, # (b, t, (fs c), h, w)
                z_cond=chunk_zs, # (b, t, c_z, z_dim, z_dim)
                external_cond=conditions[t:t + generation_chunk_size],
                return_all_timesteps=False
            )
            chunk_xs_pred = residual_pred + chunk_org_xs_pred
            xs_pred += chunk_xs_pred
            xs_pred_all += xs_pred

            t += generation_chunk_size

            # TODO: do we need to feed new pred into transition model again to get posterior z_chunk? 

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
