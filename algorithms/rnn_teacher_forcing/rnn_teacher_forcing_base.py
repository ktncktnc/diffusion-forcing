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
from .models.diffusion_transition import DiffusionTransitionModel
from utils.logging_utils import log_video


class RNN_TeacherForcingBase(BasePytorchAlgo):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.x_shape = cfg.x_shape
        self.z_shape = cfg.z_shape
        self.frame_stack = cfg.frame_stack
        self.cfg.diffusion.cum_snr_decay = self.cfg.diffusion.cum_snr_decay**self.frame_stack
        self.x_stacked_shape = list(cfg.x_shape)
        self.x_stacked_shape[0] *= cfg.frame_stack
        self.is_spatial = len(self.x_shape) == 3  # pixel
        self.gt_cond_prob = cfg.gt_cond_prob  # probability to condition one-step diffusion o_t+1 on ground truth o_t
        self.gt_first_frame = cfg.gt_first_frame
        self.context_frames = cfg.context_frames  # number of context frames at validation time
        self.calc_crps_sum = cfg.calc_crps_sum
        self.external_cond_dim = cfg.external_cond_dim
        self.uncertainty_scale = cfg.uncertainty_scale
        self.sampling_timesteps = cfg.diffusion.sampling_timesteps
        self.validation_step_outputs = []
        self.min_crps_sum = float("inf")
        self.learnable_init_z = cfg.learnable_init_z

        super().__init__(cfg)

    def _build_model(self):
        self.transition_model = DiffusionTransitionModel(
            self.x_stacked_shape, self.z_shape, self.external_cond_dim, self.cfg.diffusion
        )
        self.register_data_mean_std(self.cfg.data_mean, self.cfg.data_std)
        if self.learnable_init_z:
            self.init_z = nn.Parameter(torch.randn(list(self.z_shape)), requires_grad=True)

    def configure_optimizers(self):
        transition_params = list(self.transition_model.parameters())
        if self.learnable_init_z:
            transition_params.append(self.init_z)
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

    def _preprocess_batch(self, batch):
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

        xs = self._normalize_x(xs)
        xs = rearrange(xs, "b (t fs) c ... -> t b (fs c) ...", fs=self.frame_stack).contiguous()

        if self.learnable_init_z:
            init_z = self.init_z[None].expand(batch_size, *self.z_shape)
        else:
            init_z = torch.zeros(batch_size, *self.z_shape)
            init_z = init_z.to(xs.device)

        return xs, conditions, masks, init_z

    def reweigh_loss(self, loss, weight=None):
        loss = rearrange(loss, "t b (fs c) ... -> t b fs c ...", fs=self.frame_stack)
        if weight is not None:
            expand_dim = len(loss.shape) - len(weight.shape) - 1
            weight = rearrange(weight, "(t fs) b ... -> t b fs ..." + " 1" * expand_dim, fs=self.frame_stack)
            weight = weight
            loss = loss * weight
        return loss.mean()

    def training_step(self, batch, batch_idx):
        # training step for dynamics
        # training in diffusion forcing scheme: z_t+1 ~ p(z_t+1 | z_t, x_t^i, c_t) => reduce error accumulation?
        xs, conditions, masks, *_, init_z = self._preprocess_batch(batch)

        n_frames, batch_size, _, *_ = xs.shape
        xs_pred = [xs[0]] # using first frame as ground truth
        zs = [init_z]
        noised_xs = [xs[0]]

        loss = []
        z = init_z
        cum_snr = None
        for t in range(1, n_frames):
            deterministic_noise_level = None

            z_next, x_next_pred, l, cum_snr, noised_x_next = self.transition_model(
                xs[t-1], z, t, xs[t], deterministic_noise_level, conditions[t], cum_snr=cum_snr
            )

            z = z_next
            xs_pred.append(x_next_pred)
            noised_xs.append(noised_x_next)
            zs.append(z_next)
            loss.append(l)

        xs_pred = torch.stack(xs_pred)
        noised_xs = torch.stack(noised_xs)
        zs = torch.stack(zs)
        loss = torch.stack(loss)

        x_loss = self.reweigh_loss(loss, masks[self.frame_stack:])
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

        output_dict = {
            "loss": loss,
            "xs_pred": self._unnormalize_x(xs_pred),
            "xs": self._unnormalize_x(xs),
            "zs": zs,
            'noised_xs': self._unnormalize_x(noised_xs),
        }

        return output_dict

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, namespace="validation"):
        if self.calc_crps_sum:
            # repeat batch for crps sum for time series prediction
            batch = [d[None].expand(self.calc_crps_sum, *([-1] * len(d.shape))).flatten(0, 1) for d in batch]

        xs, conditions, masks, *_, init_z = self._preprocess_batch(batch)

        n_frames, batch_size, *_ = xs.shape
        xs_pred = [xs[0]]
        zs = [init_z]
        xs_pred_all = []
        z = init_z

        # context
        # use GT frame to generate hidden z, then use z to generate next frame
        n_context = self.context_frames // self.frame_stack
        for t in range(1, n_context):
            t_tensor = torch.full((batch_size,), t, device=z.device).long()
            z_next = self.transition_model.predict_z_next(
                x=xs[t-1],
                t=t_tensor,
                z_cond=z,
                external_cond=conditions[t],
            )
            x_next_pred, _ = self.transition_model.ddim_sample(
                shape=xs[t].shape,
                t=t_tensor,
                z_cond=z_next,
                external_cond=conditions[t]
            )
            xs_pred.append(x_next_pred)
            zs.append(z_next)

            # if t == 1:
            #     all_noised_xs = []
            #     xs_noised = []
            #     for j in np.linspace(0, self.cfg.diffusion.timesteps - 1, 100):
            #         t_tensor = torch.full((batch_size,), j, device=z.device).long()
            #         # _, x_next_pred, l, cum_snr = self.transition_model(
            #         #     xs[t-1], z, t, xs[t], j, conditions[t], cum_snr=None
            #         # )
            #         x_noised = self.transition_model.q_sample(xs[t-1], t_tensor)
            #         preds = self.transition_model.predict_diffusion_target(x_noised, t_tensor, z_next)
            #         # all_noised_xs.append(preds)
            #         xs_noised.append(x_noised)
            #         all_noised_xs.append(preds.pred_noise)

            #     xs_noised = torch.stack(xs_noised)
            #     all_noised_xs = torch.stack(all_noised_xs)

            #     xs_noised = rearrange(xs_noised, "t b (fs c) ... -> (t fs) b c ...", fs=self.frame_stack)
            #     all_noised_xs = rearrange(all_noised_xs, "t b (fs c) ... -> (t fs) b c ...", fs=self.frame_stack)

            #     xs_noised = self._unnormalize_x(xs_noised)
            #     all_noised_xs = self._unnormalize_x(all_noised_xs)

            #     log_video(
            #         all_noised_xs,
            #         xs_noised,
            #         step=None if namespace == "test" else self.global_step,
            #         namespace=f"{namespace}_reverse_process_vis",
            #         context_frames=0,
            #         logger=self.logger.experiment,
            #     )

            z = z_next


        # prediction
        # x = x_next_pred
        t = n_context
        while len(xs_pred) < n_frames:
            tensor_t = torch.full((batch_size,), t, device=z.device).long()
            z_next = self.transition_model.predict_z_next(
                x=xs_pred[-1],
                t=tensor_t,
                z_cond=z,
                external_cond=conditions[t],
            )
            x_next, _ = self.transition_model.ddim_sample(
                shape=xs[t].shape,
                t=tensor_t,
                z_cond=z_next,
                external_cond=conditions[t],
            )
            # x_next, z_next = self.transition_model.ddim_sample(
            #     shape=x.shape,
            #     t=t,
            #     z_cond=z,
            #     external_cond=conditions[t],
            # )

            xs_pred.append(x_next)
            zs.append(z_next)
            z = z_next
            t += 1

        xs_pred = torch.stack(xs_pred)
        zs = torch.stack(zs)
        loss = F.mse_loss(xs_pred, xs, reduction="none")
        loss = self.reweigh_loss(loss, masks)

        xs = rearrange(xs, "t b (fs c) ... -> (t fs) b c ...", fs=self.frame_stack)
        xs_pred = rearrange(xs_pred, "t b (fs c) ... -> (t fs) b c ...", fs=self.frame_stack)

        xs = self._unnormalize_x(xs)
        xs_pred = self._unnormalize_x(xs_pred)

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

        #self.validation_step_outputs.append((xs_pred.detach().cpu(), xs.detach().cpu()))

        return {
            "loss": loss,
            "xs_pred": xs_pred,
            "xs": xs,
            "zs": zs,
        }

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
