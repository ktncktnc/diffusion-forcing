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
from .models.rnn_unet import RNNUNet


class RNNBase(BasePytorchAlgo):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.x_shape = cfg.x_shape
        self.z_shape = cfg.z_shape
        self.frame_stack = cfg.frame_stack
        self.x_stacked_shape = list(cfg.x_shape)
        self.x_stacked_shape[0] *= cfg.frame_stack
        self.is_spatial = len(self.x_shape) == 3  # pixel
        self.gt_first_frame = cfg.gt_first_frame
        self.context_frames = cfg.context_frames  # number of context frames at validation time
        self.chunk_size = cfg.chunk_size
        self.calc_crps_sum = cfg.calc_crps_sum
        self.external_cond_dim = cfg.external_cond_dim
        self.uncertainty_scale = cfg.uncertainty_scale
        self.validation_step_outputs = []
        self.min_crps_sum = float("inf")
        self.learnable_init_z = cfg.learnable_init_z

        super().__init__(cfg)

    def _build_model(self):
        self.transition_model = RNNUNet(
            x_shape=self.x_stacked_shape,
            z_shape=self.z_shape,
            external_cond_dim=self.external_cond_dim,
            cfg=self.cfg.model
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
        """
        Preprocess a batch of data for RNN processing.
        
        This function:
        1. Extracts and validates frames from the batch
        2. Creates masks from nonterminal flags
        3. Processes external conditions if present
        4. Normalizes and rearranges the frame dimensions
        5. Initializes the latent state z
        
        Args:
            batch (list): A batch of data where:
                - batch[0] contains the frames with shape [batch_size, n_frames, ...]
                - batch[1] contains external conditions (optional)
                - batch[-1] contains nonterminal flags
        
        Returns:
            tuple: Contains:
                - xs (Tensor): Normalized and rearranged frames [n_frames, batch_size, ...]
                - conditions (list/Tensor): Processed external conditions
                - masks (Tensor): Mask tensor derived from nonterminals
                - init_z (Tensor): Initial latent state
        
        Raises:
            ValueError: If number of frames or context frames is not divisible by frame_stack
        """
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

        # if padding_first_frame:
        #     xs = torch.cat([torch.zeros_like(xs[:1]), xs], 0)
        #     conditions = torch.cat([torch.zeros_like(conditions[:1]), conditions], 0)

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
            loss = loss * weight

        return loss.mean()

    def rollout(self, first_x, init_z, ts, conditions):
        """
        Rollout the model for n_frames.
        
        Args:
            xs (Tensor): Initial frames [n_frames, batch_size, ...]
            init_z (Tensor): Initial latent state
            conditions (Tensor): External conditions [n_frames, batch_size, ...]
            n_frames (int): Number of frames to rollout
        
        Returns:
            tuple: Contains (not include first frame and init_z):
                - xs_pred (Tensor): Predicted frames [n_frames, batch_size, ...]
                - zs (list): Latent states for each frame
        """
        xs_pred = []
        zs = []
        z = init_z
        x = first_x

        for i, t in enumerate(ts):
            x_next, z_next = self.roll_1_step(
                x=x,
                t=t,
                z=z,
                condition=conditions[i]
            )
            z = z_next
            x = x_next
            xs_pred.append(x_next)
            zs.append(z)

        xs_pred = torch.stack(xs_pred)
        zs = torch.stack(zs)

        return xs_pred, zs
    
    def roll_1_step(self, x, t, z, condition):
        x_next, z_next, _ = self.transition_model(
            x=x,
            t=t,
            z_cond=z,
            x_next=None,
            external_cond=condition
        )
        return x_next, z_next

    def forward(self, batch, batch_idx, n_frames=None):
        """
        Generate n-1 frames given the first frame.
        """
        xs, conditions, masks, *_, init_z = self._preprocess_batch(batch)
        xs_n_frames, batch_size, _, *_ = xs.shape
        n_frames = n_frames or xs_n_frames

        xs_pred = [xs[0]]
        loss = []
        z = init_z
        zs = [z]

        for t in range(0, n_frames-1):
            x_next, z_next, _ = self.transition_model(
                x=xs[t],
                t=t,
                z_cond=z,
                x_next=xs[t+1],
                external_cond=conditions[t],
            )
            z = z_next
            xs_pred.append(x_next)
            zs.append(z)

        xs_pred = torch.stack(xs_pred)
        zs = torch.stack(zs)
        
        xs = rearrange(xs, "t b (fs c) ... -> (t fs) b c ...", fs=self.frame_stack)
        xs_pred = rearrange(xs_pred, "t b (fs c) ... -> (t fs) b c ...", fs=self.frame_stack)
        xs = self._unnormalize_x(xs)
        xs_pred = self._unnormalize_x(xs_pred)

        return xs_pred, xs, zs


    def training_step(self, batch, batch_idx):
        # training step for dynamics
        xs, conditions, masks, *_, init_z = self._preprocess_batch(batch)

        n_frames, batch_size, _, *_ = xs.shape
        # n_context = self.context_frames // self.frame_stack

        xs_pred = [xs[0]]
        loss = []
        z = init_z

        for t in range(1, n_frames):
            x_next, z_next, l = self.transition_model(
                x=xs[t-1],
                x_next=xs[t],
                z_cond=z,
                external_cond=conditions[t],
                t=t
            )
            z = z_next
            xs_pred.append(x_next)
            loss.append(l)

        xs_pred = torch.stack(xs_pred) # n_frames - 1 w.o. first frame
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
        xs = self._unnormalize_x(xs)
        xs_pred = self._unnormalize_x(xs_pred)

        output_dict = {
            "loss": loss,
            "xs_pred": xs_pred,
            "xs": xs,
        }

        return output_dict

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, namespace="validation"):
        """
        Generate n-1 next frames given the first frame.
        """
        if self.calc_crps_sum:
            # repeat batch for crps sum for time series prediction
            batch = [d[None].expand(self.calc_crps_sum, *([-1] * len(d.shape))).flatten(0, 1) for d in batch]
        
        xs, conditions, masks, *_, init_z = self._preprocess_batch(batch)
        n_frames, batch_size, *_ = xs.shape
        xs_pred = [xs[0]]
        xs_pred_all = []
        z = init_z
        zs = [z]

        # using first frame as input, generate n-1 next frames
        # => only generate n-1 frames
        # context
        n_context = self.context_frames // self.frame_stack
        for t in range(1, n_context):
            x_next_pred, z = self.roll_1_step(xs[t], t, z, conditions[t])
            xs_pred.append(x_next_pred)
            zs.append(z)

        t = n_context
        # prediction
        while len(xs_pred) < n_frames:
            x_next, z_next, _ = self.transition_model(
                x=xs_pred[-1],
                t=t,
                z_cond=z,
                x_next=None,
                external_cond=conditions[t]
            )
            z = z_next
            xs_pred.append(x_next)
            zs.append(z)
            t+=1
            
        xs_pred = torch.stack(xs_pred)
        zs = torch.stack(zs)
        # using n_context frame as groundtruth => remove n_context frame
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

        output_dict = {
            "loss": loss,
            "xs_pred": xs_pred,
            "xs": xs,
            "zs": zs,
        }
        
        return output_dict

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
    