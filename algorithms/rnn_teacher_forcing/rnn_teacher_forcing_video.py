import numpy as np
from typing import Any, Dict
from einops import rearrange
import torch
from omegaconf import DictConfig
from lightning.pytorch.utilities.types import STEP_OUTPUT

from .rnn_teacher_forcing_base import RNN_TeacherForcingBase
from algorithms.common.metrics.video import VideoMetric
from utils.logging_utils import log_video, log_multiple_videos


class RNN_TeacherForcingVideo(RNN_TeacherForcingBase):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.strict_loading = False

    def _build_model(self):
        super()._build_model()
        self.metrics = VideoMetric(
            metric_types=self.cfg.evaluation.metrics,
            frame_wise=self.cfg.evaluation.frame_wise,
            max_frames=self.cfg.evaluation.max_frames,
        )

    def training_step(self, batch, batch_idx):
        # if batch_idx == 0:
        #     self.visualize_noise(batch)

        outputs = super().training_step(batch, batch_idx)
        n_samples = 4
        if self.logger and batch_idx % 1000 == 0:
            xs_pred = outputs["xs_pred"][:, :n_samples]
            xs = outputs["xs"][:, :n_samples]
            zs = outputs["zs"][:, :n_samples]
            noised_xs = outputs["noised_xs"][:, :n_samples]

            # norm zs to [0,1], min max keep dim 0
            min_vals = zs.view(zs.shape[0], -1).min(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            max_vals = zs.view(zs.shape[0], -1).max(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            zs = (zs - min_vals) / (max_vals - min_vals + 1e-8)
            zs = zs[:, :, :3, :, :]

            # Calculate metrics
            self.metrics(xs_pred, xs)

            # log the video
            log_multiple_videos(
                [xs_pred, noised_xs, zs, xs],
                step=self.global_step,
                namespace="training_vis",
                context_frames=0,
                logger=self.logger.experiment,
            )

        return outputs

    def on_validation_epoch_start(self) -> None:
        if self.cfg.evaluation.seed is not None:
            self.generator = torch.Generator(device=self.device).manual_seed(self.cfg.evaluation.seed)
    
    def validation_step(self, batch, batch_idx, namespace="validation"):
        # outputs = super().validation_step(batch, batch_idx, namespace)
        outputs = super().training_step(batch, batch_idx)
        xs_pred = outputs["xs_pred"]
        xs = outputs["xs"]
        zs = outputs["zs"]
        noised_xs = outputs["noised_xs"]
        # # norm zs to [0,1], min max keep dim 0
        # # norm zs to [0,1], min max keep dim 0
        min_vals = zs.view(zs.shape[0], -1).min(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        max_vals = zs.view(zs.shape[0], -1).max(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        zs = (zs - min_vals) / (max_vals - min_vals + 1e-8)
        zs = zs[:, :, :3, :, :]
        # print('zs', zs.shape)
        # print('zs mean', zs.mean((1,2,3,4)))
        # print('zs std', zs.std((1,2,3,4)))

        # Calculate metrics
        self.metrics(xs_pred, xs)
        print('xs', xs.shape)
        print('xs_pred', xs_pred.shape)
        print('noised_xs', noised_xs.shape)
        print('zs', zs.shape)
        # log the video
        if self.logger:
            log_multiple_videos(
                [xs_pred, noised_xs, zs, xs],
                step=None if namespace == "test" else self.global_step,
                namespace=namespace + "_vis",
                context_frames=0,
                logger=self.logger.experiment,
            )
            


    def on_validation_epoch_end(self, namespace="validation") -> None:
        self.log_dict(
            self.metrics.log('generation'),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

    def test_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        return self.validation_step(*args, **kwargs, namespace="test")

    def on_test_epoch_start(self) -> None:
        self.on_validation_epoch_start()

    def on_test_epoch_end(self) -> None:
        self.on_validation_epoch_end(namespace="test")

    def visualize_noise(self, batch):
        self.log_dict({"pixel_mean": torch.mean(batch[0]), "pixel_std": torch.std(batch[0])})

        xs = self._preprocess_batch(batch)[0]

        xs = rearrange(xs, "t b (fs c) ... -> (t fs) b c ...", fs=self.frame_stack)
        batch_size = xs.shape[1]
        x = xs[0]
        xs = []
        xs_noised = []
        for t in np.linspace(0, self.cfg.diffusion.timesteps - 1, 100):
            xs.append(x)
            t = torch.Tensor([int(t)] * batch_size).long().to(x.device)
            x = self.transition_model.q_sample(x, t)
            xs_noised.append(x)

        xs = self._unnormalize_x(torch.stack(xs))
        xs_noised = self._unnormalize_x(torch.stack(xs_noised))

        log_video(
            xs_noised,
            xs,
            step=self.global_step,
            namespace="noise_visualization",
            context_frames=0,
            logger=self.logger.experiment,
        )

    def load_state_dict(self, state_dict, strict = True, assign = False):
        return super().load_state_dict(state_dict, False, assign)