import numpy as np
from einops import rearrange
import torch
from typing import Any, Dict
from omegaconf import DictConfig
from lightning.pytorch.utilities.types import STEP_OUTPUT
from .rnn_dc_base import RNN_DiffusionCorrectionBase
from algorithms.common.metrics.video import VideoMetric
from utils.logging_utils import log_video, log_multiple_videos, get_validation_metrics_for_videos
from algorithms.common.base_pytorch_algo import BasePytorchAlgo


class RNN_DiffusionCorrectionVideo(RNN_DiffusionCorrectionBase):
    def __init__(self, original_algo: BasePytorchAlgo, cfg: DictConfig):
        super().__init__(original_algo, cfg)

    def _build_model(self):
        super()._build_model()
        self.metrics = VideoMetric(
            metric_types=self.cfg.evaluation.metrics,
            frame_wise=self.cfg.evaluation.frame_wise,
            max_frames=self.cfg.evaluation.max_frames,
        )

    def training_step(self, batch, batch_idx):
        output_dict = super().training_step(batch, batch_idx)

        n_samples = 5
        # norm zs to [0,1], min max keep dim 0
        zs = output_dict["zs"]
        min_vals = zs.reshape(zs.shape[1], -1).min(dim=1, keepdim=True)[0].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        max_vals = zs.reshape(zs.shape[1], -1).max(dim=1, keepdim=True)[0].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        zs = (zs - min_vals) / (max_vals - min_vals + 1e-8)
        zs = zs[:, :, :3, :, :]

        if self.logger is not None and ((batch_idx < 15000 and batch_idx % 1500 == 0) or (batch_idx >= 15000 and batch_idx % 5000 == 0)):
            log_multiple_videos(
                [output_dict["xs_pred"][:, :n_samples], output_dict["org_xs_pred"][:, :n_samples], zs[:, :n_samples], output_dict["xs"][:, :n_samples]],
                step=self.global_step,
                namespace="training_vis_all",
                logger=self.logger.experiment,
                add_red_border=True
            )

        return output_dict
    
    def on_validation_epoch_start(self) -> None:
        if self.cfg.evaluation.seed is not None:
            self.generator = torch.Generator(device=self.device).manual_seed(self.cfg.evaluation.seed)

    def validation_step(self, batch, batch_idx, namespace="validation"):
        outputs = super().validation_step(batch, batch_idx, namespace)
        xs_pred = outputs["xs_pred"]
        org_xs_pred = outputs["org_xs_pred"]
        xs = outputs["xs"]

        # Calculate metrics
        self.metrics(xs_pred, xs)

        # log the video
        if self.logger:
            log_multiple_videos(
                [xs_pred, org_xs_pred, xs],
                step=None if namespace == "test" else self.global_step,
                namespace=namespace + "_vis_all",
                logger=self.logger.experiment,
                add_red_border=True
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