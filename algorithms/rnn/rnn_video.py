import numpy as np
from einops import rearrange
import torch
from omegaconf import DictConfig
from typing import Any, Dict
from lightning.pytorch.utilities.types import STEP_OUTPUT

from .rnn_base import RNNBase
from algorithms.common.metrics.video import VideoMetric
from utils.logging_utils import log_video


class RNNVideo(RNNBase):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

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

        output_dict = super().training_step(batch, batch_idx)
        if self.logger is not None and ((batch_idx < 15000 and batch_idx % 2500 == 0) or (batch_idx >= 15000 and batch_idx % 10000 == 0)):
            log_video(
                output_dict["xs_pred"],
                output_dict["xs"],
                step=self.global_step,
                namespace="training_vis",
                logger=self.logger.experiment
            )
        return output_dict
    
    def on_validation_epoch_start(self) -> None:
        if self.cfg.evaluation.seed is not None:
            self.generator = torch.Generator(device=self.device).manual_seed(self.cfg.evaluation.seed)

    def validation_step(self, batch, batch_idx, namespace="validation", is_log_video=True, cal_metrics=True, use_groundtruth=False):
        outputs = super().validation_step(batch, batch_idx, namespace, use_groundtruth)
        xs_pred = outputs["xs_pred"]
        xs = outputs["xs"]

        # Calculate metrics
        if cal_metrics:
            self.metrics(xs_pred, xs)

        # log the video
        if is_log_video and self.logger:
            log_video(
                xs_pred,
                xs,
                step=None if namespace == "test" else self.global_step,
                namespace=namespace + "_vis",
                context_frames=0,
                logger=self.logger.experiment,
            )
        return outputs

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

    def load_state_dict(self, state_dict, strict = True, assign = False):
        return super().load_state_dict(state_dict, False, assign)