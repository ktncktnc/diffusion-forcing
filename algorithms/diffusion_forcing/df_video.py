from omegaconf import DictConfig
import torch
from typing import Optional, Any, Dict, Literal, Callable, Tuple
from lightning.pytorch.utilities.types import STEP_OUTPUT
from algorithms.common.old_metrics import (
    FrechetInceptionDistance,
    LearnedPerceptualImagePatchSimilarity,
    FrechetVideoDistance,
)
from algorithms.common.metrics.video import VideoMetric
from .df_base import DiffusionForcingBase
from utils.logging_utils import log_video, get_validation_metrics_for_videos


class DiffusionForcingVideo(DiffusionForcingBase):
    """
    A video prediction algorithm using Diffusion Forcing.
    """

    def __init__(self, cfg: DictConfig):
        self.n_tokens = cfg.n_frames // cfg.frame_stack  # number of max tokens for the model
        super().__init__(cfg)

    def _build_model(self):
        super()._build_model()
        self.metrics = VideoMetric(metric_types=self.cfg.evaluation.metrics)

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        output_dict = super().training_step(batch, batch_idx)
        # log the video
        if batch_idx % 5000 == 0 and self.logger:
            log_video(
                output_dict["xs_pred"],
                output_dict["xs"],
                step=self.global_step,
                namespace="training_vis",
                logger=self.logger.experiment,
            )
        return output_dict
    
    def on_validation_epoch_start(self) -> None:
        if self.cfg.evaluation.seed is not None:
            self.generator = torch.Generator(device=self.device).manual_seed(self.cfg.evaluation.seed)
    
    def validation_step(self, batch, batch_idx, namespace="validation"):
        outputs = super().validation_step(batch, batch_idx, namespace)
        xs_pred = outputs["xs_pred"]
        xs = outputs["xs"]

        # Calculate metrics
        self.metrics(xs_pred, xs)

        # log the video
        if self.logger:
            log_video(
                xs_pred,
                xs,
                step=None if namespace == "test" else self.global_step,
                namespace=namespace + "_vis",
                context_frames=self.context_frames,
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