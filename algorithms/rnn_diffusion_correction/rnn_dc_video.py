import numpy as np
from einops import rearrange
import torch
from omegaconf import DictConfig

from .rnn_dc_base import RNN_DiffusionCorrectionBase
from algorithms.common.old_metrics import (
    FrechetInceptionDistance,
    LearnedPerceptualImagePatchSimilarity,
    FrechetVideoDistance,
)
from utils.logging_utils import log_video, log_multiple_videos, get_validation_metrics_for_videos
from algorithms.common.base_pytorch_algo import BasePytorchAlgo


class RNN_DiffusionCorrectionVideo(RNN_DiffusionCorrectionBase):
    def __init__(self, original_algo: BasePytorchAlgo, cfg: DictConfig):
        self.metrics = cfg.metrics
        super().__init__(original_algo, cfg)

    def _build_model(self):
        super()._build_model()

        self.validation_fid_model = FrechetInceptionDistance(feature=64) if "fid" in self.metrics else None
        self.validation_lpips_model = LearnedPerceptualImagePatchSimilarity() if "lpips" in self.metrics else None
        self.validation_fvd_model = [FrechetVideoDistance()] if "fvd" in self.metrics else None

    def training_step(self, batch, batch_idx):
        output_dict = super().training_step(batch, batch_idx)

        if batch_idx % 500 == 0 and self.logger is not None:
            log_multiple_videos(
                [output_dict["xs_pred"], output_dict["org_xs_pred"], output_dict["xs"]],
                step=self.global_step,
                namespace="training_vis_all",
                logger=self.logger.experiment,
                add_red_border=True
            )

        return output_dict

    def on_validation_epoch_end(self, namespace="validation"):
        if not self.validation_step_outputs:
            return

        xs_pred = []
        org_xs_pred = []
        xs = []
        for pred, org_pred, gt in self.validation_step_outputs:
            xs_pred.append(pred)
            org_xs_pred.append(org_pred)
            xs.append(gt)
        xs_pred = torch.cat(xs_pred, 1)
        org_xs_pred = torch.cat(org_xs_pred, 1)
        xs = torch.cat(xs, 1)

        if self.logger is not None:
            log_multiple_videos(
                [xs_pred, org_xs_pred, xs],
                step=None if namespace == "test" else self.global_step,
                namespace=namespace + "_vis_all",
                logger=self.logger.experiment,
                add_red_border=True
            )

        metric_dict = get_validation_metrics_for_videos(
            xs_pred[self.context_frames :],
            xs[self.context_frames :],
            lpips_model=self.validation_lpips_model,
            fid_model=self.validation_fid_model,
            fvd_model=(self.validation_fvd_model[0] if self.validation_fvd_model else None),
        )
        self.log_dict(
            {f"{namespace}/{k}": v for k, v in metric_dict.items()}, on_step=False, on_epoch=True, prog_bar=True
        )

        org_metric_dict = get_validation_metrics_for_videos(
            org_xs_pred[self.context_frames :],
            xs[self.context_frames :],
            lpips_model=self.validation_lpips_model,
            fid_model=self.validation_fid_model,
            fvd_model=(self.validation_fvd_model[0] if self.validation_fvd_model else None),
        )
        self.log_dict(
            {f"{namespace}/org_{k}": v for k, v in org_metric_dict.items()}, on_step=False, on_epoch=True, prog_bar=True
        )

        self.validation_step_outputs.clear()

    def on_test_epoch_end(self) -> None:
        return self.on_validation_epoch_end(namespace="test")

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
