from typing import Any, Literal
import torch
from einops import rearrange
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.image.lpip import (
    LearnedPerceptualImagePatchSimilarity as _LearnedPerceptualImagePatchSimilarity,
    _valid_img,
)
from torchmetrics.utilities.imports import _LPIPS_AVAILABLE
from .shared_registry import SharedVideoMetricModelRegistry
from .types import VideoMetricModelType


class LearnedPerceptualImagePatchSimilarity(_LearnedPerceptualImagePatchSimilarity):
    """
    Calculates Learned Perceptual Image Patch Similarity (LPIPS) metric.
    Requires a batch of images of shape (B, C, H, W) and range [0, 1] (normalize=True) or [-1, 1] (normalize=False).
    """

    def __init__(
        self,
        registry: SharedVideoMetricModelRegistry,
        reduction: Literal["sum", "mean"] = "mean",
        normalize: bool = False,
        frame_wise=False,
        max_frames=None,
        **kwargs: Any,
    ) -> None:
        Metric.__init__(self, **kwargs)  # pylint: disable=non-parent-init-called

        if not _LPIPS_AVAILABLE:
            raise ModuleNotFoundError(
                "LPIPS metric requires that lpips is installed."
                " Either install as `pip install torchmetrics[image]` or `pip install lpips`."
            )

        self.registry = registry
        self.frame_wise = frame_wise
        self.max_frames = max_frames

        valid_reduction = ("mean", "sum")
        if reduction not in valid_reduction:
            raise ValueError(
                f"Argument `reduction` must be one of {valid_reduction}, but got {reduction}"
            )
        self.reduction = reduction

        if not isinstance(normalize, bool):
            raise ValueError(
                f"Argument `normalize` should be an bool but got {normalize}"
            )
        self.normalize = normalize

        if self.frame_wise:
            assert self.max_frames is not None, "When frame_wise is True, max_frames must be set."
            assert self.max_frames > 0, "max_frames must be greater than 0."
            self.add_state("sum_scores", torch.zeros(self.max_frames), dist_reduce_fx="sum")
            self.add_state("total", torch.zeros(self.max_frames), dist_reduce_fx="sum")
        else:        
            self.add_state("sum_scores", torch.tensor(0.0), dist_reduce_fx="sum")
            self.add_state("total", torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, img1: Tensor, img2: Tensor) -> None:  # type: ignore
        """Update internal states with lpips score."""
        if self.frame_wise:
            n_frames = img1.shape[0]
            batch_size = img1.shape[1]
            n_frames_to_update = min(n_frames, self.max_frames)
            
            # Process each frame separately to get per-frame LPIPS
            for t in range(n_frames_to_update):
                # Extract frame t across all batches
                frame1 = img1[t]  # Shape: [b, c, h, w]
                frame2 = img2[t]  # Shape: [b, c, h, w]
                
                # Validate image format
                if not (_valid_img(frame1, self.normalize) and _valid_img(frame2, self.normalize)):
                    raise ValueError(
                        f"Expected images at frame {t} to be normalized tensors with shape [N, 3, H, W]."
                        f" Got shapes {frame1.shape} and {frame2.shape} with values in range"
                        f" {[frame1.min(), frame1.max()]} and {[frame2.min(), frame2.max()]} when all values are"
                        f" expected to be in the {[0,1] if self.normalize else [-1,1]} range."
                    )
                
                # Compute LPIPS for this frame
                frame_loss = self.registry(
                    VideoMetricModelType.LPIPS, frame1, frame2, normalize=self.normalize
                ).squeeze()
                
                # Update metrics for this frame
                self.sum_scores[t] += frame_loss.sum()
                self.total[t] += batch_size
        else:
            if not (_valid_img(img1, self.normalize) and _valid_img(img2, self.normalize)):
                raise ValueError(
                    "Expected both input arguments to be normalized tensors with shape [N, 3, H, W]."
                    f" Got input with shape {img1.shape} and {img2.shape} and values in range"
                    f" {[img1.min(), img1.max()]} and {[img2.min(), img2.max()]} when all values are"
                    f" expected to be in the {[0,1] if self.normalize else [-1,1]} range."
                )
            loss = self.registry(
                VideoMetricModelType.LPIPS, img1, img2, normalize=self.normalize
            ).squeeze()
            # pylint: disable=no-member
            self.sum_scores += loss.sum()
            self.total += img1.shape[0]
