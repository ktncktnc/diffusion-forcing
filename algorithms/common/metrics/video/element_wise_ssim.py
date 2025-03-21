from typing import Any, Optional, Sequence, Union, Literal
from torchmetrics import StructuralSimilarityIndexMeasure
from torch import Tensor
from einops import rearrange

class ElementWiseSSIM(StructuralSimilarityIndexMeasure):
    """
    Element-wise Structural Similarity Index Measure (SSIM) for video.
    """

    def __init__(
        self,
        n_frames: int,
        gaussian_kernel: bool = True,
        sigma: Union[float, Sequence[float]] = 1.5,
        kernel_size: Union[int, Sequence[int]] = 11,
        reduction: Literal["elementwise_mean", "sum"] = "elementwise_mean",
        data_range: Optional[float] = None,
        k1: float = 0.01,
        k2: float = 0.03,
        return_full_image: bool = False,
        return_contrast_sensitivity: bool = False,
        **kwargs: Any,
    ):
        self.n_frames = n_frames
        self.org_reduction = reduction
        super().__init__(
            gaussian_kernel=gaussian_kernel,
            sigma=sigma,
            kernel_size=kernel_size,
            reduction="none",
            data_range=data_range,
            k1=k1,
            k2=k2,
            return_full_image=return_full_image,
            return_contrast_sensitivity=return_contrast_sensitivity,
            **kwargs,
        )

    def update(self, preds: Tensor, target: Tensor) -> None:
        n_frames = preds.shape[0]
        batch_size = preds.shape[1]

        assert n_frames == self.n_frames, f"Expected {self.n_frames} frames, but got {n_frames}"

        preds = rearrange(preds, "t b ... -> (t b) ...")
        target = rearrange(target, "t b ... -> (t b) ...")

        super().update(preds, target)

    def compute(self) -> Tensor:
        similarity = super().compute()
        similarity = rearrange(similarity, "(t b) ... -> t b ...", t=self.n_frames)

        if self.org_reduction == "elementwise_mean":
            similarity = similarity.mean(1)
        elif self.org_reduction == "sum":
            similarity = similarity.sum(1)

        return similarity

