from typing import Any, Optional, Tuple, Union, Literal
from torch import Tensor
from torchmetrics.image import (
    StructuralSimilarityIndexMeasure,
    PeakSignalNoiseRatio,
)
from einops import rearrange


class FrameWise_PSNR(PeakSignalNoiseRatio):
    def __init__(
        self,
        data_range: Optional[float] = None,
        base: float = 10.0,
        reduction: Literal["elementwise_mean", "sum", "none", None] = "elementwise_mean",
        dim: Optional[Union[int, Tuple[int, ...]]] = None,
        max_frames: int = 1,
        **kwargs: Any,
    ):
        super().__init__(
            data_range=data_range,
            base=base,
            reduction=reduction,
            dim=dim,
            **kwargs,
        )
        self.max_frames = max_frames
        if self.max_frames > 1:
            self.frame_wise = True
        else:
            self.frame_wise = False
    
    def compute(self) -> Tensor:
        output = super().compute()
        return rearrange(output, "(t b) -> t b", t=self.max_frames).mean(dim=1)
    
