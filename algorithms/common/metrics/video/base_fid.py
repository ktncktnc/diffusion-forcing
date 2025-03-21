from typing import Optional
from abc import ABC, abstractmethod
import torch
from einops import rearrange, einsum
from torch import Tensor
from torchmetrics import Metric
import numpy as np
import scipy.linalg
from torchmetrics.image import FrechetInceptionDistance as _FrechetInceptionDistance
from torchmetrics.image.fid import _compute_fid 
from .shared_registry import SharedVideoMetricModelRegistry
from torchmetrics.utilities import rank_zero_info


def batch_sqrtm(input_data: Tensor) -> Tensor:
    """
    Args:
        input_data (Tensor): Batch of positive definite matrices with shape (B,N,N)

    Returns:
        Tensor: Batch of matrix square roots with shape (B,N,N)
    """
    # Convert to numpy array for scipy operation
    batch_size = input_data.shape[0]
    matrix_size = input_data.shape[1]
    
    # Move to CPU and convert to numpy
    input_np = input_data.detach().cpu().numpy().astype(np.float_)
    
    # Initialize output array
    sqrtm_np = np.zeros_like(input_np)
    
    # Compute matrix square root for each matrix in the batch
    for i in range(batch_size):
        scipy_res, _ = scipy.linalg.sqrtm(input_np[i], disp=False)
        sqrtm_np[i] = scipy_res.real
    
    # Convert back to torch tensor and move to the same device as input
    sqrtm = torch.from_numpy(sqrtm_np).to(input_data)
    
    # Save for backward pass
    return sqrtm


def _compute_batch_fid(mu1: Tensor, sigma1: Tensor, mu2: Tensor, sigma2: Tensor, eps: float = 1e-6) -> Tensor:
    r"""Adjusted version of `Fid Score`_

    The Frechet Inception Distance between two multivariate Gaussians X_x ~ N(mu_1, sigm_1)
    and X_y ~ N(mu_2, sigm_2) is d^2 = ||mu_1 - mu_2||^2 + Tr(sigm_1 + sigm_2 - 2*sqrt(sigm_1*sigm_2)).

    Args:
        mu1: mean of activations calculated on predicted (x) samples
        sigma1: covariance matrix over activations calculated on predicted (x) samples
        mu2: mean of activations calculated on target (y) samples
        sigma2: covariance matrix over activations calculated on target (y) samples
        eps: offset constant - used if sigma_1 @ sigma_2 matrix is singular

    Returns:
        Scalar value of the distance between sets.
    """
    diff = mu1 - mu2

    covmean = batch_sqrtm(sigma1.bmm(sigma2))
    # Product might be almost singular
    if not torch.isfinite(covmean).all():
        rank_zero_info(f"FID calculation produces singular product; adding {eps} to diagonal of covariance estimates")
        offset = torch.eye(sigma1.size(0), device=mu1.device, dtype=mu1.dtype) * eps
        covmean = batch_sqrtm((sigma1 + offset).bmm(sigma2 + offset))

    diff_dot = einsum(diff, "b i->b")
    tr_covmean = einsum(covmean, "b i i->b")
    tr_sigma1 = einsum(sigma1, "b i i->b")
    tr_sigma2 = einsum(sigma2, "b i i->b")
    return diff_dot + tr_sigma1 + tr_sigma2 - 2 * tr_covmean


class BaseFrechetDistance(_FrechetInceptionDistance, ABC):
    """
    Base class for Fréchet distance metrics (e.g. FID, FVD).
    AAdapted from `torchmetrics.image.FrechetInceptionDistance` to work with shared model registry and support different feature extractors and modalities (e.g. images, videos).
    """

    orig_dtype: torch.dtype

    def __init__(
        self,
        registry: Optional[SharedVideoMetricModelRegistry],
        features: int,
        reset_real_features=True,
        frame_wise=False,
        max_frames: Optional[int] = None,
        **kwargs,
    ):
        # pylint: disable=non-parent-init-called
        Metric.__init__(self, **kwargs)

        self.registry = registry
        if not isinstance(reset_real_features, bool):
            raise ValueError("Argument `reset_real_features` expected to be a bool")
        self.reset_real_features = reset_real_features

        num_features = features
        self.frame_wise = frame_wise
        self.max_frames = max_frames
        mx_nb_feets = (num_features, num_features)
        # Calculate features for each timesteps
        if self.frame_wise:
            assert max_frames is not None

            self.add_state(
                "real_features_sum",
                torch.zeros((max_frames, num_features)).double(),
                dist_reduce_fx="sum",
            )
            self.add_state(
                "real_features_cov_sum",
                torch.zeros((max_frames, num_features, num_features)).double(),
                dist_reduce_fx="sum",
            )
            self.add_state(
                "real_features_num_samples", torch.zeros((max_frames,)).long(), dist_reduce_fx="sum"
            )

            self.add_state(
                "fake_features_sum",
                torch.zeros((max_frames, num_features)).double(),
                dist_reduce_fx="sum",
            )
            self.add_state(
                "fake_features_cov_sum",
                torch.zeros((max_frames, num_features, num_features)).double(),
                dist_reduce_fx="sum",
            )
            self.add_state(
                "fake_features_num_samples", torch.zeros((max_frames,)).long(), dist_reduce_fx="sum"
            )
        # Merge timesteps into batch_size, calculate features for all timesteps
        else:
            self.add_state(
                "real_features_sum",
                torch.zeros(num_features).double(),
                dist_reduce_fx="sum",
            )
            self.add_state(
                "real_features_cov_sum",
                torch.zeros(mx_nb_feets).double(),
                dist_reduce_fx="sum",
            )
            self.add_state(
                "real_features_num_samples", torch.tensor(0).long(), dist_reduce_fx="sum"
            )

            self.add_state(
                "fake_features_sum",
                torch.zeros(num_features).double(),
                dist_reduce_fx="sum",
            )
            self.add_state(
                "fake_features_cov_sum",
                torch.zeros(mx_nb_feets).double(),
                dist_reduce_fx="sum",
            )
            self.add_state(
                "fake_features_num_samples", torch.tensor(0).long(), dist_reduce_fx="sum"
            )

    @property
    def is_empty(self) -> bool:
        # pylint: disable=no-member
        if self.frame_wise:
            return (
                (self.real_features_num_samples == 0).any()
                or (self.fake_features_num_samples == 0).any()
            )
        return (
            self.real_features_num_samples == 0 or self.fake_features_num_samples == 0
        )

    @abstractmethod
    def extract_features(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    @staticmethod
    def _check_input(fake: Tensor, real: Tensor) -> bool:
        return True

    def _update(self, x: Tensor, real: bool) -> None:
        # pylint: disable=no-member
        if self.frame_wise:
            n_frames = x.shape[0]
            assert n_frames == self.max_frames, f"Expected {self.max_frames} frames, got {n_frames}."
            x = rearrange(x, "t b ... -> (t b) ...")

        features = self.extract_features(x)
        self.orig_dtype = features.dtype
        features = features.double()

        if self.frame_wise:
            features = rearrange(features, "(t b) ... -> t b ...", t=n_frames)
            
        if features.dim() == 1:
            features = features.unsqueeze(0)

        # timesteps first
        if self.frame_wise:
            if real:
                self.real_features_sum += features.sum(dim=1)
                self.real_features_cov_sum += features.transpose(1,2).bmm(features)
                self.real_features_num_samples += features.size(1)
            else:
                self.fake_features_sum += features.sum(dim=1)
                self.fake_features_cov_sum += features.transpose(1,2).bmm(features)
                self.fake_features_num_samples += features.size(1)
        # batch first
        else:
            if real:
                self.real_features_sum += features.sum(dim=0)
                self.real_features_cov_sum += features.t().mm(features)
                self.real_features_num_samples += features.size(0)
            else:
                self.fake_features_sum += features.sum(dim=0)
                self.fake_features_cov_sum += features.t().mm(features)
                self.fake_features_num_samples += features.size(0)

    def update(self, fake: Tensor, real: Tensor) -> None:
        if not self._check_input(fake, real):
            return
        self._update(fake, real=False)
        self._update(real, real=True)

    def compute(self) -> Tensor:
        """Calculate FID score based on accumulated extracted features from the two distributions."""
        if self.frame_wise:
            mean_real = (self.real_features_sum / self.real_features_num_samples.unsqueeze(1)).unsqueeze(1) # (t, f)
            mean_fake = (self.fake_features_sum / self.fake_features_num_samples.unsqueeze(1)).unsqueeze(1) # (t, f)

            # mean_real.t().mm(mean_real) # (f, f)
            # self.real_features_num_samples * mean_real.t().mm(mean_real)
            cov_real_num = self.real_features_cov_sum - self.real_features_num_samples[...,None, None] * mean_real.transpose(1,2).bmm(mean_real)
            cov_real = cov_real_num / (self.real_features_num_samples[...,None, None] - 1)
            cov_fake_num = self.fake_features_cov_sum - self.fake_features_num_samples[...,None, None] * mean_fake.transpose(1,2).bmm(mean_fake)
            cov_fake = cov_fake_num / (self.fake_features_num_samples[...,None, None] - 1)
            output = _compute_batch_fid(mean_real.squeeze(1), cov_real, mean_fake.squeeze(1), cov_fake).to(self.orig_dtype)
            return output
            
        else:
            mean_real = (self.real_features_sum / self.real_features_num_samples).unsqueeze(0)
            mean_fake = (self.fake_features_sum / self.fake_features_num_samples).unsqueeze(0)

            cov_real_num = self.real_features_cov_sum - self.real_features_num_samples * mean_real.t().mm(mean_real)
            cov_real = cov_real_num / (self.real_features_num_samples - 1)
            cov_fake_num = self.fake_features_cov_sum - self.fake_features_num_samples * mean_fake.t().mm(mean_fake)
            cov_fake = cov_fake_num / (self.fake_features_num_samples - 1)
            return _compute_fid(mean_real.squeeze(0), cov_real, mean_fake.squeeze(0), cov_fake).to(self.orig_dtype)