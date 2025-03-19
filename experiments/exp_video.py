from datasets.video import (
    MinecraftVideoDataset,
    DmlabVideoDataset,
)
from algorithms.diffusion_forcing import DiffusionForcingVideo
from algorithms.rnn_diffusion_forcing import RNN_DiffusionForcingVideo
from algorithms.rnn_diffusion_correction import RNN_DiffusionCorrectionVideo
from algorithms.joint_rnn_diffusion_correction import JointRNN_DiffusionCorrectionVideo
from algorithms.rnn import RNNVideo
from .exp_base import BaseLightningExperiment


class VideoPredictionExperiment(BaseLightningExperiment):
    """
    A video prediction experiment
    """

    compatible_algorithms = dict(
        df_video=DiffusionForcingVideo,
        rnn_df_video=RNN_DiffusionForcingVideo,
        rnn_dc_video=RNN_DiffusionCorrectionVideo,
        joint_rnn_dc_video=JointRNN_DiffusionCorrectionVideo,
        rnn_video=RNNVideo
    )

    need_original_algo = ["rnn_dc_video", "joint_rnn_dc_video"]

    compatible_datasets = dict(
        # video datasets
        video_minecraft=MinecraftVideoDataset,
        video_dmlab=DmlabVideoDataset,
    )
