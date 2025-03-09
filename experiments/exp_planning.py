from datasets import Maze2dOfflineRLDataset
from algorithms.diffusion_forcing import DiffusionForcingPlanning
from algorithms.rnn_diffusion_forcing import RNN_DiffusionForcingPlanning
from .exp_base import BaseLightningExperiment


class PlanningExperiment(BaseLightningExperiment):
    """
    A Partially Observed Markov Decision Process experiment
    """

    compatible_algorithms = dict(
        df_planning=DiffusionForcingPlanning,
        rnn_df_planning=RNN_DiffusionForcingPlanning
    )

    compatible_datasets = dict(
        # Planning datasets
        maze2d_umaze=Maze2dOfflineRLDataset,
        maze2d_medium=Maze2dOfflineRLDataset,
        maze2d_large=Maze2dOfflineRLDataset,
    )
