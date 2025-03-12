from typing import Sequence
import torch
import random
import os
import numpy as np
import cv2
from omegaconf import DictConfig
from torchvision import transforms
from pathlib import Path
from abc import abstractmethod, ABC
from torch.utils.data import DataLoader
import json


class BasePredictionVideoDataset(torch.utils.data.Dataset, ABC):
    """
    Base class for video datasets with prediction from a model. Videos may be of variable length.

    Folder structure of each dataset:
    - [save_dir] (specified in config, e.g., data/phys101)
        - /[split] (one per split)
            - /data_folder_name (e.g., videos)
            metadata.json
    """

    def __init__(self, cfg: DictConfig, original_dataloader: DataLoader, original_algo, root_dir: str, split: str = "training"):
        super().__init__()
        self.cfg = cfg
        self.original_dataloader = original_dataloader
        self.original_dataset = original_dataloader.dataset

        self.original_algo = original_algo
        self.split = split
        self.save_dir = Path(root_dir)
        self.split_dir = self.save_dir / f"{split}"
        self.split_dir.mkdir(exist_ok=True, parents=True)
        
        self.on_the_fly = cfg.on_the_fly

        if not self.on_the_fly:
            raise NotImplementedError("Currently only on-the-fly prediction is supported")

    
    # def generate_dataset(self):
    #     """
    #     Using model to generate predictions
    #     TODO: Currently only for RNN
    #     """
    #     # No need to save ground truth, only predictions
    #     all_z = []
    #     all_pred_xs = []
    #     for batch_idx, batch in enumerate(self.original_dataloader):
    #         self.original_algo.validation_step(batch, batch_idx, save_z=True)
        
    #     for batch in self.original_algo.validation_step_outputs:
    #         pred_x, _, z = batch
    #         all_z.append(z)
    #         all_pred_xs.append(pred_x)
        
    #     # stack all predictions along the batch dimension
    #     all_z = torch.stack(all_z, 1)
    #     all_pred_xs = torch.stack(all_pred_xs, 1)

    #     # save by video
    #     for video_idx in range(len(self.original_dataset.video_len())):
    #         begin_idx, 
    #          = self.original_dataset.video_idx_to_begin_and_end(video_idx)
    #         z = all_z[begin_idx:end_idx]
    #         pred_xs = all_pred_xs[begin_idx:end_idx]

    #         torch.save(z, self.split_dir / f"z_{video_idx}.pt")
    #         torch.save(pred_xs, self.split_dir / f"pred_xs_{video_idx}.pt")

    def __len__(self):
        return len(self.original_dataset)
    
    def __getitem__(self, idx):
        item = self.original_dataset[idx]

        if self.on_the_fly:
            _, (pred_x, z) = self.original_algo.validation_step(item, idx, return_prediction=True)
            return item, pred_x, z
        else:
            raise NotImplementedError("Currently only on-the-fly prediction is supported")

