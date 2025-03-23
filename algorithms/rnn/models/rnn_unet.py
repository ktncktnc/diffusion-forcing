from .unet import TransitionUnet, TransitionMlp
from .resnet import ResBlock2d
import torch.nn as nn
import torch
from typing import Optional, Tuple, Union
from collections import namedtuple
import torch.nn.functional as F


ModelPrediction = namedtuple("ModelPrediction", ["pred_x", "pred_z"])

class RNNUNet(nn.Module):
    def __init__(self, x_shape, z_shape, external_cond_dim, cfg):
        super().__init__()
        self.cfg = cfg
        self.x_shape = x_shape
        self.z_shape = z_shape
        self.external_cond_dim = external_cond_dim
        self.network_size = cfg.network_size
        self.num_gru_layers = cfg.num_gru_layers
        self.num_mlp_layers = cfg.num_mlp_layers
        self.self_condition = cfg.self_condition

        self._build_model()
    
    def _build_model(self):
        x_channel = self.x_shape[0]
        z_channel = self.z_shape[0]
        if len(self.x_shape) == 3:
            self.model = TransitionUnet(
                z_channel=z_channel,
                x_channel=x_channel,
                external_cond_dim=self.external_cond_dim,
                network_size=self.network_size,
                num_gru_layers=self.num_gru_layers,
                self_condition=self.self_condition,
            )

            self.x_from_z = nn.Sequential(
                ResBlock2d(z_channel, x_channel),
                nn.Conv2d(x_channel, x_channel, 1, padding=0),
            )

        elif len(self.x_shape) == 1:
            self.model = TransitionMlp(
                z_dim=z_channel,
                x_dim=x_channel,
                external_cond_dim=self.external_cond_dim,
                network_size=self.network_size,
                num_gru_layers=self.num_gru_layers,
                num_mlp_layers=self.num_mlp_layers,
                self_condition=False,
            )

            self.x_from_z = nn.Linear(z_channel, x_channel)

        else:
            raise ValueError(f"x_shape must have 1 or 3 dims but got shape {self.x_shape}")

    def forward(self, x, t, z_cond, x_next, external_cond=None):
        """
        Forward pass for the RNN UNet model.
        
        Args:
            x: Input tensor
            t: Time step
            z_cond: Latent conditioning variable
            external_cond: Optional external conditioning information
            x_self_cond: Optional self-conditioning on x
            
        Returns:
            ModelPrediction containing predicted next state (pred_x) and next latent state (pred_z)
        """
        pred = self.model_predictions(x, t, z_cond, external_cond)
        pred_x = pred.pred_x
        pred_z = pred.pred_z
        if x_next is not None:
            loss = F.mse_loss(pred_x, x_next.detach(), reduction="none")
        else:
            loss = None

        return pred_x, pred_z, loss


    def model_predictions(self, x, t, z_cond, external_cond=None, x_self_cond=None):
        if type(t) == int:
            t = torch.tensor([t]*x.shape[0], device=x.device, dtype=x.dtype)

        z_next = self.model(x, t, z_cond, external_cond, x_self_cond)
        pred_next_x = self.x_from_z(z_next)
        return ModelPrediction(pred_next_x, z_next)
    
    def mapping_config(self):
        return {
            "network_size": self.network_size,
            "num_gru_layers": self.num_gru_layers,
            "self_condition": self.self_condition,
        }