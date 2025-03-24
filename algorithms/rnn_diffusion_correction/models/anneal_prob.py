import math
import torch.nn as nn


class CosineAnnealProb(nn.Module):
    def __init__(self, initial_prob, min_prob, anneal_steps):
        super(CosineAnnealProb, self).__init__()
        self.initial_prob = initial_prob
        self.min_prob = min_prob
        self.anneal_steps = anneal_steps

    def forward(self, t):
        prob = self.min_prob + 0.5 * (self.initial_prob - self.min_prob) * (
            1 + math.cos(t * math.pi / self.anneal_steps)
        )
        return prob
