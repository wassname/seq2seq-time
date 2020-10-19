import torch
from torch import nn
from torch.nn import functional as F

class BaselineLast(nn.Module):
    def __init__(self):
        super().__init__()
        self.std = nn.Parameter(torch.tensor(1.))

    def forward(self, past_x, past_y, future_x, future_y=None):
        device = next(self.parameters()).device
        B, S, F = future_x.shape
        mean = past_y[:, -1:].repeat(1, S, 1)
        std = (self.std * 1.0).repeat(1, S, 1)
        return torch.distributions.Normal(mean, std), {}
