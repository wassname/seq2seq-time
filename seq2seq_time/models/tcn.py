import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.nn import functional as F

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()


class Conv(nn.Module):
    """Causal convolution layer."""

    def __init__(
        self,
        n_inputs,
        n_outputs,
        kernel_size,
        stride,
        dilation,
        padding,
        causal=True,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            n_inputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.chomp = Chomp1d(padding)
        self.causal = causal

    def forward(self, x):
        out = self.conv(x)
        if self.causal:
            out = self.chomp(out)
        return out


class TemporalBlock(nn.Module):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        kernel_size,
        stride,
        dilation,
        padding,
        dropout=0.2,
    ):
        super(TemporalBlock, self).__init__()
        self.conv1 = Conv(
            n_inputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = Conv(
            n_outputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.relu1, self.dropout1, self.conv2, self.relu2, self.dropout2
        )
        self.downsample = (
            Conv(
                n_inputs,
                n_outputs,
                1,
                stride=1,
                padding=0,
                dilation=1,
                causal=False,
            )
            if n_inputs != n_outputs
            else None
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = x
        for i, l in enumerate(self.net):
            out = l(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """
    See:
    - https://arxiv.org/pdf/1803.01271.pdf
    - https://github.com/locuslab/TCN
    """
    def __init__(
        self,
        num_inputs,
        num_channels,
        kernel_size=2,
        dropout=0.2,
        embedding_dim=2,
    ):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            ]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        out = x
        for l in self.network:
            out = l(out)
        return out


class TCNSeq(nn.Module):
    """
    See:
    - https://arxiv.org/pdf/1803.01271.pdf
    - https://github.com/locuslab/TCN
    """
    def __init__(
        self,
        x_dim,
        y_dim,
        hidden_size=32,
        nlayers=6,
        kernel_size=2,
        dropout=0.2,
    ):
        super().__init__()
        self.tcn = TemporalConvNet(
            num_inputs=x_dim + y_dim,
            kernel_size=kernel_size,
            num_channels=[hidden_size] * nlayers,
            dropout=dropout)
        self._min_std = 0.01
        self.mean = nn.Linear(hidden_size, y_dim)
        self.std = nn.Linear(hidden_size, y_dim)

    def forward(self, past_x, past_y, future_x, future_y=None):
        device = next(self.parameters()).device
        B, S, _ = future_x.shape
        future_y_fake = past_y[:, -1:, :].repeat(1, S, 1).to(device)
        context = torch.cat([past_x, past_y], -1)
        target = torch.cat([future_x, future_y_fake], -1)
        x = torch.cat([context, target * 1], 1).detach()

        out = self.tcn(x.permute(0, 2, 1)).permute(0, 2, 1)
        
        # Seems to help a little, especially with extrapolating out of bounds
        steps = past_y.shape[1]
        mean = self.mean(out)[:, steps:, :]
        log_sigma = self.std(out)[:, steps:, :]
        
        sigma = self._min_std + (1 - self._min_std) * F.softplus(log_sigma)
        return torch.distributions.Normal(mean, sigma), {}
