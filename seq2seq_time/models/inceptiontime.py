# from https://mohcinemadkour.github.io/posts/2019/10/Machine%20Learning,%20timeseriesAI,%20Time%20Series%20Classification,%20fastai_timeseries,%20TSC%20bechmark/


# This is an unofficial PyTorch implementation by Ignacio Oguiza - oguiza@gmail.com based on:

# Fawaz, H. I., Lucas, B., Forestier, G., Pelletier, C., Schmidt, D. F., Weber, J., ... & Petitjean, F. (2019). InceptionTime: Finding AlexNet for Time Series Classification. arXiv preprint arXiv:1909.04939.
# Official InceptionTime tensorflow implementation: https://github.com/hfawaz/InceptionTime

import torch
import torch.nn as nn
from torch.nn import functional as F


def noop(x):
    return x


def shortcut(c_in, c_out):
    return nn.Sequential(
        *[nn.Conv1d(c_in, c_out, kernel_size=1), nn.BatchNorm1d(c_out)]
    )


class InceptionLayer(nn.Module):
    def __init__(self, c_in, bottleneck=32, kernel_size=40, nb_filters=32):

        super().__init__()
        self.bottleneck = (
            nn.Conv1d(c_in, bottleneck, 1) if bottleneck and c_in > 1 else noop
        )
        mts_feat = bottleneck or c_in
        conv_layers = []
        kss = [kernel_size // (2 ** i) for i in range(3)]
        # ensure odd kss until nn.Conv1d with padding='same' is available in pytorch 1.3
        kss = [ksi if ksi % 2 != 0 else ksi - 1 for ksi in kss]
        for i in range(len(kss)):
            conv_layers.append(
                nn.Conv1d(mts_feat, nb_filters, kernel_size=kss[i], padding=kss[i] // 2)
            )
        self.conv_layers = nn.ModuleList(conv_layers)
        self.maxpool = nn.MaxPool1d(3, stride=1, padding=1)
        self.conv = nn.Conv1d(c_in, nb_filters, kernel_size=1)
        self.bn = nn.BatchNorm1d(nb_filters * 4)
        self.act = nn.ReLU()

    def forward(self, x):
        input_tensor = x
        x = self.bottleneck(input_tensor)
        for i in range(3):
            out_ = self.conv_layers[i](x)
            if i == 0:
                out = out_
            else:
                out = torch.cat((out, out_), 1)
        mp = self.conv(self.maxpool(input_tensor))
        inc_out = torch.cat((out, mp), 1)
        return self.act(self.bn(inc_out))


class InceptionBlock(nn.Module):
    def __init__(
        self, c_in, bottleneck=32, kernel_size=40, nb_filters=32, residual=True, num_layers=6
    ):

        super().__init__()

        self.residual = residual
        self.num_layers = num_layers

        # inception & residual layers
        inc_mods = []
        res_layers = []
        res = 0
        for d in range(num_layers):
            inc_mods.append(
                InceptionLayer(
                    c_in if d == 0 else nb_filters * 4,
                    bottleneck=bottleneck if d > 0 else 0,
                    kernel_size=kernel_size,
                    nb_filters=nb_filters,
                )
            )
            if self.residual and d % 3 == 2:
                res_layers.append(
                    shortcut(c_in if res == 0 else nb_filters * 4, nb_filters * 4)
                )
                res += 1
            else:
                res_layer = res_layers.append(None)
        self.inc_mods = nn.ModuleList(inc_mods)
        self.res_layers = nn.ModuleList(res_layers)
        self.act = nn.ReLU()

    def forward(self, x):
        res = x
        for d, l in enumerate(range(self.num_layers)):
            x = self.inc_mods[d](x)
            if self.residual and d % 3 == 2:
                res = self.res_layers[d](res)
                x += res
                res = x
                x = self.act(x)
        return x



class InceptionTimeSeq(nn.Module):
    def __init__(
        self,
        x_dim,
        y_dim,
        hidden_size=32,
        layers=6,
        kernel_size=40,
        bottleneck=16,
        residual=True
    ):
        super().__init__()
        self.inc_block = InceptionBlock(
            x_dim + y_dim,
            bottleneck=bottleneck,
            kernel_size=kernel_size,
            nb_filters=hidden_size,
            residual=residual,
            num_layers=layers,
        )
        self._min_std = 0.01
        self.mean = nn.Linear(hidden_size*4, y_dim)
        self.std = nn.Linear(hidden_size*4, y_dim)

    def forward(self, past_x, past_y, future_x, future_y=None):
        device = next(self.parameters()).device
        B, S, _ = future_x.shape
        future_y_fake = past_y[:, -1:, :].repeat(1, S, 1).to(device)
        context = torch.cat([past_x, past_y], -1)
        target = torch.cat([future_x, future_y_fake], -1)
        x = torch.cat([context, target * 1], 1).detach()

        out = self.inc_block(x.permute(0, 2, 1)).permute(0, 2, 1)
        
        # Seems to help a little, especially with extrapolating out of bounds
        steps = past_y.shape[1]
        mean = self.mean(out)[:, steps:, :]
        log_sigma = self.std(out)[:, steps:, :]
        
        sigma = self._min_std + (1 - self._min_std) * F.softplus(log_sigma)
        return torch.distributions.Normal(mean, sigma), {}
