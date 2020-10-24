import torch
from torch import nn
from torch.nn import functional as F

class LSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64, lstm_layers=3, lstm_dropout=0, _min_std = 0.05, nan_value=0):
        super().__init__()
        self._min_std = _min_std
        self.nan_value = nan_value

        self.lstm = nn.LSTM(
            input_size=input_size + output_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=lstm_layers,
            dropout=lstm_dropout,
        )
        self.mean = nn.Linear(hidden_size, output_size)
        self.std = nn.Linear(hidden_size, output_size)

    def forward(self, past_x, past_y, future_x, future_y=None):
        device = next(self.parameters()).device
        B, S, _ = future_x.shape
        future_y_fake = past_y[:, -1:, :].repeat(1, S, 1).to(device)
        # future_y_fake = (
        #     torch.ones(past_y.shape[0], future_x.shape[1], past_y.shape[2]).float().to(device) * self.nan_value
        # )
        context = torch.cat([past_x, past_y], -1).detach()
        target = torch.cat([future_x, future_y_fake], -1).detach()
        x = torch.cat([context, target * 1], 1).detach()
        
        steps = past_y.shape[1]
        outputs, _ = self.lstm(x)
        outputs = outputs[:, steps:, :]
        
        # outputs: [B, T, num_direction * H]
        mean = self.mean(outputs)
        log_sigma = self.std(outputs)
        sigma = self._min_std + (1 - self._min_std) * F.softplus(log_sigma)
        y_dist = torch.distributions.Normal(mean, sigma)
        return y_dist, {}
