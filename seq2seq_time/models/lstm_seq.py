import torch
from torch import nn
from torch.nn import functional as F

class LSTMSeq(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=32, lstm_layers=2, lstm_dropout=0, _min_std = 0.05, nan_value=0):
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
        x = torch.cat([past_x, past_y], -1).detach()
        
        steps = future_x.shape[1]
        outputs, _ = self.lstm(x)
        outputs = outputs[:, -steps:, :]
        
        # outputs: [B, T, num_direction * H]
        mean = self.mean(outputs)
        log_sigma = self.std(outputs)
        sigma = self._min_std + (1 - self._min_std) * F.softplus(log_sigma)
        y_dist = torch.distributions.Normal(mean, sigma)
        return y_dist, {}
