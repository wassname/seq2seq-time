import torch
from torch import nn
from torch.nn import functional as F

class LSTMSeq2Seq(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=32, lstm_layers=2, lstm_dropout=0, _min_std = 0.05):
        super().__init__()
        self._min_std = _min_std

        self.encoder = nn.LSTM(
            input_size=input_size + output_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=lstm_layers,
            dropout=lstm_dropout,
        )
        self.decoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=lstm_layers,
            dropout=lstm_dropout,
        )
        self.mean = nn.Linear(hidden_size, output_size)
        self.std = nn.Linear(hidden_size, output_size)

    def forward(self, past_x, past_y, future_x, future_y=None):
        x = torch.cat([past_x, past_y], -1)
        _, (h_out, cell) = self.encoder(x)
        
        # output = [batch size, seq len, hid dim * n directions]
        outputs, (_, _) = self.decoder(future_x, (h_out, cell))
        
        # outputs: [B, T, num_direction * H]
        mean = self.mean(outputs)
        log_sigma = self.std(outputs)
        sigma = self._min_std + (1 - self._min_std) * F.softplus(log_sigma)
        y_dist = torch.distributions.Normal(mean, sigma)
        return y_dist, {}
