import torch
from torch import nn
from torch.nn import functional as F


class TransformerSeq(nn.Module):
    """
    A single transformer, masking nan or 0
    """
    def __init__(self, x_dim, y_dim, attention_dropout=0, nhead=8, nlayers=2, hidden_size=16, nan_value=0, min_std=0.01):
        super().__init__()
        self._min_std = min_std
        self.nan_value = nan_value
        enc_x_dim = x_dim + y_dim

        self.enc_emb = nn.Linear(enc_x_dim, hidden_size)
        encoder_norm = nn.LayerNorm(hidden_size)
        layer_enc = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            dim_feedforward=hidden_size*4,
            dropout=attention_dropout,
            nhead=nhead,
            # activation
        )
        self.encoder = nn.TransformerEncoder(
            layer_enc, num_layers=nlayers, norm=encoder_norm
        )
        self.mean = nn.Linear(hidden_size, y_dim)
        self.std = nn.Linear(hidden_size, y_dim)

    def forward(self, past_x, past_y, future_x, future_y=None):
        device = next(self.parameters()).device
        x = torch.cat([past_x, past_y], -1).detach()

        # Masks
        x_mask = torch.isfinite(x) & (x != self.nan_value)
        x[~x_mask] = 0
        x = x.detach()
        x_key_padding_mask = ~x_mask.any(-1)

        x = self.enc_emb(x).permute(1, 0, 2)
        
        outputs = self.encoder(x, src_key_padding_mask=x_key_padding_mask).permute(
            1, 0, 2
        )

        # Seems to help a little, especially with extrapolating out of bounds
        steps = future_x.shape[1]
        mean = self.mean(outputs)[:, -steps:, :]
        log_sigma = self.std(outputs)[:, -steps:, :]
        
        sigma = self._min_std + (1 - self._min_std) * F.softplus(log_sigma)
        return torch.distributions.Normal(mean, sigma), {}

