import torch
from torch import nn
from torch.nn import functional as F

from ..util import mask_upper_triangular

class CrossAttention(nn.Module):
    """
    A single transformer,  using cross attention, like in the determistic encoder in attentive neural processes.
    """
    def __init__(self, x_dim, y_dim, attention_dropout=0, nhead=8, nlayers=8, hidden_size=32, min_std=0.01):
        super().__init__()
        self._min_std = min_std
        enc_x_dim = x_dim + y_dim

        self.v_encoder = nn.Linear(enc_x_dim, hidden_size)
        self.k_encoder = nn.Linear(x_dim, hidden_size)
        self.q_encoder = nn.Linear(x_dim, hidden_size)
        self.self_attn_k = torch.nn.MultiheadAttention(
            hidden_size, nhead, bias=False, dropout=attention_dropout
        )
        self.self_attn_q = torch.nn.MultiheadAttention(
            hidden_size, nhead, bias=False, dropout=attention_dropout
        )
        self.self_attn_v = torch.nn.MultiheadAttention(
            hidden_size, nhead, bias=False, dropout=attention_dropout
        )
        self.cross_attn = torch.nn.MultiheadAttention(
            hidden_size, nhead, bias=False, dropout=attention_dropout
        )

        encoder_norm = nn.LayerNorm(hidden_size)
        layer_enc = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            dim_feedforward=hidden_size*8,
            dropout=attention_dropout,
            nhead=nhead,
            # activation
        )
        self.transformer = nn.TransformerEncoder(
            layer_enc, num_layers=nlayers, norm=encoder_norm
        )
        self.mean = nn.Linear(hidden_size, y_dim)
        self.std = nn.Linear(hidden_size, y_dim)

    def forward(self, past_x, past_y, future_x, future_y=None):
        device = next(self.parameters()).device
        context = torch.cat([past_x, past_y], -1).detach()

        # Masks
        B, C, _ = past_x.shape
        past_causal_mask = mask_upper_triangular(C, device)
        B, T, _ = future_x.shape
        future_causal_mask = mask_upper_triangular(T, device)

        # Change feature size
        k = self.k_encoder(past_x).permute(1, 0, 2)
        q = self.q_encoder(future_x).permute(1, 0, 2)
        v = self.v_encoder(context).permute(1, 0, 2)

        # # Self attention with causal mask
        # v = self.self_attn_v(v, v, v, attn_mask=past_causal_mask)[0]
        # q = self.self_attn_q(q, q, q, attn_mask=future_causal_mask)[0]
        # k = self.self_attn_k(k, k, k, attn_mask=past_causal_mask)[0]

        # Cross attention
        h = self.cross_attn(query=q, key=k, value=v)[0]

        # Transformer
        outputs = self.transformer(h, mask=future_causal_mask)
        outputs = outputs.permute(1, 0, 2)

        # Head
        mean = self.mean(outputs)
        log_sigma = self.std(outputs)
        
        sigma = self._min_std + (1 - self._min_std) * F.softplus(log_sigma)
        return torch.distributions.Normal(mean, sigma), {}

