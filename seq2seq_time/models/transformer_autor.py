from tqdm.auto import tqdm
from torch import nn
import torch
from torch.nn import functional as F


import fast_transformers
from fast_transformers.builders import TransformerEncoderBuilder

class TransformerAutoR(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_out_size=256, nlayers=8, n_heads=8, use_lstm=False, attention_dropout=0, dropout=0, min_std=0.01):
        super().__init__()
        self._min_std = min_std
        self.use_lstm = use_lstm
        hidden_out_size = hidden_out_size//n_heads

        x_size = x_dim + y_dim
        
        # TODO embedd both X's the same
        if use_lstm:            
            self.x_emb = LSTMBlock(x_size, x_size)
        
        self.enc_emb = nn.Linear(x_size, hidden_out_size*n_heads)
        self.encoder = fast_transformers.builders.TransformerEncoderBuilder.from_kwargs(
            attention_type="causal-linear",
            n_layers=nlayers,
            n_heads=n_heads,
            feed_forward_dimensions=hidden_out_size*8*n_heads,
            query_dimensions=hidden_out_size,
            value_dimensions=hidden_out_size,
            attention_dropout=attention_dropout,
            dropout=dropout,
        ).get()
        self.mean = nn.Linear(hidden_out_size*n_heads, y_dim)
        self.std = nn.Linear(hidden_out_size*n_heads, y_dim)

    def forward(self, past_x, past_y, future_x, future_y=None, mask_context=True, mask_target=True):
        device = next(self.parameters()).device
        B, S, _ = future_x.shape
        future_y_fake = past_y[:, -1:, :].repeat(1, S, 1).to(device)
        # future_y_fake = (
        #     torch.ones(past_y.shape[0], future_x.shape[1], past_y.shape[2]).float().to(device) * 0
        # )
        context = torch.cat([past_x, past_y], -1)
        target = torch.cat([future_x, future_y_fake], -1)
        x = torch.cat([context, target * 1], 1).detach()        
        
        # LSTM
        if self.use_lstm:  
            x = self.x_emb(x)
            # Size([B, T, Y]) -> Size([B, T, Y])
        
        # Embed
        x = self.enc_emb(x)
        
        # requires  (B, C, hidden_dim)
        steps = past_y.shape[1]
        N = x.shape[1]
        mask = fast_transformers.masking.TriangularCausalMask(N, device=device)
        outputs = self.encoder(x, attn_mask=mask)[:, steps:, :]
        
        # Size([B, T, emb_dim])
        mean = self.mean(outputs)
        log_sigma = self.std(outputs)
        sigma = self._min_std + (1 - self._min_std) * F.softplus(log_sigma)
        y_dist = torch.distributions.Normal(mean, sigma)

        return (
            y_dist,
            {}
        )


