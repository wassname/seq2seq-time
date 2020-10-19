import torch
from torch import nn
from torch.nn import functional as F



class TransformerSeq2Seq(nn.Module):
    def __init__(self, x_size, y_size, hidden_size=16, nhead=8, nlayers=2, attention_dropout=0, min_std=0.01, nan_value=0):
        super().__init__()
        self._min_std = min_std
        self.nan_value = nan_value
        
        self.enc_emb = nn.Linear(x_size + y_size, hidden_size)
        self.dec_emb = nn.Linear(x_size, hidden_size)
        
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
        
        layer_dec = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            dim_feedforward=hidden_size*4,
            dropout=attention_dropout,
            nhead=nhead,
        )
        decoder_norm = nn.LayerNorm(hidden_size)
        self.decoder = nn.TransformerDecoder(
            layer_dec, num_layers=nlayers, norm=decoder_norm
        )
        self.mean = nn.Linear(hidden_size, y_size)
        self.std = nn.Linear(hidden_size, y_size)


    def forward(self, past_x, past_y, future_x, future_y=None):
        device = next(self.parameters()).device
        x = torch.cat([past_x, past_y], -1)

        # Masks
        future_mask = torch.isfinite(future_x) & (future_x!=self.nan_value)
        tgt_key_padding_mask = ~future_mask.any(-1)

        past_mask = torch.isfinite(x) & (x!=self.nan_value)
        src_key_padding_mask = ~past_mask.any(-1)# * float('-inf')
        
        # Embed
        x = self.enc_emb(x)
        # Size([B, C, X]) -> Size([B, C, hidden_dim])
        future_x = self.dec_emb(future_x)
        # Size([B, C, T]) -> Size([B, C, hidden_dim])

        x = x.permute(1, 0, 2)  # (B,C,hidden_dim) -> (C,B,hidden_dim)
        future_x = future_x.permute(1, 0, 2) 
        # requires  (C, B, hidden_dim)
        memory = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        # In transformers the memory and future_x need to be the same length. Lets use a permutation invariant agg on the context
        # Then expand it, so it's available as we decode, conditional on future_x
        # (C, B, emb_dim) -> (B, emb_dim) -> (T, B, emb_dim)
        # In transformers the memory and future_x need to be the same length. Lets use a permutation invariant agg on the context
        # Then expand it, so it's available as we decode, conditional on future_x
        memory = memory.max(dim=0, keepdim=True)[0].expand_as(future_x)

        outputs = self.decoder(future_x, memory, tgt_key_padding_mask=tgt_key_padding_mask)
        
        # [T, B, emb_dim] -> [B, T, emb_dim]
        outputs = outputs.permute(1, 0, 2).contiguous()
        # Size([B, T, emb_dim])
        mean = self.mean(outputs)
        log_sigma = self.std(outputs)
        sigma = self._min_std + (1 - self._min_std) * F.softplus(log_sigma)
        return torch.distributions.Normal(mean, sigma), {}

