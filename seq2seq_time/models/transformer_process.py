import torch
from torch import nn
from torch.nn import functional as F

from ..util import mask_upper_triangular



class LatentEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_size=32,
        latent_dim=32,
        min_std=0.01,
        dropout=0,
        nhead=8,
        num_layers=2,
    ):
        super().__init__()
        self.enc_emb = nn.Linear(input_dim, hidden_size)

        encoder_norm = nn.LayerNorm(hidden_size)
        layer_enc = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            dim_feedforward=hidden_size*8,
            dropout=dropout,
            nhead=nhead,
            # activation
        )
        self.encoder = nn.TransformerEncoder(
            layer_enc, num_layers=num_layers, norm=encoder_norm
        )
        self.mean = nn.Linear(hidden_size, latent_dim)
        self.log_var = nn.Linear(hidden_size, latent_dim)
        self._min_std = min_std

    def forward(self, x, y):
        encoder_input = torch.cat([x, y], dim=-1)
        # Latent Encoder
        x = self.enc_emb(encoder_input)
        # Size([B, C, X]) -> Size([B, C, hidden_size])
        x = x.permute(1, 0, 2)  # (B,C,hidden_size) -> (C,B,hidden_size)
        # requires  (C, B, hidden_size)
        r = self.encoder(x)
        r = r.permute(1, 0, 2)  # (C,B,hidden_size) -> (B,C,hidden_size)
        r = r.mean(1)
        mean = self.mean(r)
        log_sigma = self.log_var(r)
        sigma = self._min_std + (1 - self._min_std) * torch.sigmoid(log_sigma * 0.5)
        dist = torch.distributions.Normal(mean, sigma)
        return dist



class Decoder(nn.Module):
    def __init__(
        self,
        x_size,
        y_size,
        hidden_size=32,
        latent_dim=32,
        num_layers=3,
        use_deterministic_path=True,
        min_std=0.01,
        nhead=8,
        dropout=0,
    ):
        super(Decoder, self).__init__()
        self.dec_emb = nn.Linear(x_size, hidden_size)
        self.z_emb = nn.Linear(latent_dim, hidden_size)
        layer_dec = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            dim_feedforward=hidden_size*8,
            dropout=dropout,
            nhead=nhead,
        )
        decoder_norm = nn.LayerNorm(hidden_size)
        self._decoder = nn.TransformerDecoder(
            layer_dec, num_layers=num_layers, norm=decoder_norm
        )
        self._mean = nn.Linear(hidden_size, y_size)
        self._std = nn.Linear(hidden_size, y_size)
        self._min_std = min_std

    def forward(self, z, future_x):
        # concatenate future_x and representation
        future_x = self.dec_emb(future_x)
        future_x = future_x.permute(1, 0, 2)

        z = self.z_emb(z)
        z = z.permute(1, 0, 2) 
        # requires  (C, B, hidden_size)

        # r = torch.cat([z, future_x], dim=-1)

        r = self._decoder(future_x, z)
        
        # [T, B, emb_dim] -> [B, T, emb_dim]
        r = r.permute(1, 0, 2).contiguous()

        # Get the mean and the variance
        mean = self._mean(r)
        log_sigma = self._std(r)

        # Bound or clamp the variance
        sigma = self._min_std + (1 - self._min_std) * F.softplus(log_sigma)

        dist = torch.distributions.Normal(mean, sigma)
        return dist
        
class TransformerProcess(nn.Module):
    # WIP trying one that encodes a dist
    # TODO autoregressive mask
    def __init__(self, x_size, y_size, hidden_size=16, latent_dim=32, nhead=8, nlayers=2, attention_dropout=0, min_std=0.01):
        super().__init__()
        self._min_std = min_std

        self._latent_encoder = LatentEncoder(
            x_size + y_size,
            hidden_size=hidden_size,
            latent_dim=latent_dim,
            num_layers=nlayers,
            dropout=attention_dropout,
            min_std=min_std,
            nhead=nhead,
        )

        self._decoder = Decoder(
            x_size,
            y_size,
            hidden_size=hidden_size,
            latent_dim=latent_dim,
            dropout=attention_dropout,
            min_std=min_std,
            num_layers=nlayers,
            nhead=nhead,
        )

    def forward(self, past_x, past_y, future_x, future_y=None):
        device = next(self.parameters()).device

        dist_prior = self._latent_encoder(past_x, past_y)

        if (future_y is not None):
            # If future_y is provided, we can create an auxilary loss
            # Making sure the encoded distribition from the past
            # Is as close as possible to the future
            x = torch.cat([past_x, future_x], 1)
            y = torch.cat([past_y, future_y], 1)
            dist_post = self._latent_encoder(x, y)

        if self.training:
            # Sample from latent space during training
            z = dist_prior.rsample()
        else:
            z = dist_prior.loc
        num_targets = future_x.size(1)
        z = z.unsqueeze(1).repeat(1, num_targets, 1)  # [B, T_target, H]

        dist = self._decoder(z, future_x)
        loss = None
        if future_y is not None:
            log_p = dist.log_prob(future_y).mean(-1)
            kl_loss = torch.distributions.kl_divergence(dist_post, dist_prior).mean(
                -1
            )  # [B, R].mean(-1)
            kl_loss = kl_loss[:, None].expand(log_p.shape)
            mse_loss = F.mse_loss(dist.loc, future_y, reduction="none")[
                :, : past_x.size(1)
            ].mean()
            loss = (kl_loss - log_p).mean()
        return dist, {'loss': loss}

