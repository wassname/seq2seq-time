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
        self.mean = nn.Linear(hidden_size*3, latent_dim)
        self.log_var = nn.Linear(hidden_size*3, latent_dim)
        self._min_std = min_std

    def forward(self, x, y):
        encoder_input = torch.cat([x, y], dim=-1)
        # Latent Encoder
        x = self.enc_emb(encoder_input) # Size([B, S, X]) -> Size([B, S, hidden_size])        
        x = x.permute(1, 0, 2)  # (B,S,hidden_size) -> (S,B,hidden_size)

        # autoregressive mask
        device = next(self.parameters()).device
        N = x.shape[0]
        mask = mask_upper_triangular(N, device)

        r = self.encoder(x, mask=mask)
        r = r.permute(1, 0, 2)  # (S,B,hidden_size) -> (B,S,hidden_size)

        # Aggregation (max/mean/last)
        r_mean = r.mean(1)  #  (B,S,hidden_size) ->  (B,hidden_size)
        r_last = r[:, -1, :] 
        r_max = r.max(1)[0]
        r = torch.cat([r_mean, r_last, r_max], -1)
        
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

    def forward(self, z, x):

        # (B, S, latent_size) -> (B, S, H) -> (S, B, H)
        x = self.dec_emb(x).permute(1, 0, 2)

        # (B, S, latent_size) -> (B, S, H) -> (S, B, H)
        z = self.z_emb(z).permute(1, 0, 2) 

        # autoregressive mask
        device = next(self.parameters()).device
        N = x.shape[0]
        mask = mask_upper_triangular(N, device)

        r = self._decoder(x, z, tgt_mask=mask)
        
        # [S, B, H] -> [B, S, H]
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
    def __init__(self, x_size, y_size, hidden_size=64, latent_dim=32, nhead=8, nlayers=4, dropout=0, min_std=0.01):
        super().__init__()
        self._min_std = min_std

        self._latent_encoder = LatentEncoder(
            x_size + y_size,
            hidden_size=hidden_size,
            latent_dim=latent_dim,
            num_layers=nlayers,
            dropout=dropout,
            min_std=min_std,
            nhead=nhead,
        )

        self._decoder = Decoder(
            x_size,
            y_size,
            hidden_size=hidden_size,
            latent_dim=latent_dim,
            dropout=dropout,
            min_std=min_std,
            num_layers=nlayers,
            nhead=nhead,
        )

    def forward(self, past_x, past_y, future_x, future_y=None):
        device = next(self.parameters()).device

        dist_prior = self._latent_encoder(past_x, past_y)

        if (future_y is not None):
            x = torch.cat([past_x, future_x], 1)
            y = torch.cat([past_y, future_y], 1)
            dist_post = self._latent_encoder(x, y)

        if self.training:
            # Sample from latent space during training
            z = dist_prior.rsample()
        else:
            z = dist_prior.loc
        num_targets = future_x.size(1)
        z = z.unsqueeze(1).repeat(1, num_targets, 1)  # [B, S_target, H]

        dist_out = self._decoder(z, future_x)
        loss = None
        if future_y is not None:
            # Make sure output dist matches label
            log_p = dist_out.log_prob(future_y).mean(-1)

            # Making sure the encoded distribition from the past is as close as possible to the future
            kl_loss = torch.distributions.kl_divergence(dist_post, dist_prior).mean(
                -1
            )  # [B, R].mean(-1)
            kl_loss = kl_loss[:, None].expand(log_p.shape)
            loss = (kl_loss - log_p).mean()
        return dist_out, {'loss': loss}

