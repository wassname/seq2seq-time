"""Recurrent Attentive Neural Process."""

import torch
from torch import nn
import torch.nn.functional as F
import math


class LSTMBlock(nn.Module):
    """Wrapper to return only lstm output."""
    def __init__(
        self,
        in_channels,
        out_channels,
        dropout=0,
        batchnorm=False,
        bias=False,
        num_layers=1,
    ):
        super().__init__()
        self._lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=out_channels,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bias=bias,
        )

    def forward(self, x):
        return self._lstm(x)[0]


class NPBlockRelu2d(nn.Module):
    """Block for Neural Processes."""

    def __init__(
        self, in_channels, out_channels, dropout=0, batchnorm=False, bias=False
    ):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout2d(dropout)
        self.norm = nn.BatchNorm2d(out_channels) if batchnorm else False

    def forward(self, x):
        # x.shape is (Batch, Sequence, Channels)
        # We pass a linear over it which operates on the Channels
        x = self.act(self.linear(x))

        # Now we want to apply batchnorm and dropout to the channels. So we put it in shape
        # (Batch, Channels, Sequence, None) so we can use Dropout2d & BatchNorm2d
        x = x.permute(0, 2, 1)[:, :, :, None]

        if self.norm:
            x = self.norm(x)

        x = self.dropout(x)
        return x[:, :, :, 0].permute(0, 2, 1)


class BatchMLP(nn.Module):
    """Apply MLP to the final axis of a 3D tensor (reusing already defined MLPs).

    Args:
        input: input tensor of shape [B,n,d_in].
        output_sizes: An iterable containing the output sizes of the MLP as defined
            in `basic.Linear`.
    Returns:
        tensor of shape [B,n,d_out] where d_out=output_size
    """

    def __init__(
        self, input_size, output_size, num_layers=2, dropout=0, batchnorm=False
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.initial = NPBlockRelu2d(
            input_size, output_size, dropout=dropout, batchnorm=batchnorm
        )
        self.encoder = nn.Sequential(
            *[
                NPBlockRelu2d(
                    output_size, output_size, dropout=dropout, batchnorm=batchnorm
                )
                for _ in range(num_layers - 2)
            ]
        )
        self.final = nn.Linear(output_size, output_size)

    def forward(self, x):
        x = self.initial(x)
        x = self.encoder(x)
        return self.final(x)


class Attention(nn.Module):
    def __init__(
        self,
        hidden_dim,
        attention_type,
        attention_layers=2,
        n_heads=8,
        x_dim=1,
        rep="mlp",
        dropout=0,
        batchnorm=False,
    ):
        super().__init__()
        self._rep = rep

        if self._rep == "mlp":
            self.batch_mlp_k = BatchMLP(
                x_dim,
                hidden_dim,
                attention_layers,
                dropout=dropout,
                batchnorm=batchnorm,
            )
            self.batch_mlp_q = BatchMLP(
                x_dim,
                hidden_dim,
                attention_layers,
                dropout=dropout,
                batchnorm=batchnorm,
            )

        self._W = torch.nn.MultiheadAttention(
            hidden_dim, n_heads, bias=False, dropout=dropout
        )
        self._attention_func = self._pytorch_multihead_attention

    def forward(self, k, v, q):
        if self._rep == "mlp":
            k = self.batch_mlp_k(k)
            q = self.batch_mlp_q(q)
        rep = self._attention_func(k, v, q)
        return rep

    def _pytorch_multihead_attention(self, k, v, q):
        # Pytorch multiheaded attention takes inputs if diff order and permutation
        q = q.permute(1, 0, 2)
        k = k.permute(1, 0, 2)
        v = v.permute(1, 0, 2)
        o = self._W(q, k, v)[0]
        return o.permute(1, 0, 2)


class LatentEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=32,
        latent_dim=32,
        self_attention_type="dot",
        n_encoder_layers=3,
        min_std=0.01,
        batchnorm=False,
        dropout=0,
        attention_dropout=0,
        use_self_attn=True,
        attention_layers=2,
        use_lstm=False,
    ):
        super().__init__()
        # self._input_layer = nn.Linear(input_dim, hidden_dim)
        if use_lstm:
            self._encoder = LSTMBlock(
                input_dim,
                hidden_dim,
                batchnorm=batchnorm,
                dropout=dropout,
                num_layers=n_encoder_layers,
            )
        else:
            self._encoder = BatchMLP(
                input_dim,
                hidden_dim,
                batchnorm=batchnorm,
                dropout=dropout,
                num_layers=n_encoder_layers,
            )
        if use_self_attn:
            self._self_attention = Attention(
                hidden_dim,
                self_attention_type,
                attention_layers,
                rep="identity",
                dropout=attention_dropout,
            )
        self._penultimate_layer = nn.Linear(hidden_dim, hidden_dim)
        self._mean = nn.Linear(hidden_dim, latent_dim)
        self._log_var = nn.Linear(hidden_dim, latent_dim)
        self._min_std = min_std
        self._use_lstm = use_lstm
        self._use_self_attn = use_self_attn

    def forward(self, x, y):
        encoder_input = torch.cat([x, y], dim=-1)

        # Pass final axis through MLP
        encoded = self._encoder(encoder_input)

        # Aggregator: take the mean over all points
        if self._use_self_attn:
            attention_output = self._self_attention(encoded, encoded, encoded)
            mean_repr = attention_output.mean(dim=1)
        else:
            mean_repr = encoded.mean(dim=1)

        # Have further MLP layers that map to the parameters of the Gaussian latent
        mean_repr = torch.relu(self._penultimate_layer(mean_repr))

        # Then apply further linear layers to output latent mu and log sigma
        mean = self._mean(mean_repr)
        log_var = self._log_var(mean_repr)

        sigma = self._min_std + (1 - self._min_std) * torch.sigmoid(log_var * 0.5)
        dist = torch.distributions.Normal(mean, sigma)
        return dist, log_var


class DeterministicEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        x_dim,
        hidden_dim=32,
        n_d_encoder_layers=3,
        self_attention_type="dot",
        cross_attention_type="dot",
        use_self_attn=True,
        attention_layers=2,
        batchnorm=False,
        dropout=0,
        attention_dropout=0,
        use_lstm=False,
    ):
        super().__init__()
        self._use_self_attn = use_self_attn
        # self._input_layer = nn.Linear(input_dim, hidden_dim)
        if use_lstm:
            self._d_encoder = LSTMBlock(
                input_dim,
                hidden_dim,
                batchnorm=batchnorm,
                dropout=dropout,
                num_layers=n_d_encoder_layers,
            )
        else:
            self._d_encoder = BatchMLP(
                input_dim,
                hidden_dim,
                batchnorm=batchnorm,
                dropout=dropout,
                num_layers=n_d_encoder_layers,
            )
        if use_self_attn:
            self._self_attention = Attention(
                hidden_dim,
                self_attention_type,
                attention_layers,
                rep="identity",
                dropout=attention_dropout,
            )
        self._cross_attention = Attention(
            hidden_dim,
            cross_attention_type,
            x_dim=x_dim,
            attention_layers=attention_layers,
        )

    def forward(self, past_x, past_y, future_x):
        # Concatenate x and y along the filter axes
        d_encoder_input = torch.cat([past_x, past_y], dim=-1)

        # Pass final axis through MLP
        d_encoded = self._d_encoder(d_encoder_input)

        if self._use_self_attn:
            d_encoded = self._self_attention(d_encoded, d_encoded, d_encoded)

        # Apply attention as mean aggregation
        h = self._cross_attention(past_x, d_encoded, future_x)

        return h


class Decoder(nn.Module):
    def __init__(
        self,
        x_dim,
        y_dim,
        hidden_dim=32,
        latent_dim=32,
        n_decoder_layers=3,
        use_deterministic_path=True,
        min_std=0.01,
        batchnorm=False,
        dropout=0,
        use_lstm=False,
    ):
        super(Decoder, self).__init__()
        self._future_transform = nn.Linear(x_dim, hidden_dim)
        if use_deterministic_path:
            hidden_dim_2 = 2 * hidden_dim + latent_dim
        else:
            hidden_dim_2 = hidden_dim + latent_dim

        if use_lstm:
            self._decoder = LSTMBlock(
                hidden_dim_2,
                hidden_dim_2,
                batchnorm=batchnorm,
                dropout=dropout,
                num_layers=n_decoder_layers,
            )
        else:
            self._decoder = BatchMLP(
                hidden_dim_2,
                hidden_dim_2,
                batchnorm=batchnorm,
                dropout=dropout,
                num_layers=n_decoder_layers,
            )
        self._mean = nn.Linear(hidden_dim_2, y_dim)
        self._std = nn.Linear(hidden_dim_2, y_dim)
        self._use_deterministic_path = use_deterministic_path
        self._min_std = min_std

    def forward(self, r, z, future_x):
        # concatenate future_x and representation
        x = self._future_transform(future_x)

        if self._use_deterministic_path:
            z = torch.cat([r, z], dim=-1)

        r = torch.cat([z, x], dim=-1)

        r = self._decoder(r)

        # Get the mean and the variance
        mean = self._mean(r)
        log_sigma = self._std(r)

        # Bound or clamp the variance
        sigma = self._min_std + (1 - self._min_std) * F.softplus(log_sigma)

        dist = torch.distributions.Normal(mean, sigma)
        return dist, log_sigma


class RANP(nn.Module):
    """Recurrent Attentive Neural Process for Sequential Data."""
    def __init__(
        self,
        x_dim,  # features in input
        y_dim,  # number of features in output
        hidden_dim=32,  # size of hidden space
        latent_dim=32,  # size of latent space
        n_latent_encoder_layers=2,
        n_det_encoder_layers=2,  # number of deterministic encoder layers
        n_decoder_layers=2,
        use_deterministic_path=True,
        min_std=0.01,  # To avoid collapse use a minimum standard deviation, should be much smaller than variation in labels
        dropout=0,
        use_self_attn=True,
        attention_dropout=0,
        batchnorm=False,
        attention_layers=2,
        use_rnn=True,  # use RNN/LSTM
        use_lstm_le=False,  # use another LSTM in latent encoder instead of MLP
        use_lstm_de=False,  # use another LSTM in determinstic encoder instead of MLP
        use_lstm_d=False,  # use another lstm in decoder instead of MLP
        **kwargs,
    ):

        super().__init__()

        self._use_rnn = use_rnn

        if self._use_rnn:
            self._lstm = nn.LSTM(
                input_size=x_dim,
                hidden_size=hidden_dim,
                num_layers=attention_layers,
                dropout=dropout,
                batch_first=True,
            )
            x_dim = hidden_dim

        self._latent_encoder = LatentEncoder(
            x_dim + y_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            n_encoder_layers=n_latent_encoder_layers,
            attention_layers=attention_layers,
            dropout=dropout,
            use_self_attn=use_self_attn,
            attention_dropout=attention_dropout,
            batchnorm=batchnorm,
            min_std=min_std,
            use_lstm=use_lstm_le,
        )

        self._deterministic_encoder = DeterministicEncoder(
            input_dim=x_dim + y_dim,
            x_dim=x_dim,
            hidden_dim=hidden_dim,
            n_d_encoder_layers=n_det_encoder_layers,
            attention_layers=attention_layers,
            use_self_attn=use_self_attn,
            dropout=dropout,
            batchnorm=batchnorm,
            attention_dropout=attention_dropout,
            use_lstm=use_lstm_de,
        )

        self._decoder = Decoder(
            x_dim,
            y_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            dropout=dropout,
            batchnorm=batchnorm,
            min_std=min_std,
            n_decoder_layers=n_decoder_layers,
            use_deterministic_path=use_deterministic_path,
            use_lstm=use_lstm_d,
        )
        self._use_deterministic_path = use_deterministic_path

    def forward(self, past_x, past_y, future_x, future_y=None):

        if self._use_rnn:
            # see https://arxiv.org/abs/1910.09323 where x is substituted with h = RNN(x)
            # x need to be provided as [B, T, H]
            S = past_x.shape[1]
            x = torch.cat([past_x, future_x], 1)
            x, _ = self._lstm(x)
            past_x = x[:, :S]
            future_x = x[:, S:]
            # future_x, _ = self._lstm(future_x)
            # past_x, _ = self._lstm(past_x)

        dist_prior, log_var_prior = self._latent_encoder(past_x, past_y)

        if (future_y is not None):
            dist_post, log_var_post = self._latent_encoder(future_x, future_y)

        if self.training:
            z = dist_prior.rsample()
        else:
            z = dist_prior.loc

        num_targets = future_x.size(1)
        z = z.unsqueeze(1).repeat(1, num_targets, 1)  # [B, T_target, H]

        if self._use_deterministic_path:
            r = self._deterministic_encoder(
                past_x, past_y, future_x
            )  # [B, T_target, H]
        else:
            r = None

        dist, log_sigma = self._decoder(r, z, future_x)
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
        return dist, {'loss':loss}

