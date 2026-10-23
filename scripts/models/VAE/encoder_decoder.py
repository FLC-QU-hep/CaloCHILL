import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import BlockMask
from utils.flex_attn import (
    MHFlexEncoderLayer,
)


class EncoderBlock(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        max_len=3008,
        dropout=0.1,
        qk_norm=False,
        qk_norm_affine=True,
        track_qk_scale=False,
    ):
        super().__init__()
        self.layer = MHFlexEncoderLayer(
            d_model,
            nhead,
            max_len=max_len,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            qk_norm=qk_norm,
            qk_norm_affine=qk_norm_affine,
            track_qk_scale=track_qk_scale,
        )
        self.print_mask = True

    def forward(
        self,
        x,
        block_mask: BlockMask = None,
        effective_p_dim: int = None,
    ):
        out = self.layer(
            x,
            block_mask=block_mask,
            effective_p_dim=effective_p_dim,
        )

        self.print_mask = False
        return out


class DecoderBlock(nn.Module):
    def __init__(
        self,
        d_model,
        z_dim,
        nhead,
        act,
        downsample_points=1024,
        max_outputs=6704,
        dim_feedforward=None,
        dropout=0.1,
        gaussian_mixture=1,
    ):
        """
        Decoder block with latent sampling and optional conditioning (context + label).
        """
        super().__init__()
        self.d_model = d_model
        self.downsample_points = downsample_points
        self.z_dim = z_dim
        self.gaussian_mixture = gaussian_mixture
        self.c = 0.9189385332  # normalization constant for Gaussian log-prob
        self.print_mask_posterior = True
        self.print_mask_decoder = True
        self.print_mask_prior = True
        self.z_linear = nn.Linear(z_dim, d_model)
        # Prior network
        self.prior_net = nn.Sequential(
            nn.Linear(d_model, d_model),
            act,
            nn.Linear(
                d_model,
                z_dim * 2
                if gaussian_mixture == 1
                else (z_dim * 2 * gaussian_mixture + gaussian_mixture),
            ),
        )
        # Posterior network
        if self.gaussian_mixture != 1:
            self.to_pi = nn.Linear(d_model, self.gaussian_mixture)
            self.to_mu = nn.Linear(d_model, self.gaussian_mixture * z_dim)
            self.to_log_var = nn.Linear(d_model, self.gaussian_mixture * z_dim)
        else:
            self.pi = 0
            self.to_mu = nn.Sequential(
                nn.Linear(d_model, z_dim), act, nn.Linear(z_dim, z_dim)
            )
            self.to_log_var = nn.Sequential(
                nn.Linear(d_model, z_dim), act, nn.Linear(z_dim, z_dim)
            )

        self.forward_linear = nn.Linear(z_dim, d_model)
        self.gamma_mlp = nn.Sequential(nn.Linear(d_model, d_model))
        self.beta_mlp = nn.Sequential(nn.Linear(d_model, d_model))

        if dim_feedforward is None:
            dim_feedforward = d_model * 4

        # Transformer for decoding
        self.layer = MHFlexEncoderLayer(
            d_model,
            nhead,
            max_len=self.downsample_points,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

    def compute_posterior(
        self,
        x,
        c=None,
        layer=None,
        mint_per_l=None,
    ):
        """
        x: encoder features [B, N, d_model]
        """
        h = x

        if c is not None:
            h = h + c
        if layer is not None:
            h = h + layer
        if mint_per_l is not None:
            h = h + mint_per_l

        mu, log_var = self.to_mu(h), self.to_log_var(h)
        if self.gaussian_mixture == 1:
            mu = 2.0 * torch.tanh(mu / 2.0)
            log_var = torch.clamp(log_var, min=-2, max=2)
            return mu, log_var, self.pi
        else:
            pi = torch.softmax(self.to_pi(x), dim=-1)
            mu = mu.view(*mu.shape[:-1], self.gaussian_mixture, self.z_dim)
            log_var = log_var.view(
                *log_var.shape[:-1], self.gaussian_mixture, self.z_dim
            )
            return mu, log_var, pi

    def compute_prior(
        self,
        z_above,
        c,
        y=None,
        mint_per_layer=None,
        effective_p_dim: int = None,
    ):
        """
        c: context embedding
        y: label embedding
        """

        cond = c
        if y is not None:
            cond = (
                self._concat_condition(y, c, mint_per_layer)
                .mean(dim=1, keepdim=True)
                .repeat(1, self.downsample_points, 1)
            )
            if effective_p_dim is not None:
                cond = cond[:, :effective_p_dim, :]

        if z_above is not None:
            cond = self.z_linear(z_above) + cond

        stats = self.prior_net(cond)
        mu_p, log_var_p = stats.chunk(2, dim=-1)
        mu_p = 2.0 * torch.tanh(mu_p / 2.0)
        log_var_p = torch.clamp(log_var_p, min=-2, max=2)

        return mu_p, log_var_p

    def sample(self, mu, logvar, pi=0):
        if self.gaussian_mixture != 1:
            B, N, K, D = mu.shape
            # Sample mixture component
            cat = torch.distributions.Categorical(probs=pi)
            component = cat.sample()
            # I select mu and logvar from the gaussian I chose
            mu = mu[
                torch.arange(B).unsqueeze(1), torch.arange(N).unsqueeze(0), component, :
            ]
            logvar = logvar[
                torch.arange(B).unsqueeze(1), torch.arange(N).unsqueeze(0), component, :
            ]

        # sampling
        eps = torch.randn_like(mu)
        z = mu + torch.exp(logvar / 2.0) * eps

        return z, eps

    def log_p_var_normal(self, samples, mu, logvar, pi=0):
        if self.gaussian_mixture > 1:
            samples = samples.unsqueeze(2)  # [B, N, 1, D]

        log_probs = -0.5 * (
            (samples - mu) ** 2 / torch.exp(logvar) + logvar + 2 * self.c
        )
        log_probs = log_probs.sum(dim=-1)  # [B, N, K] or [B, N]

        if self.gaussian_mixture > 1:
            weighted = log_probs + torch.log(pi + 1e-10)  # [B, N, K]
            log_probs = torch.logsumexp(weighted, dim=-1)  # [B, N, K]
        return log_probs

    def log_p_normal(self, samples):
        return (-0.5 * (samples**2 + 2 * self.c)).sum(dim=-1)

    def compute_kl(self, mu, logvar):
        return -0.5 * (logvar - torch.exp(logvar) - mu.pow(2) + 1).sum(dim=-1)

    def _concat_condition(self, y, c, mint_per_layer=None):
        """
        Concatenate conditioning vectors if provided.
        """
        cond = y
        if c is not None:
            cond = torch.cat([cond, c], dim=1)  # broadcast along sequence

        if mint_per_layer is not None:
            cond = torch.cat([cond, mint_per_layer], dim=1)
        return cond

    def forward(
        self,
        z,
        o=None,
        c=None,
        y=None,
        mint_per_layer=None,
        block_mask: BlockMask = None,
        effective_p_dim: int = None,
    ):
        """
        o: current output [B, N, z_dim]
        c: context [B, 1, cond_dim]
        mint: min time per layer [B, 1, cond_dim]
        y: label [B, N, y_dim]
        """
        z = self.forward_linear(z)
        if o is not None:
            z = z + o

        if y is not None:
            # FiLM conditioning
            layer_global = y.mean(dim=1)  # [B, D]
            gamma = self.gamma_mlp(layer_global).unsqueeze(1)  # [B, 1, D]
            beta = self.beta_mlp(layer_global).unsqueeze(1)  # [B, 1, D]
            z = gamma * z + beta

        if c is not None:
            z = z + c

        if mint_per_layer is not None:
            z = z + mint_per_layer

        z = self.layer(
            z,
            block_mask=block_mask,
            effective_p_dim=effective_p_dim,
        )
        self.print_mask_decoder = False
        return z
