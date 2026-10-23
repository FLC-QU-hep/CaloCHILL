import math

import torch
import torch.nn as nn
from models.common import Embedder
from models.VAE.down_up_sample import CrossAttnDownsample, CrossAttnUpsample
from models.VAE.encoder_decoder import DecoderBlock, EncoderBlock
from models.VAE.softpooling_downsampling import PointCloudDownsamplingNoAttn
from utils.compr_utils import (
    Compute_log_probabilities,
    get_activation,
)


def within_layer_pos(tokens, PAD_ID=None, num_classes=None):
    """For each position, count how many earlier valid positions in the same row
    share its token value (0 for padding positions). O(B * N * num_classes) time
    and memory instead of the O(B * N^2) an all-pairs comparison would need --
    num_classes (here calorimeter_layers + 1) is a small constant independent of N.
    Verified bit-identical to the previous all-pairs implementation.
    """
    B, N = tokens.shape
    device = tokens.device

    if PAD_ID is not None:
        valid = tokens != PAD_ID
    else:
        valid = torch.ones(B, N, dtype=torch.bool, device=device)

    if num_classes is None:
        num_classes = int(tokens.max().item()) + 1

    tokens_long = tokens.long()
    one_hot = nn.functional.one_hot(tokens_long, num_classes=num_classes).to(
        torch.int32
    )
    one_hot = one_hot * valid.unsqueeze(
        -1
    )  # padding never counts as a match for anyone

    # exclusive cumulative count per class: occurrences strictly before this position
    cum_before = torch.cumsum(one_hot, dim=1) - one_hot

    layer_ordering = torch.gather(
        cum_before, dim=2, index=tokens_long.unsqueeze(-1)
    ).squeeze(-1)

    layer_ordering = layer_ordering * valid

    return layer_ordering.to(torch.int32)


# Transformer VAE
class Compressor(nn.Module):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()
        self.gaussian_prior = getattr(cfg, "gaussian_prior", True)
        input_dim = cfg.input_feats
        self.hidden_dim = cfg.hidden_dim
        n_layers = cfg.n_layers
        self.n_layers = n_layers
        self.out_refinement_layers = cfg.out_refinement_layers
        self.z_dim = cfg.z_dim
        n_heads = cfg.num_heads
        ctx_dim = cfg.cond_feats
        self.activation = get_activation(cfg.activation)
        self.embed_kwargs = cfg.to_dict(["embed_kwargs"])["embed_kwargs"]
        self.downsample_points = cfg.p_dim
        self.downsampling_flag = cfg.downsampling_flag
        self.n_gaussian_mixture = cfg.n_gaussian_mixture
        self.include_time = cfg.include_time
        self.calorimeter_layers = cfg.calorimeter_layers
        self.PAD_ID = 0
        self.log_prob_class = Compute_log_probabilities()

        # defining number of neighbors for sparse attn
        self.neighbors_size = (
            math.ceil(cfg.max_outputs / self.downsample_points)
            if cfg.neighbors is None
            else cfg.neighbors
        )

        self.pos_embedding = nn.Embedding(
            num_embeddings=int(cfg.max_points_per_layer),
            embedding_dim=self.hidden_dim,
            padding_idx=self.PAD_ID,
        )

        if self.embed_kwargs["activate_fourier_layer"]:
            dim_final = input_dim + input_dim * self.embed_kwargs["N_freqs"] * len(
                self.embed_kwargs["periodic_fns"]
            )

            self.input_proj = nn.Sequential(
                Embedder(input_dim, **self.embed_kwargs),
                nn.Linear(dim_final, self.hidden_dim // 2),
                nn.GELU(),
                nn.Linear(self.hidden_dim // 2, self.hidden_dim),
            )
        else:
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, self.hidden_dim),
                self.activation,
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )

        # Conditioning embeddings
        self.ctx_proj = nn.Sequential(
            nn.Linear(ctx_dim, self.hidden_dim),
            self.activation,
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.embed_layers = nn.Embedding(
            num_embeddings=self.calorimeter_layers + 1,
            embedding_dim=self.hidden_dim,
            padding_idx=self.PAD_ID,
        )

        if self.include_time:
            self.embed_mint = nn.Sequential(
                nn.Linear(self.calorimeter_layers, self.hidden_dim),
                self.activation,
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
        if self.downsampling_flag:
            # Downsampling attention
            downsampling_points = []
            downsampling_points += [
                self.downsample_points for _ in range(cfg.downsamp_layers)
            ]

            if cfg.downsampling_method == "CrossAttn":
                self.down_attn = nn.ModuleList(
                    [
                        CrossAttnDownsample(
                            self.hidden_dim,
                            downsampling_points[i],
                            n_heads,
                            max_outputs=cfg.max_points,
                        )
                        for i in range(cfg.downsamp_layers)
                    ]
                )
            elif cfg.downsampling_method == "SoftPooling":
                self.down_attn = nn.ModuleList(
                    [
                        PointCloudDownsamplingNoAttn(
                            embed_dim=self.hidden_dim,
                            pooled_points=self.downsample_points,
                            latent_dim=self.hidden_dim,
                            num_latents=self.downsample_points,
                        )
                        for i in range(cfg.downsamp_layers)
                    ]
                )
            else:
                raise ValueError("Unknown downsampling method")

        # Encoder blocks
        self.encoder_layers = nn.ModuleList(
            [
                EncoderBlock(
                    self.hidden_dim,
                    n_heads,
                    max_len=self.downsample_points,
                )
                for _ in range(n_layers)
            ]
        )

        self.decoder = nn.ModuleList(
            [
                DecoderBlock(
                    self.hidden_dim,
                    self.z_dim,
                    n_heads,
                    self.activation,
                    downsample_points=self.downsample_points
                    if self.downsampling_flag
                    else cfg.max_points,
                    max_outputs=cfg.max_points,
                    gaussian_mixture=self.n_gaussian_mixture,
                )
                for _ in range(n_layers)
            ]
        )

        if self.downsampling_flag:
            self.conv_layer = nn.ModuleList(
                [
                    CrossAttnUpsample(
                        self.hidden_dim,
                        cfg.max_points,
                        self.downsample_points,
                        num_heads=n_heads,
                    )
                    for _ in range(n_layers)
                ]
            )

        self.out_refinement = nn.ModuleList(
            [
                EncoderBlock(
                    self.hidden_dim,
                    n_heads,
                    max_len=cfg.max_points,
                )
                for _ in range(self.out_refinement_layers)
            ]
        )

        self.output_proj = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, input_dim),
            self.activation,
            nn.Linear(input_dim, input_dim),
        )

    def embed_conditioning(self, ctx, y, mint_per_layer=None, padding=None):
        """Embed context and label information."""
        ctx_emb, y_emb, mint_emb = None, None, None
        if ctx is not None:
            ctx_emb = self.ctx_proj(ctx).unsqueeze(1)  # [B, 1, 2] --> [B, 1, hidden]

        if y is not None:
            tokens = y[:, :, 0].int() + 1
            if padding is not None:
                tokens = tokens.masked_fill(padding, self.PAD_ID)

            internal_pos = within_layer_pos(
                tokens, self.PAD_ID, num_classes=self.calorimeter_layers + 1
            )
            pos_emb = self.pos_embedding(internal_pos).float()
            y_emb = self.embed_layers(tokens).float()  # [B, N, 1] --> [B, N, hidden]

        if self.include_time and mint_per_layer is not None:
            mint_emb = self.embed_mint(
                mint_per_layer.unsqueeze(1)
            )  # [B, 1, 78] --> [B, 1, hidden]

        return ctx_emb, y_emb, mint_emb, pos_emb

    def encoder(
        self,
        x,
        ctx=None,
        y_emb=None,
        mint_per_layer=None,
        masks=None,
        layer=None,
        effective_p_dim: int = None,
    ):
        # Bottom-up encoding
        h = self.input_proj(x)

        if mint_per_layer is not None:
            h = h + mint_per_layer

        if y_emb is not None:
            h = h + y_emb

        if self.downsampling_flag:
            for mod_attn in self.down_attn:
                h = mod_attn(
                    h,
                    ctx_emb=ctx,
                    block_mask=masks[0],
                    effective_p_dim=effective_p_dim,
                )

        enc_outs = []
        for enc in self.encoder_layers:
            h = enc(
                h,
                block_mask=masks[1],
                effective_p_dim=effective_p_dim,
            )
            enc_outs.append(h)
        return enc_outs

    def forward(
        self,
        x,
        ctx=None,
        y=None,
        mint_per_layer=None,
        padding=None,
        masks=None,
        effective_p_dim: int = None,
    ):
        """
        x: [B, N, input_dim]
        c: context tensor [B, ctx_dim]
        y: label tensor [B, N, 1] or [B, N, label_dim]
        """
        B, N, D = x.shape
        layer = y.clone()
        layer[padding] = -1
        ctx_emb, y_emb, mint_emb, pos_emb = self.embed_conditioning(
            ctx,
            y,
            mint_per_layer=mint_per_layer,
            padding=padding,
        )

        enc_outs = self.encoder(
            x,
            ctx_emb,
            y_emb,
            mint_per_layer=mint_emb,
            masks=masks,
            layer=layer,
            effective_p_dim=effective_p_dim,
        )
        if torch.isnan(enc_outs[0]).any():
            print("NaN detected in enc_outs!")

        kls, mus, logvars = [], [], []
        posteriors, kl_integral, all_eps = [], [], []
        z_list = []
        z_above, q, o = None, None, None
        # Top-down decoding with conditioning
        for idx, dec in enumerate(reversed(self.decoder)):
            enc_feat = enc_outs[-idx - 1]

            # y_emb is [B, N, d] at full resolution; mean-pool to [B, 1, d] so it
            # broadcasts over the p_dim latent tokens without a shape mismatch.
            y_emb_global = (
                y_emb.mean(dim=1, keepdim=True) if y_emb is not None else None
            )
            mu, logvar, pi = dec.compute_posterior(
                enc_feat,
                c=ctx_emb,
                layer=y_emb_global,
                mint_per_l=mint_emb,
            )
            z, eps = dec.sample(mu, logvar, pi)

            logqz = dec.log_p_var_normal(z, mu, logvar, pi)
            # this won't work with GMM
            kl_integral.append(self.log_prob_class.compute_kl(mu, logvar))

            mu_p, logvar_p = None, None
            if self.gaussian_prior:
                logpz = dec.log_p_normal(z)
            else:
                mu_p, logvar_p = dec.compute_prior(
                    z_above=z_above,
                    c=ctx_emb,
                    y=y_emb,
                    mint_per_layer=mint_emb,
                    effective_p_dim=effective_p_dim,
                )
                logpz = dec.log_p_var_normal(z, mu_p, logvar_p, pi=0)

            z_above = z
            kl = logqz - logpz
            if kl.mean() > 2000:
                print(f"Warning: KL > 1000 at layer {idx}: mean={kl.mean().item():.1f}")
            kls.append(kl)

            o = dec(
                z,
                o=o if idx != 0 else None,
                c=ctx_emb,
                y=y_emb,
                mint_per_layer=mint_emb,
                block_mask=masks[1],
                effective_p_dim=effective_p_dim,
            )

            if self.downsampling_flag:
                q = self.conv_layer[idx](
                    o,
                    N,
                    q=q,
                    y=y_emb,
                    ctx=ctx_emb
                    if idx == 0
                    else None,  # inject cond only at the first latent
                    pos_emb=pos_emb,
                    block_mask=masks[2],
                    effective_p_dim=effective_p_dim,
                )

            mus.append(mu)
            logvars.append(logvar)

            if self.n_gaussian_mixture > 1:
                eps = eps.unsqueeze(2).expand(-1, -1, mu.shape[2], -1)
            posteriors.append(
                torch.cat(
                    [eps.unsqueeze(-1), mu.unsqueeze(-1), logvar.unsqueeze(-1)], dim=-1
                ).unsqueeze(1)
            )
            all_eps.append(eps.unsqueeze(1))
            z_list.append(z.unsqueeze(1))

        for ref in self.out_refinement:
            q = ref(
                q,
                block_mask=masks[3],
            )
        recon = self.output_proj(q)
        return {
            "set": recon,
            "mu": mus[-1].mean(dim=1),
            "logvar": logvars[-1].mean(dim=1),
            "z": torch.cat(z_list, dim=1),
            "all_eps": torch.cat(all_eps, dim=1),
            "posteriors": torch.cat(posteriors, dim=1),
            "kls": kls,
            "kls_integral": kl_integral,
            "var_o": torch.var(o, dim=0).mean(),
        }

    @torch.no_grad()
    def encode_for_latent(
        self,
        x,
        ctx=None,
        y=None,
        mint_per_layer=None,
        padding=None,
        masks=None,
        effective_p_dim: int = None,
        return_params: bool = False,
    ):
        """
        Encode input x into the hierarchical latent posterior parameters used for latent diffusion.

        Args:
            x: tensor [B, N, input_dim]
            ctx: context tensor [B, ctx_dim]
            y: label tensor [B, N, 1] or [B, N, label_dim]
            return_params: return the posterior parameters (mu, log_var) instead of
                a posterior sample z = mu + exp(log_var / 2) * eps. Single-Gaussian
                posteriors only.
        Returns:
            - z [B, P, z_dim * n_layers], or (mu, log_var) of that shape if return_params
        """
        layer = y.clone()
        layer[padding] = -1

        ctx_emb, y_emb, mint_emb, _ = self.embed_conditioning(
            ctx,
            y,
            mint_per_layer=mint_per_layer,
            padding=padding,
        )
        enc_outs = self.encoder(
            x,
            ctx_emb,
            y_emb,
            mint_per_layer=mint_emb,
            masks=masks,
            layer=layer,
            effective_p_dim=effective_p_dim,
        )

        z_list = []
        mu_list, logvar_list = [], []
        for idx, dec in enumerate(reversed(self.decoder)):
            enc_feat = enc_outs[-idx - 1]

            y_emb_global = (
                y_emb.mean(dim=1, keepdim=True) if y_emb is not None else None
            )
            mu, log_var, pi = dec.compute_posterior(
                enc_feat,
                c=ctx_emb,
                layer=y_emb_global,
                mint_per_l=mint_emb,
            )
            if return_params:
                if dec.gaussian_mixture != 1:
                    raise NotImplementedError(
                        "encode_for_latent(return_params=True) needs a single-Gaussian "
                        f"posterior, got a mixture of {dec.gaussian_mixture}"
                    )
                mu_list.append(mu)
                logvar_list.append(log_var)
                continue
            z, _ = dec.sample(mu, log_var, pi)
            z_list.append(z)

        if return_params:
            return torch.cat(mu_list, dim=-1), torch.cat(logvar_list, dim=-1)
        return torch.cat(z_list, dim=-1)

    @torch.no_grad()
    def sample(
        self,
        B=1,
        N=None,
        c=None,
        y=None,
        padding=None,
        masks=None,
        effective_p_dim: int = None,
        mint_per_layer=None,
        device=None,
        z=None,
    ):
        """
        Generate new points from the model.

        Args:
            B: batch size
            N: number of output points
            c: optional conditioning [B, ctx_dim]
            y: optional label embeddings [B, N, y_dim]
            padding: optional mask for valid points in y [B, N]: False for valid
            deterministic: if True, use mu instead of random eps
        Returns:
            recon: [B, N, input_dim]
        """
        layer = y.clone()
        layer[padding] = -1

        if device is None:
            device = next(self.parameters()).device

        if self.downsampling_flag:
            if effective_p_dim is not None:
                n_down = effective_p_dim
            else:
                n_down = self.downsample_points
        else:
            n_down = N if N is not None else 64

        ctx_emb, y_emb, mint_emb, pos_emb = self.embed_conditioning(
            c,
            y,
            mint_per_layer=mint_per_layer,
            padding=padding,
        )

        if z is None:
            z1 = torch.randn(B, n_down, self.z_dim).to(device)
        else:
            z1 = z[:, :, : self.z_dim]

        z_above, q, o = None, None, None
        z_all = []
        for i, dec in enumerate(reversed(self.decoder)):
            if i == 0:
                z = z1
            else:
                mu_p, logvar_p = dec.compute_prior(
                    z_above=z_above,
                    c=ctx_emb,
                    y=y_emb,
                    effective_p_dim=effective_p_dim,
                )
                z = mu_p + torch.randn_like(mu_p) * torch.exp(0.5 * logvar_p)
                z_above = z

            z_all.append(z)
            o = dec(
                z,
                o=o if i != 0 else None,
                c=ctx_emb,
                y=y_emb,
                mint_per_layer=mint_emb,
                block_mask=masks[1],
                effective_p_dim=effective_p_dim,
            )

            if self.downsampling_flag:
                q = self.conv_layer[i](
                    o,
                    y.shape[1],
                    q=q,
                    y=y_emb,
                    ctx=ctx_emb,
                    pos_emb=pos_emb,
                    block_mask=masks[2],
                    effective_p_dim=effective_p_dim,
                )
        for ref in self.out_refinement:
            q = ref(
                q,
                block_mask=masks[3],
            )
        # z_sampled: [B, n_layers, p_dim, z_dim] matching the shape of the posterior z
        z_sampled = torch.stack(z_all, dim=1)
        return self.output_proj(q), z_sampled


if __name__ == "__main__":
    # test sample for latent
    model = Compressor()
    x = torch.randn(2, 2048, 4)
    z = model.encode_for_latent(x, y=x[:, :, :1])
    out = model.sample_for_latent(z, y=x[:, :, :1])
    print(out.shape)
