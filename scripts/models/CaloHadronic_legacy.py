"""CaloHadronic point-cloud EDM diffusion, Oct 2025 version (pion-clouds commit c034126).

The model of run HGx9_ecal_hcal_withT_2025_10_09__13_46_37: a transformer denoiser on
the hits (x, z, logE, logT) with per-hit layer conditioning (the per-layer minimum
time branch exists but, as trained, does not reach the output). Ported verbatim apart from unused code (consistency distillation,
ECAL conditioning, other samplers); parameter names are unchanged, so old
checkpoints load with strict=True.

    python scripts/training.py --net CaloHadronic
"""

import k_diffusion as K
import torch
import torch.nn as nn
from models.common import Embedder


class CaloClouds2_Attention(nn.Module):
    def __init__(self, args):
        super().__init__()
        t = args.transformer
        self.point_dim = t.features
        net = PointCloudNet(
            point_dim=t.features,
            context_dim=t.cond_features,
            y_dim=t.y_dim,
            include_time=t.include_time,
            nhead=t.nhead,
            d_model=t.d_model,
            num_layers=t.num_layers,
            dim_feedforward=t.dim_feedforward,
            decoder_crossattn=t.decoder_crossattn,
            dropout_rate=t.dropout_rate,
            calorimeter_layers=t.calorimeter_layers,
            **t.to_dict(["embed_kwargs"])["embed_kwargs"],
        )
        self.diffusion = Denoiser(
            net,
            point_dim=t.features,
            sigma_data=args.model_sigma.sigma_data,
            device=args.device,
            diffusion_loss=args.diffusion_loss,
        )

    def get_loss(
        self,
        x,
        noise,
        sigma,
        cond_feats,
        padding_mask=None,
        y=None,
        mint_per_layer=None,
    ):
        """x, noise: (B, N, d); sigma: (B,); cond_feats: (B, C)."""
        return self.diffusion.loss(
            x,
            noise,
            sigma,
            context=cond_feats,
            padding_mask=padding_mask,
            y=y,
            mint_per_layer=mint_per_layer,
        ).mean()

    def sample(
        self,
        cond_feats,
        num_points,
        config,
        y=None,
        mint_per_layer=None,
        padding_mask=None,
    ):
        if config.sampler != "heun":
            raise NotImplementedError("only the heun sampler is ported")
        x_T = (
            torch.randn(
                [cond_feats.size(0), num_points, self.point_dim],
                device=cond_feats.device,
            )
            * config.sigma_max
        )
        sigmas = K.sampling.get_sigmas_karras(
            config.num_steps,
            config.sigma_min,
            config.sigma_max,
            rho=config.rho,
            device=cond_feats.device,
        )
        return K.sampling.sample_heun(
            self.diffusion,
            x_T,
            sigmas,
            extra_args={
                "context": cond_feats,
                "padding_mask": padding_mask,
                "y": y,
                "mint_per_layer": mint_per_layer,
            },
            s_churn=config.s_churn,
            s_noise=config.s_noise,
            disable=True,
        )


# from: https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/layers.py#L12
class Denoiser(nn.Module):
    """A Karras et al. preconditioner for denoising diffusion models."""

    def __init__(
        self, inner_model, point_dim, sigma_data=0.5, device="cuda", diffusion_loss="l2"
    ):
        super().__init__()
        self.inner_model = inner_model
        self.sigma_data1 = sigma_data
        if isinstance(sigma_data, float):
            sigma_data = [sigma_data for _ in range(point_dim)]
        if len(sigma_data) != point_dim:
            raise ValueError(
                f"sigma_data must be either a float or a list of point_dim({point_dim}) floats."
            )
        # plain attribute, not a buffer: the old checkpoints do not contain it
        self.sigma_data = torch.tensor(sigma_data, device=device)  # d,
        self.diffusion_loss = diffusion_loss

    def get_scalings(self, sigma):  # B,
        sigma_data = self.sigma_data.expand(sigma.shape[0], -1)  # B, d
        sigma = K.utils.append_dims(sigma, sigma_data.ndim)  # B, d
        c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)
        c_out = sigma * sigma_data / (sigma**2 + sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def loss(self, input, noise, sigma, **kwargs):
        c_skip, c_out, c_in = [
            x.unsqueeze(1) for x in self.get_scalings(sigma)
        ]  # B,1,d

        noised_input = input + noise * K.utils.append_dims(sigma, input.ndim)
        if torch.isnan(noised_input).any():
            raise ValueError("model output is NaN")
        model_output, mask = self.inner_model(noised_input * c_in, sigma, **kwargs)
        target = (input - c_skip * noised_input) / c_out

        if self.diffusion_loss == "EDM-monotonic":
            # from Understanding Diffusion Objectives as the ELBO with Simple Data Augmentation
            weights = (sigma**2 + self.sigma_data1**2) / (sigma * self.sigma_data1) ** 2
            peak_id = torch.argmax(weights)
            edm_mono_weights = weights.clone()
            edm_mono_weights[:peak_id] = weights[peak_id]
            # as trained: (B,1,1) weights times the (n_valid,) per-hit loss, so the
            # caller's .mean() is mean(weights) * mean over all valid hits
            if mask is not None:
                return K.utils.append_dims(edm_mono_weights, input.ndim) * (
                    model_output[~mask] - target[~mask]
                ).pow(2).flatten(1).mean(1)
            return K.utils.append_dims(edm_mono_weights, input.ndim) * (
                model_output - target
            ).pow(2).flatten(1).mean(1)
        if self.diffusion_loss == "l2":
            if mask is not None:
                return (model_output[~mask] - target[~mask]).pow(2).flatten(1).mean(1)
            return (model_output - target).pow(2).flatten(1).mean(1)
        if self.diffusion_loss == "l1":
            return (model_output - target).abs().flatten(1).mean(1)
        raise ValueError("diffusion_loss must be l1, l2 or EDM-monotonic")

    def forward(self, input, sigma, **kwargs):
        c_skip, c_out, c_in = [
            x.unsqueeze(1) for x in self.get_scalings(sigma)
        ]  # B,1,d
        model_output, mask = self.inner_model(input * c_in, sigma, **kwargs)
        return model_output * c_out + input * c_skip


class PointCloudNet(nn.Module):
    def __init__(
        self,
        point_dim,
        context_dim,
        y_dim,
        include_time=False,
        d_model=128,
        time_dim=64,
        nhead=4,
        num_layers=6,
        dim_feedforward=2048,
        decoder_crossattn=False,
        dropout_rate=0,
        calorimeter_layers=78,
        **embed_kwargs,
    ):
        super().__init__()
        fourier_scale = (
            16  # 1 in k-diffusion, 16 in EDM, 30 in Score-based generative modeling
        )
        self.y_dim = y_dim
        self.decoder_crossattn = decoder_crossattn
        self.include_time = include_time

        if self.include_time:
            self.time_proj_forth = nn.Sequential(
                nn.Linear(1, calorimeter_layers),
                nn.GELU(),
                nn.Linear(calorimeter_layers, calorimeter_layers),
            )
            self.mintPerL_embedding = nn.Sequential(
                nn.Linear(calorimeter_layers, calorimeter_layers * 2),
                nn.GELU(),
                nn.Linear(calorimeter_layers * 2, calorimeter_layers),
            )
            self.time_proj_back = nn.Sequential(
                nn.Linear(calorimeter_layers, calorimeter_layers),
                nn.GELU(),
                nn.Linear(calorimeter_layers, 1),
            )

        self.timestep_embeding = nn.Sequential(
            K.layers.FourierFeatures(1, time_dim, std=fourier_scale),  # fixed (buffer)
            nn.Linear(time_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, d_model),
        )
        self.context_embeding = nn.Sequential(
            nn.Linear(context_dim, context_dim),
            nn.GELU(),
            nn.Linear(context_dim, d_model),
        )

        if embed_kwargs["activate_fourier_layer"]:
            dim_final = (
                point_dim * embed_kwargs["N_freqs"] * len(embed_kwargs["periodic_fns"])
            )
            if embed_kwargs["include_input"]:
                dim_final += point_dim
            self.point_embeding = nn.Sequential(
                Embedder(point_dim, **embed_kwargs),
                nn.Linear(dim_final, d_model),
            )
        else:
            self.point_embeding = nn.Sequential(
                nn.Linear(point_dim, point_dim * 4),
                nn.GELU(),
                nn.Linear(point_dim * 4, d_model),
            )

        if self.y_dim > 0:
            self.embed_layers = nn.Embedding(
                num_embeddings=calorimeter_layers + 1, embedding_dim=self.y_dim
            )
            self.embed_x_and_y = nn.Sequential(
                nn.Linear(d_model + y_dim, d_model + y_dim),
                nn.GELU(),
                nn.Linear(d_model + y_dim, d_model),
            )

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            dropout=dropout_rate,
            dim_feedforward=dim_feedforward,
        )
        self.denoiser = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)

        if self.decoder_crossattn:
            transformer_decoder_layer = nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                batch_first=True,
                dropout=dropout_rate,
                dim_feedforward=dim_feedforward,
            )
            self.trasf_decoder = nn.TransformerDecoder(
                transformer_decoder_layer, num_layers=1
            )

        self.decoder = nn.Sequential(
            nn.Linear(d_model, int(d_model / 2)),
            nn.GELU(),
            nn.Linear(int(d_model / 2), int(d_model / 4)),
            nn.GELU(),
            nn.Linear(int(d_model / 4), point_dim),
        )

    def forward(
        self, x, sigma, context, y=None, mint_per_layer=None, padding_mask=None
    ):
        """
        x: hits at noise level sigma (B, N, d); sigma: (B,); context: (B, F);
        y: layer index per hit (B, N, 1); mint_per_layer: (B, layers);
        padding_mask: (B, N), True = padded hit.
        """
        batch_size, N, _ = x.shape
        sigma = sigma.view(batch_size, 1, 1)
        context = context.view(batch_size, 1, -1)
        # as trained: embed the raw hits, so the time/mint branch below never reaches the output (its weights stayed at init)
        points_emb = self.point_embeding(x)

        if self.include_time:
            mint_per_layer = mint_per_layer.view(batch_size, 1, -1)
            t = self.time_proj_forth(x[:, :, -1:])
            mint = self.mintPerL_embedding(mint_per_layer)
            x = torch.cat([x[:, :, :-1], self.time_proj_back(t + mint)], dim=-1)

        # formulation from EDM paper / k-diffusion
        c_noise = sigma.log() / 4
        time_emb = self.timestep_embeding(c_noise)  # (B, 1, d_model)
        ctx_emb = self.context_embeding(context)  # (B, 1, d_model)

        if self.y_dim > 0:
            y_cond = self.embed_layers(y[:, :, 0].int() + 1).float()  # (B, N, y_dim)
            x = self.embed_x_and_y(torch.cat([points_emb, y_cond], dim=-1))
        else:
            x = points_emb

        x = torch.cat([x, time_emb, ctx_emb], dim=1)  # (B, N+2, d_model)

        # True = not attended; the time and context tokens are never padded
        mask = None
        if padding_mask is not None:
            context_pad = torch.zeros(
                batch_size, 2, dtype=torch.bool, device=padding_mask.device
            )
            mask = torch.cat([padding_mask, context_pad], dim=-1)

        x = self.denoiser(x, src_key_padding_mask=mask)
        if self.decoder_crossattn:
            x = self.trasf_decoder(
                tgt=points_emb,
                memory=x,
                tgt_is_causal=False,
                memory_key_padding_mask=mask,
                tgt_key_padding_mask=padding_mask,
            )
        else:
            x = x[:, :-2]  # drop the time and context tokens

        return self.decoder(x), padding_mask
