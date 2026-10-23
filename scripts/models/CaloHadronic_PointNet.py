import k_diffusion as K
import torch
from models.common import Embedder
from models.VAE.encoder_decoder import EncoderBlock
from torch import nn


# from: https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/layers.py#L12
class Denoiser(nn.Module):
    """A Karras et al. preconditioner for denoising diffusion models."""

    def __init__(
        self,
        inner_model,
        point_dim,
        sigma_data=0.5,
        device="cuda",
        distillation=False,
        sigma_min=0.002,
        diffusion_loss="l2",
        diffusion_way="EDM",
    ):
        super().__init__()
        self.inner_model = inner_model
        # Callable actually used in the forward passes. Kept inside a list so that
        # nn.Module does not register it as a submodule: a torch.compile wrapper
        # would otherwise prefix every state_dict key with "_orig_mod.", breaking
        # the loading of checkpoints saved before compilation was enabled.
        self._net = [inner_model]
        self.sigma_data1 = sigma_data
        if isinstance(sigma_data, float):
            sigma_data = [sigma_data for _ in range(point_dim)]
        if len(sigma_data) != point_dim:
            raise ValueError(
                f"sigma_data must be either a float or a list of point_dim({point_dim}) floats."
            )
        self.sigma_data = torch.tensor(sigma_data, device=device)  # 4,
        self.device = device
        self.distillation = distillation
        self.sigma_min = sigma_min
        self.diffusion_loss = diffusion_loss
        self.diffusion_way = diffusion_way

    def set_compiled_model(self, compiled_model):
        """Route the forward passes through `compiled_model` (a torch.compile
        wrapper of `inner_model`), leaving the registered module tree untouched."""
        self._net[0] = compiled_model

    def get_scalings(self, sigma):  # B,
        sigma_data = self.sigma_data.expand(sigma.shape[0], -1)  # B, 4
        sigma = K.utils.append_dims(sigma, sigma_data.ndim)  # B, 4
        c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)  # B, 4
        c_out = sigma * sigma_data / (sigma**2 + sigma_data**2) ** 0.5  # B, 4
        c_in = 1 / (sigma**2 + sigma_data**2) ** 0.5  # B, 4
        return c_skip, c_out, c_in

    def get_scalings_for_boundary_condition(
        self, sigma
    ):  # B,   # for consistency model
        sigma_data = self.sigma_data.expand(sigma.shape[0], -1)  # B, 4
        sigma = K.utils.append_dims(sigma, sigma_data.ndim)  # B, 4
        c_skip = sigma_data**2 / ((sigma - self.sigma_min) ** 2 + sigma_data**2)  # B, 4
        c_out = (
            (sigma - self.sigma_min) * sigma_data / (sigma**2 + sigma_data**2) ** 0.5
        )  # B, 4
        c_in = 1 / (sigma**2 + sigma_data**2) ** 0.5  # B, 4
        return c_skip, c_out, c_in

    def loss(self, input, noise, sigma, **kwargs):
        c_skip, c_out, c_in = [
            x.unsqueeze(1) for x in self.get_scalings(sigma)
        ]  # B,1,4
        if self.diffusion_way == "EDM":
            noised_input = input + noise * K.utils.append_dims(sigma, input.ndim)
            if torch.isnan(noised_input).any():
                raise ValueError("model output is NaN")
            model_output, mask = self._net[0](noised_input * c_in, sigma, **kwargs)
            target = (input - c_skip * noised_input) / c_out

            if self.diffusion_loss == "EDM-monotonic":
                # from Understanding Diffusion Objectives as the ELBO with Simple Data Augmentation
                weights = (sigma**2 + self.sigma_data1**2) / (
                    sigma * self.sigma_data1
                ) ** 2
                # weights = 1/sigma_data^2 + 1/sigma^2 is strictly decreasing in sigma:
                # cap by value at the weight reached when sigma == sigma_data.
                cap = 2.0 / self.sigma_data1**2
                edm_mono_weights = weights.clamp(max=cap)

                if mask is not None:
                    # mask convention: True = padding/invalid, so ~mask = valid points
                    sq_err = (model_output - target).pow(2)  # (B, N, D)
                    valid = (
                        (~mask).unsqueeze(-1) if mask.ndim == 2 else (~mask)
                    )  # broadcast to (B, N, D)
                    sq_err = sq_err * valid
                    per_sample_loss = sq_err.flatten(1).sum(1) / valid.flatten(1).sum(
                        1
                    ).clamp(min=1)
                    return (
                        K.utils.append_dims(edm_mono_weights, input.ndim)
                        .flatten(1)
                        .mean(1)
                        * per_sample_loss
                    )
                else:
                    return K.utils.append_dims(edm_mono_weights, input.ndim) * (
                        model_output - target
                    ).pow(2).flatten(1).mean(1)
            if self.diffusion_loss == "l2":
                if mask is not None:
                    # mask convention: True = padding/invalid, so ~mask = valid points
                    sq_err = (model_output - target).pow(2)  # (B, N, D)
                    valid = (~mask).unsqueeze(-1) if mask.ndim == 2 else (~mask)
                    sq_err = sq_err * valid
                    return sq_err.flatten(1).sum(1) / valid.flatten(1).sum(1).clamp(
                        min=1
                    )
                else:
                    return (model_output - target).pow(2).flatten(1).mean(1)

            elif self.diffusion_loss == "l1":
                return (model_output - target).abs().flatten(1).mean(1)
            else:
                raise ValueError("diffusion_loss must be either l1 or l2")

        elif self.diffusion_way == "cosine":
            # this is wrong!!!!
            alpha_t = torch.cos((torch.pi / 2) * sigma)  # Clean data scale
            sigma_t = torch.sin((torch.pi / 2) * sigma)  # Noise scale
            lambda_t = 0.5
            noised_input = (
                K.utils.append_dims(alpha_t, input.ndim) * input
                + K.utils.append_dims(sigma_t, input.ndim) * noise
            )
            target = (input - c_skip * noised_input) / c_out
            model_output, mask = self._net[0](noised_input * c_in, sigma, **kwargs)
            w_t = K.utils.append_dims(torch.exp(-lambda_t * sigma / 2), input.ndim)
            return w_t * (model_output[~mask] - target[~mask]).pow(2).flatten(1).mean(1)

    def forward(
        self, input, sigma, **kwargs
    ):  # same as "denoise" in KarrasDenoiser of CM code
        if isinstance(sigma, float) or isinstance(sigma, int):
            sigma = (
                torch.tensor([sigma] * input.shape[0], dtype=torch.float32)
                .to(input.device)
                .unsqueeze(1)
            )
        if not self.distillation:
            c_skip, c_out, c_in = [
                x.unsqueeze(1) for x in self.get_scalings(sigma)
            ]  # B,1,4
        else:
            c_skip, c_out, c_in = [
                x.unsqueeze(1) for x in self.get_scalings_for_boundary_condition(sigma)
            ]
        # CM code did an additional resacling of the time sigma for the time conditing
        model_output, mask = self._net[0](input * c_in, sigma, **kwargs)
        return model_output * c_out + input * c_skip

    # inspired by https://github.com/openai/consistency_models/blob/main/cm/karras_diffusion.py#L106


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, max_len=6500):
        super().__init__()

        self.pe = nn.Parameter(torch.zeros(max_len, d_model))

    def forward(self, x):
        x = x + self.pe[: x.size(1)]
        return x


class PointCloudNet(nn.Module):
    def __init__(
        self,
        args,
        use_sigma_embedding: bool = True,
    ):
        """`use_sigma_embedding=False` drops `timestep_embeding` and makes `sigma`
        an ignored argument of `forward`. EDM needs it -- sigma is the real noise
        level and the only way the denoiser is told where on the schedule it is.
        MeanFlow does not: CaloHadronic_FM carries the two times itself, through
        t_embeding and r_embeding summed into `cond`, and was passing a literal
        `torch.ones` for sigma. log(1)/4 = 0 for every sample, so the whole stack
        reduced to one fixed vector added to every token -- redundant with the
        biases downstream, and ~66k parameters of nothing."""
        super().__init__()
        fourier_scale = (
            16  # 1 in k-diffusion, 16 in EDM, 30 in Score-based generative modeling
        )
        point_dim = args.transformer.input_feats
        context_dim = args.transformer.cond_feats
        self.y_dim = args.transformer.layer_dim
        ecal_dim = args.transformer.ecal_feats

        time_dim = args.transformer.time_dim
        include_time = args.transformer.include_time
        ecal_compressed = args.transformer.ecal_compressed
        nhead = args.transformer.nhead
        d_model = args.transformer.d_model
        num_layers = args.transformer.num_layers
        calorimeter_layers = args.transformer.calorimeter_layers
        self.calorimeter_layers = calorimeter_layers
        downsample_factor = args.downsample_factor
        embed_kwargs = args.transformer.to_dict(["embed_kwargs"])["embed_kwargs"]

        self.num_layers = num_layers
        self.d_model = d_model
        self.ecal_dim = ecal_dim
        self.context_dim = context_dim
        self.ecal_compressed = ecal_compressed
        self.include_time = include_time
        self.downsample_factor = downsample_factor

        # Learnable absolute positional embedding over the latent token index: the VAE
        # latent is ordered (windowed down/up-sampling over layer-sorted points) and the
        # decoder reads that order, while the banded attention mask is translation-
        # equivariant. Zero init, excluded from weight decay. Off by default so older
        # checkpoints load; LatentDiff only (downsample_factor is None otherwise).
        latent_pos_emb = bool(getattr(args.transformer, "latent_pos_emb", False))
        if latent_pos_emb and downsample_factor is None:
            raise ValueError(
                "transformer.latent_pos_emb needs the latent token count, but "
                "downsample_factor is None. It is only set for net='LatentDiff' "
                "(see training.py get_model); the hit-level point clouds this net "
                "is used for otherwise really are unordered."
            )
        self.latent_pos_emb = (
            nn.Parameter(torch.zeros(downsample_factor, d_model))
            if latent_pos_emb
            else None
        )

        if self.include_time:
            # Per-layer minimum hit time, [B, calorimeter_layers]: a per-shower
            # conditioning feature, embedded to d_model and added to static_cond.
            self.mintPerL_embedding = nn.Sequential(
                nn.Linear(calorimeter_layers, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            self.mint_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.use_sigma_embedding = use_sigma_embedding
        self.timestep_embeding = (
            nn.Sequential(
                K.layers.FourierFeatures(
                    1, time_dim, std=fourier_scale
                ),  # 1D Fourier features --> with register_buffer, so weights are not trained
                nn.Linear(time_dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, d_model),
            )
            if use_sigma_embedding
            else None
        )

        self.context_embeding = nn.Sequential(
            nn.Linear(context_dim, context_dim),
            nn.GELU(),
            nn.Linear(context_dim, d_model),
        )

        if embed_kwargs["activate_fourier_layer"]:
            if embed_kwargs["include_input"]:
                dim_final = point_dim + point_dim * embed_kwargs["N_freqs"] * len(
                    embed_kwargs["periodic_fns"]
                )
            else:
                dim_final = (
                    point_dim
                    * embed_kwargs["N_freqs"]
                    * len(embed_kwargs["periodic_fns"])
                )

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
            # Layer conditioning: `y` holds integer layer ids, each layer gets a learned
            # embedding lifted to d_model by `linear_y`; pooled in `_layer_cond`.
            self.embed_layers = nn.Embedding(
                num_embeddings=calorimeter_layers + 1, embedding_dim=self.y_dim
            )
            self.linear_y = nn.Sequential(
                nn.Linear(self.y_dim, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            # Pins the magnitude of the pooled layer conditioning.
            self.layer_cond_norm = nn.LayerNorm(d_model, eps=1e-6)
        # `EncoderBlock` is shared with the VAE (VAE/compressor5.py) and hard-codes
        # dim_feedforward = 4 * d_model. A config without transformer.dropout uses 0.1.
        self.denoiser = nn.ModuleList(
            [
                EncoderBlock(
                    d_model,
                    nhead,
                    max_len=self.downsample_factor,
                    dropout=float(getattr(args.transformer, "dropout", 0.1)),
                    # RMS-normalise q and k: the only bound on the attention logits
                    # (pre_attn_norm bounds x, not Wq/Wk). Needed by MeanFlow, whose
                    # self-referential target does not resist logit-scale growth.
                    qk_norm=True,
                    # No learnable gain: redundant with the projection, and it lets the
                    # logit scale grow again. Without it the scale is sqrt(head_dim).
                    qk_norm_affine=False,
                    # Fills a non-persistent buffer for the mf/qk_logit_scale metric;
                    # changes no arithmetic.
                    track_qk_scale=True,
                )
                for _ in range(num_layers)
            ]
        )

        # Pre-attention LayerNorm, one per block: MHFlexEncoderLayer is post-LN, so
        # otherwise nothing bounds the scale of the attention input.
        self.pre_attn_norm = nn.ModuleList(
            [nn.LayerNorm(d_model, eps=1e-6) for _ in range(num_layers)]
        )

        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, point_dim),
        )

    def _layer_profile(self, y, padding=None):
        """Normalised points-per-layer profile, [B, calorimeter_layers].

        Padded points carry layer index 0 (dataset.py zeroes them), which cannot be
        told apart from a genuine layer-0 hit, so they are dropped via the padding
        mask whenever the caller supplies one.
        """
        idx = y[:, :, 0].long().clamp(0, self.calorimeter_layers - 1)
        weight = (
            torch.ones_like(idx, dtype=torch.float32)
            if padding is None
            else (~padding.bool()).to(torch.float32)
        )
        profile = torch.zeros(
            idx.shape[0], self.calorimeter_layers, device=y.device, dtype=torch.float32
        )
        profile.scatter_add_(1, idx, weight)
        return profile / profile.sum(dim=1, keepdim=True).clamp(min=1.0)

    def _layer_cond(self, profile):
        """Per-shower layer conditioning, [B, d_model], from the normalised
        points-per-layer profile [B, calorimeter_layers].

        `y` is a per-hit integer layer id, so the natural encoding is an embedding
        table: layer l -> embed_layers[l + 1] -> linear_y -> a d_model vector v_l.
        The old code built the shower vector as
        `linear_y(embed_layers(y + 1)).mean(dim=1)`, and because y takes only
        calorimeter_layers distinct values that mean is identically

            sum_l (n_l / N) * v_l   =   profile @ V,    V[l] = linear_y(embed[l + 1])

        so this computes exactly the same quantity, with two differences. `profile`
        comes from `_layer_profile`, which drops padded points via the mask instead
        of counting them as layer-0 hits. And `linear_y` runs on the
        calorimeter_layers embedding rows once rather than on every point in the
        batch, which is the same arithmetic for far less work.

        Note the ordering: linear_y is applied per layer and the mixing by `profile`
        happens afterwards, so the map from profile to output is linear. That is the
        old behaviour, kept deliberately.

        Pooling with the normalised `profile` (a mean) rather than the raw counts (a
        sum) costs exactly one scalar, the total hit count N, because sum = N * mean.
        The profile SHAPE survives in full: `rows` has rank calorimeter_layers, so
        `profile -> profile @ rows` is injective and all of it is recoverable. N
        itself is already handed to `context_embeding` -- cond_feats[:, 1] correlates
        1.0000 with the hit count -- while summing would make this vector's magnitude
        swing with N over a 3x range between showers and dwarf the token features it
        is added to. The LayerNorm below also makes the choice moot: LayerNorm is
        scale-invariant, so LN(sum) and LN(mean) are the same tensor.
        """
        rows = self.linear_y(
            self.embed_layers.weight[1 : self.calorimeter_layers + 1]
        )  # (calorimeter_layers, d_model)
        return self.layer_cond_norm(profile.to(rows.dtype) @ rows)  # (B, d_model)

    def compute_static_cond(
        self,
        context,
        y=None,
        effective_p_dim: int = None,
        N: int = None,
        padding=None,
        mint_per_layer=None,
    ):
        """Conditioning that depends on neither sigma nor x, so it can be computed
        once per sampling call instead of once per denoiser evaluation. Mirrors the
        truncation and the ctx/layer embedding done in `forward`."""
        batch_size = context.shape[0]
        # built from the untruncated y on purpose: the slice below only exists to
        # match the attention's sequence length, it is not a statement about which
        # hits belong to the shower
        profile = self._layer_profile(y, padding) if self.y_dim > 0 else None
        if effective_p_dim is not None and N is not None and effective_p_dim < N:
            if y is not None:
                y = y[:, :effective_p_dim]
        cond = self.context_embeding(context.view(batch_size, 1, -1))
        if self.y_dim > 0:
            cond = cond + self._layer_cond(profile).unsqueeze(1)
        if self.include_time and mint_per_layer is not None:
            cond = cond + self.mint_norm(
                self.mintPerL_embedding(mint_per_layer.view(batch_size, -1))
            ).unsqueeze(1)
        return cond

    def forward(
        self,
        x,
        sigma,
        context,
        y=None,
        mint_per_layer=None,
        block_mask=None,
        effective_p_dim: int = None,
        static_cond=None,
        padding=None,
    ):
        """
        Args:
            x:  Point clouds at some timestep t, (B, N, d).
            sigma:     Time. (B, ).  --> becomes "sigma" in k-diffusion
            context:  Shape latents. (B, F).
        """
        batch_size, N, D = x.shape

        # before the effective_p_dim slice below, for the same reason as in
        # compute_static_cond()
        profile = self._layer_profile(y, padding) if self.y_dim > 0 else None

        # mask convention: True = padding/invalid
        padding_mask = torch.zeros(batch_size, N, dtype=torch.bool, device=x.device)
        if effective_p_dim is not None and effective_p_dim < N:
            # truncate to effective_p_dim so the sequence length matches what
            # block_mask was built for (avoids the pad_to_max_len_fast_x assert
            # in the windowed attention, which requires N <= effective_p_dim)
            padding_mask[:, effective_p_dim:] = True
            x = x[:, :effective_p_dim]
            if y is not None:
                y = y[:, :effective_p_dim]

        if self.use_sigma_embedding:
            sigma = sigma.view(batch_size, 1, 1)  # (B, 1, 1)
        context = context.view(batch_size, 1, -1)  # (B, 1, F)

        x = self.point_embeding(x)

        # formulation from EDM paper / k-diffusion. Skipped entirely under MeanFlow,
        # which passes sigma=None and supplies its own t and r through static_cond.
        if self.use_sigma_embedding:
            c_noise = sigma.log() / 4  # (B, 1, 1)
            time_emb = self.timestep_embeding(c_noise)  # (B, 1, d_model)
        else:
            time_emb = 0.0
        if static_cond is None:
            # ctx embedding + layer conditioning; neither depends on sigma or x, so
            # `sample()` precomputes them once and passes them in as static_cond.
            static_cond = self.context_embeding(context)  # (B, 1, d_model)
            if self.y_dim > 0:
                static_cond = static_cond + self._layer_cond(profile).unsqueeze(
                    1
                )  # [B, 1, d_model]
            if self.include_time and mint_per_layer is not None:
                static_cond = static_cond + self.mint_norm(
                    self.mintPerL_embedding(mint_per_layer.view(batch_size, -1))
                ).unsqueeze(1)

        # sum (not average) so the context embedding isn't diluted by the
        # timestep embedding: each carries its full magnitude into x.
        x = x + time_emb + static_cond  # (B, N, d_model)

        if self.latent_pos_emb is not None:
            # x was already truncated to effective_p_dim above, and the latent
            # tokens are index-ordered, so the head of the table is the right
            # slice -- exactly what the VAE does with query_pe[:p].
            if x.shape[1] > self.latent_pos_emb.shape[0]:
                raise ValueError(
                    f"latent_pos_emb has {self.latent_pos_emb.shape[0]} positions "
                    f"but the sequence is {x.shape[1]} long"
                )
            x = x + self.latent_pos_emb[: x.shape[1]].unsqueeze(0)

        for pre_norm, enc_layer in zip(self.pre_attn_norm, self.denoiser):
            x = pre_norm(x)
            x = enc_layer(x, block_mask=block_mask, effective_p_dim=effective_p_dim)

        x = self.decoder(x)

        if x.shape[1] < N:
            x = nn.functional.pad(x, (0, 0, 0, N - x.shape[1]))

        return x, padding_mask


class KLDloss(nn.Module):
    def __init__(self):
        super().__init__()
        self.use_cuda = torch.cuda.is_available()

    def forward(self, mu, logvar):
        B = logvar.size(0)
        KLD = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())) / (B)
        return KLD
