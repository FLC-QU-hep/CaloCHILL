import random
from itertools import islice

import k_diffusion as K
import matplotlib as mlp
import numpy as np
import torch
import torch.distributed as dist
from models.CaloHadronic_PointNet import Denoiser, KLDloss, PointCloudNet
from models.common import (
    EBSW,
    F1Score,
    L2_ChamferEval_1000,
    PerLayerOTLoss,
    chamfer_loss_v2,
    dcd_loss_from_v2,
    l2_loss_per_layer_per_shower,
    pad_tensor_to_max,
    part_chamfer_distance,
    sliced_wasserstein_per_layer_fast,
    sliced_wasserstein_per_shower_layer,
)
from models.gmm_prior import ConditionalGMMPrior
from models.VAE.compressor5 import Compressor
from torch import nn
from tqdm.auto import trange
from utils.block_mask import MAX_LEN_BUCKET, BlockMaskCache
from utils.compr_utils import Compute_log_probabilities
from utils.dynamo_compat import patch_no_tensor_aliasing_reason
from utils.flex_attn import pad_to_max_len_fast_padding
from utils.misc import get_logger

mlp.rcParams["text.usetex"] = False
logger = get_logger(name="CaloHadronic.py")


def _n_points_and_top_points_per_layer(layer_index, mask):
    layer_index = layer_index.squeeze(-1).long()  # [B, N]
    valid = ~mask.bool()  # [B, N]
    num_layers = layer_index.max() + 1
    B, N = layer_index.shape
    one_hot = torch.zeros(
        B, N, num_layers, device=layer_index.device, dtype=torch.float
    )
    one_hot.scatter_(
        2, layer_index.unsqueeze(-1), valid.float().unsqueeze(-1)
    )  # [B, N, num_layers]
    points_per_layer = one_hot.sum(dim=1)  # [B, num_layers]

    top2 = torch.topk(points_per_layer, k=2, dim=1).values  # [B, 2]
    kwargs = {
        "n_points": points_per_layer.sum(axis=1),
        "top_points_layers": top2.sum(dim=1),
    }
    return kwargs


class CompressorVAE(nn.Module):
    def __init__(self, cfg_data, args_model):
        super().__init__()
        self.cfg = cfg_data
        self.max_len = cfg_data.max_points

        self.args = args_model
        args_model.max_points_per_layer = cfg_data.max_points_per_layer
        args_model.max_points = cfg_data.max_points
        args_model.calorimeter_layers = cfg_data.calorimeter_layers

        self.include_time = args_model.include_time
        self.kl_weight = args_model.kl_weight
        self.emd_weight = args_model.emd_weight
        self.loss_mode = args_model.loss
        self.neighbors = args_model.neighbors
        # Per-layer 1-D Wasserstein on each feature axis, added to the random-direction
        # SWD. Without time, logE dominates every random projection and x/z enter only
        # at second order (z can collapse); the axis terms see each feature at first
        # order.
        self.swd_axis_weight = (
            0.0 if self.include_time else getattr(args_model, "swd_axis_weight", 0.0)
        )

        self.MASK = BlockMaskCache(
            neighbors=self.neighbors,
            p_dim=self.args.p_dim,
            device=args_model.device,
        )
        self.model = Compressor(args_model)
        # Attributes that pad a sequence out to the global max_points before
        # attention, so __calculate_block_mask() can retarget them at the batch's own
        # length. Collected before torch.compile so the wrapper is out of the way, and
        # matched by value: the latent-side attentions are built with
        # downsample_points instead and must keep it.
        self.dynamic_max_len = getattr(args_model, "dynamic_max_len", True)
        self._max_len_attrs = [
            (module, name)
            for module in self.model.modules()
            for name in ("max_len_q", "max_len_kv")
            if getattr(module, name, None) == self.max_len
        ]
        precision = getattr(args_model, "matmul_precision", None)
        if args_model.device == "cuda" and precision is not None:
            # The decode is matmul-bound: "high" uses TF32, "medium" bf16 inputs
            # (storage stays fp32). Lower settings change sampled showers, so validate
            # distributions first.
            torch.set_float32_matmul_precision(precision)
        if args_model.device == "cuda":
            # Before any compile: a stale guard source in dynamo's recompilation-reason
            # message would otherwise abort the recompile and drop us back to eager.
            patch_no_tensor_aliasing_reason()
            self.model = torch.compile(self.model, dynamic=True)

        self.prob_class = Compute_log_probabilities()

        if args_model.loss == "geom_loss":
            self.geom_loss = PerLayerOTLoss(loss="sinkhorn", p=2, blur=0.01)
            self.geom_loss = torch.compile(self.geom_loss, dynamic=True)
        elif args_model.loss in ("SWD_perlayer_fast", "SWD_pershower_layer"):
            # SWD_pershower_layer: points matched within each (shower, layer) instead of across the batch
            self.sliced_wasserstein_per_layer_fast = torch.compile(
                sliced_wasserstein_per_shower_layer
                if args_model.loss == "SWD_pershower_layer"
                else sliced_wasserstein_per_layer_fast,
                dynamic=True,
            )
        if (self.args.n_gaussian_mixture > 1) and (self.args.kl_integral_asloss):
            logger.error(
                "You are using a Gaussian Mixture with more than one component and kl_integral_asloss=True. This won't work!"
            )

        if self.loss_mode == "EBSW":
            self.ebsw = EBSW(L=100, device=args_model.device)

    def __calculate_block_mask(
        self,
        mask,
        **kwargs,
    ) -> None:
        B = mask.shape[0]

        # Compute per-sample valid point counts
        valid_counts = torch.sum(~mask, dim=1)  # [B]

        # Dynamic p_dim: cap at actual min valid points in batch, but don't exceed p_dim
        self.effective_p_dim = min(int(valid_counts.min().item()), self.args.p_dim)
        self.pad_learnable = torch.ones(B, self.effective_p_dim, dtype=torch.bool).to(
            self.args.device
        )
        if self.dynamic_max_len:
            # Pad to this batch's length (rounded up to the flex-attention block size)
            # instead of the global max_points, so small showers do not pay for a
            # max_len^2 block mask.
            bucket = getattr(self.args, "max_len_bucket", MAX_LEN_BUCKET)
            self.max_len = min(
                self.cfg.max_points, -(-mask.shape[1] // bucket) * bucket
            )
            for module, name in self._max_len_attrs:
                setattr(module, name, self.max_len)
        pad = ~pad_to_max_len_fast_padding(mask, self.max_len)

        #  mask for latent
        block_mask_lat = self.MASK.get(
            self.pad_learnable,
            self.pad_learnable,
            self.effective_p_dim,
            self.effective_p_dim,
            type="layer",
            **kwargs,
        )
        # mask for down sampling
        block_mask_ds = self.MASK.get(
            self.pad_learnable,
            pad,
            self.effective_p_dim,
            self.max_len,
            type="p_dim_based",
            **kwargs,
        )
        #  mask for up sampling
        block_mask_us = self.MASK.get(
            pad,
            self.pad_learnable,
            self.max_len,
            self.effective_p_dim,
            type="p_dim_based",
            **kwargs,
        )
        #  mask for refinement; only the out_refinement blocks read it, so skip it
        #  when there are none (checked on the module, which ablations truncate)
        refinement = getattr(self.model, "_orig_mod", self.model).out_refinement
        block_mask_ref = (
            self.MASK.get(
                pad,
                pad,
                self.max_len,
                self.max_len,
                type="layer",
                **kwargs,
            )
            if len(refinement)
            else None
        )
        self.masks = [block_mask_ds, block_mask_lat, block_mask_us, block_mask_ref]

    def get_loss(
        self,
        tar,
        ctx=None,
        y=None,
        mint_per_layer=None,
        padding_mask=None,
        beta=1,
        it=0,
        **kwargs,
    ):
        self.__calculate_block_mask(padding_mask, **kwargs)
        output = self.model(
            tar,
            ctx=ctx,
            y=y,
            mint_per_layer=mint_per_layer,
            padding=padding_mask,
            masks=self.masks,
            effective_p_dim=self.effective_p_dim,
        )

        out, kls = (
            output["set"].float(),
            output["kls"],
        )

        if torch.isnan(out).any():
            print("output has NaN")
        if torch.isnan(tar).any():
            print("target has NaN")
        if torch.isnan(y).any():
            print("layer has NaN")
        if mint_per_layer is not None and torch.isnan(mint_per_layer).any():
            print("mint_per_layer has NaN")

        if out.shape != tar.shape:
            print(
                f"Output shape {out.shape} and target shape {tar.shape} do not match!"
            )
            return None

        if padding_mask is not None:
            out[padding_mask] = 0
            tar[padding_mask] = 0

        if padding_mask is not None:
            val_mask = ~padding_mask
        else:
            val_mask = None

        if val_mask is not None and torch.isnan(val_mask.float()).any():
            print("val_mask has NaN")

        if self.loss_mode == "DCD":
            cd_loss, _, _ = dcd_loss_from_v2(
                out, tar, alpha=1000, n_lambda=1.0, valid_mask=val_mask
            )
        elif self.loss_mode == "EBSW":  # energy-based sliced wasserstein distance
            cd_loss = self.ebsw(out, tar)
        elif self.loss_mode in ("SWD_perlayer_fast", "SWD_pershower_layer"):
            n_proj = 128
            cd_loss = self.sliced_wasserstein_per_layer_fast(
                out, tar, y, valid_mask=val_mask, num_projections=n_proj
            )
            if self.swd_axis_weight > 0:
                axes = torch.eye(out.shape[-1], device=out.device)
                cd_loss = cd_loss + self.swd_axis_weight * (
                    self.sliced_wasserstein_per_layer_fast(
                        out, tar, y, valid_mask=val_mask, directions=axes
                    )
                )
        elif self.loss_mode == "L1":
            cd_loss = torch.abs(out[val_mask] - tar[val_mask]).mean()
        elif self.loss_mode == "MSE":
            diff = (out - tar) ** 2
            cd_loss = diff[~val_mask].mean()
        elif self.loss_mode == "MSE_perlayer":
            cd_loss = l2_loss_per_layer_per_shower(out, tar, y[:, :, 0])
        elif self.loss_mode == "p-CD":
            cd_loss = part_chamfer_distance(out, tar, y)
        elif self.loss_mode == "geom_loss":
            cd_loss = self.geom_loss(
                out,
                tar,
                y[:, :, 0],
                valid_mask=val_mask,
                per_layer=True,
            )
        else:
            cd_loss, _, _ = chamfer_loss_v2(out, tar, valid_mask=val_mask)

        if self.args.loss_cd:
            cd_loss2, _, _ = chamfer_loss_v2(out, tar, valid_mask=val_mask)
            cd_loss = cd_loss + self.args.loss_cd_weight * cd_loss2

        kl_loss = torch.cat(kls, dim=1).float()
        # pad the kl loss
        if padding_mask is not None:
            kl_loss = kl_loss * self.pad_learnable.float()
            kl_loss = kl_loss.sum() / (
                self.pad_learnable.float().sum() * kl_loss.shape[-1]
            )
        else:
            kl_loss = kl_loss.mean()

        total_kl_loss = beta * kl_loss
        loss = cd_loss + self.kl_weight * total_kl_loss
        if torch.isnan(loss).any():
            print("NaN detected in loss!")
            print(loss, cd_loss, total_kl_loss)
            if cd_loss.isnan().any():
                print("NaN in CD loss!")
                print(torch.isnan(out).any())
                print(torch.isnan(tar).any())
                print(torch.isnan(y).any())
                print(torch.isnan(val_mask).any())
                print(torch.isnan(ctx).any())
        return {
            "loss": loss,
            "rec_loss": cd_loss,
            "kl_loss": kl_loss,
        }

    def reconstruction_and_sampling(self, val_loader, epoch, device, test=False):
        self.model.eval()
        output_DS = None
        all_ref, all_rec, all_sampled, padding_mask = (
            [],
            [],
            [],
            [],
        )
        count = 0
        m, all_latent, all_cond, all_y, all_mint, mask_ds, mask_notds = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        all_DS, all_notDS, posteriors, all_sampled_latent = [], [], [], []
        val_slice = 2 if test else 8
        rng = random.Random(epoch)  # different each epoch, but reproducible
        skip = rng.randint(0, len(val_loader) - val_slice)
        for i, batch in enumerate(islice(val_loader, skip, skip + val_slice)):
            output_DS, output_notDS = None, None
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key][0].to(device)
            kwargs = {
                "n_points": batch["points_per_layer"].sum(dim=1),
                "top_points_layers": batch["points_per_layer"].topk(2)[0].sum(dim=1),
            }
            self.__calculate_block_mask(
                batch["padding_mask"],
                **kwargs,
            )
            with torch.no_grad():
                output = self.model(
                    batch["events"],
                    ctx=batch["cond_feats"],
                    y=batch["layer_cond"],
                    mint_per_layer=batch["mint_per_layer"],
                    padding=batch["padding_mask"],
                    masks=self.masks,
                    effective_p_dim=self.effective_p_dim,
                )
                if batch["events"].shape[1] > self.args.p_dim:
                    output_downsampledONLY = self.model(
                        batch["events"],
                        ctx=batch["cond_feats"],
                        y=batch["layer_cond"],
                        mint_per_layer=batch["mint_per_layer"],
                        padding=batch["padding_mask"],
                        masks=self.masks,
                        effective_p_dim=self.effective_p_dim,
                    )
                else:
                    output_NOTdownsampled = self.model(
                        batch["events"],
                        ctx=batch["cond_feats"],
                        y=batch["layer_cond"],
                        padding=batch["padding_mask"],
                        masks=self.masks,
                        effective_p_dim=self.effective_p_dim,
                    )
                sampled, z_sampled = self.sample(
                    B=batch["layer_cond"].shape[0],
                    N=batch["layer_cond"].shape[1],
                    c=batch["cond_feats"],
                    y=batch["layer_cond"],
                    mint_per_layer=batch["mint_per_layer"],
                    padding=batch["padding_mask"],
                    device=device,
                )
            loss_dict = self.get_loss(
                batch["events"],
                ctx=batch["cond_feats"],
                y=batch["layer_cond"],
                mint_per_layer=batch["mint_per_layer"],
                padding_mask=batch["padding_mask"],
                **kwargs,
            )
            eps, rec_pts, post = output["z"], output["set"], output["posteriors"]
            if batch["events"].shape[1] > self.args.p_dim:
                output_DS = output_downsampledONLY["set"]
                mask_ds.append(batch["padding_mask"])
            else:
                output_notDS = output_NOTdownsampled["set"]
                mask_notds.append(batch["padding_mask"])
            if rec_pts.shape != batch["events"].shape:
                return None
            if batch["padding_mask"] is not None:
                for j in range(self.args.input_feats):
                    rec_pts[:, :, j][batch["padding_mask"]] = -100
                    batch["events"][:, :, j][batch["padding_mask"]] = -100
                    sampled[:, :, j][batch["padding_mask"]] = -100
                    if batch["events"].shape[1] > self.args.p_dim:
                        output_DS[:, :, j][batch["padding_mask"]] = -100
                    else:
                        output_notDS[:, :, j][batch["padding_mask"]] = -100

            m.append(batch["events"].shape[1])
            all_ref.append(batch["events"].cpu().detach())
            posteriors.append(post.cpu().detach())
            all_rec.append(rec_pts.cpu().detach())
            all_sampled.append(sampled.cpu().detach())
            all_sampled_latent.append(z_sampled.unsqueeze(-1).cpu().detach())
            # eps == output["z"]: always [B, L, N, D] regardless of
            # n_gaussian_mixture, since DecoderBlock.sample() already selects
            # the sampled mixture component before returning z. Unsqueeze to
            # match all_sampled_latent's shape ([B, L, N, D, 1]).
            all_latent.append(eps.unsqueeze(-1).cpu().detach())
            if output_DS is not None:
                all_DS.append(output_DS.cpu().detach())
            if output_notDS is not None:
                all_notDS.append(output_notDS.cpu().detach())

            if batch["cond_feats"] is not None:
                batch["cond_feats"][:, 1] = (
                    (batch["cond_feats"][:, 1] * self.cfg.max_points).cpu().detach()
                )

            all_cond.append(batch["cond_feats"].cpu().detach())
            all_y.append(batch["layer_cond"].cpu().detach())
            all_mint.append(batch["mint_per_layer"].cpu().detach())
            if batch["padding_mask"] is not None:
                padding_mask.append(batch["padding_mask"].cpu().detach())
            count += rec_pts.shape[0]
            torch.cuda.empty_cache()
            if i == 20:
                break

        # pad latents to a common length: batches have different numbers of points
        max_shape_list = [lat.shape[2] for lat in all_latent]
        max_shape = np.max(np.array(max_shape_list))

        all_latens_padded = [
            pad_tensor_to_max(lat, max_shape, pad_dim=2) for lat in all_latent
        ]
        all_sampled_latent_padded = [
            pad_tensor_to_max(lat, max_shape, pad_dim=2) for lat in all_sampled_latent
        ]
        post_padded = [
            pad_tensor_to_max(
                p,
                max_shape,
                pad_dim=2,
                extra_dim_condition=(lambda t: t.shape[4])
                if self.args.n_gaussian_mixture > 1
                else None,
            )
            for p in posteriors
        ]

        max_shape = max(m)
        ref = torch.cat(
            [pad_tensor_to_max(ref_pts, max_shape, pad_dim=1) for ref_pts in all_ref],
            dim=0,
        )

        rec = torch.cat(
            [pad_tensor_to_max(rec_pts, max_shape, pad_dim=1) for rec_pts in all_rec],
            dim=0,
        )

        if output_DS is not None:
            ds = torch.cat(
                [pad_tensor_to_max(ds, max_shape, pad_dim=1) for ds in all_DS], dim=0
            )
        if output_notDS is not None:
            not_ds = torch.cat(
                [
                    pad_tensor_to_max(not_ds, max_shape, pad_dim=1)
                    for not_ds in all_notDS
                ],
                dim=0,
            )

        sampled = torch.cat(
            [
                pad_tensor_to_max(sampled, max_shape, pad_dim=1)
                for sampled in all_sampled
            ],
            dim=0,
        )

        if self.include_time:
            mint_per_layer = np.array(torch.cat(all_mint, dim=0))

        # Pad padding mask — uses ones instead of -100
        mask = None
        if batch["padding_mask"] is not None:
            mask = torch.cat(
                [
                    pad_tensor_to_max(pm, max_shape, pad_dim=1, pad_value=1)
                    for pm in padding_mask
                ],
                dim=0,
            )
            if mask_ds != []:
                mask_ds = torch.cat(
                    [
                        pad_tensor_to_max(pm, max_shape, pad_dim=1, pad_value=1)
                        for pm in mask_ds
                    ],
                    dim=0,
                )
            if mask_notds != []:
                mask_notds = torch.cat(
                    [
                        pad_tensor_to_max(pm, max_shape, pad_dim=1, pad_value=1)
                        for pm in mask_notds
                    ],
                    dim=0,
                )

        layer = torch.cat(
            [pad_tensor_to_max(y, max_shape, pad_dim=1) for y in all_y], dim=0
        )

        others = {
            "all_DS": ds if output_DS is not None else None,
            "mask_DS": mask_ds if output_DS is not None else None,
            "all_notDS": not_ds if output_notDS is not None else None,
            "mask_notDS": mask_notds if output_notDS is not None else None,
            "latent": torch.cat(all_latens_padded, dim=0),
            "sampled_latent": torch.cat(all_sampled_latent_padded, dim=0),
            "gaussian_mixture": self.args.n_gaussian_mixture,
            "posteriors": torch.cat(post_padded, dim=0),
            "cond_feats": torch.cat(all_cond, dim=0)
            if all_cond[0] is not None
            else None,
            "pm": mask,
            "layer": layer,
            "mint_per_layer": mint_per_layer if self.include_time else None,
        }
        cd = L2_ChamferEval_1000(rec, ref)
        f1score, p1, p2 = F1Score(rec, ref)

        all_res = {
            "cd": cd,
            "f1score": f1score.mean(),
            "loss": loss_dict["loss"].item(),
            "rec_loss": loss_dict["rec_loss"].item(),
            "kl_loss": loss_dict["kl_loss"].item(),
        }
        all_res["kl_loss"] = loss_dict["kl_loss"].item()
        return all_res, ref, rec, sampled, others

    def encode_for_latent(
        self,
        x,
        ctx=None,
        y=None,
        mint_per_layer=None,
        padding=None,
        return_params=False,
    ):
        """
        Encode the point cloud.
        Args:
            x: input point cloud
            ctx: optional conditioning [B, ctx_dim]
            y: optional label embeddings [B, N, y_dim]
            return_params: return the posterior (mu, log_var) instead of a sample
        Returns:
            encoded: [B, N, input_dim]
        """
        kwargs = _n_points_and_top_points_per_layer(y, padding)
        self.__calculate_block_mask(padding, **kwargs)
        with torch.no_grad():
            return self.model.encode_for_latent(
                x,
                ctx=ctx,
                y=y,
                mint_per_layer=mint_per_layer if self.include_time else None,
                padding=padding,
                masks=self.masks,
                effective_p_dim=self.effective_p_dim,
                return_params=return_params,
            )

    def reco(self, ref_pts, c=None, y=None, mint_per_layer=None, padding=None):
        kwargs = _n_points_and_top_points_per_layer(y, padding)
        self.__calculate_block_mask(padding, **kwargs)
        with torch.no_grad():
            out = self.model(
                ref_pts,
                ctx=c,
                y=y,
                mint_per_layer=mint_per_layer,
                padding=padding,
                masks=self.masks,
                effective_p_dim=self.effective_p_dim,
            )
        return out["set"], out["z"]

    def sample(
        self,
        B=1,
        N=None,
        c=None,
        y=None,
        mint_per_layer=None,
        padding=None,
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
            deterministic: if True, use mu instead of random eps
        Returns:
            recon: [B, N, input_dim]
        """
        kwargs = _n_points_and_top_points_per_layer(y, padding)
        self.__calculate_block_mask(padding, **kwargs)
        self.model.eval()
        with torch.no_grad():
            sample, z_sampled = self.model.sample(
                B=B,
                N=N,
                c=c,
                y=y,
                mint_per_layer=mint_per_layer if self.include_time else None,
                padding=padding,
                masks=self.masks,
                effective_p_dim=self.effective_p_dim,
                device=device,
                z=z,
            )
        return sample, z_sampled


def _token_padding(effective_p_dim, B, N, device):
    """Padding mask over the sequence the network actually returns, mirroring the
    one `PointCloudNet.forward` builds: True = padding/invalid."""
    padding = torch.zeros(B, N, dtype=torch.bool, device=device)
    if effective_p_dim is not None and effective_p_dim < N:
        padding[:, effective_p_dim:] = True
    return padding


def _lag1_autocorr(a, valid):
    """Correlation between token k and token k+1 along the sequence index, each
    shower centred over its own valid tokens.

    The VAE latent is an ORDERED sequence and its decoder reads that order: mu has
    lag-1 autocorrelation ~0.66, and permuting one shower's tokens drops the decoded
    Esum from 1.00 x REC to 0.91, the score of a prior draw. The 2026-09-10 run's
    samples sat at 0.04, flat at every lag -- the right multiset of tokens in no
    particular order -- because the denoiser had no absolute positional input.
    Nothing in the loss can see that; this can. Compare a sample's value against the
    same statistic on the training target, which is the number to reach."""
    mu = (a * valid).sum(dim=1, keepdim=True) / valid.sum(dim=1, keepdim=True).clamp(
        min=1
    )
    c = (a - mu) * valid
    return (c[:, :-1] * c[:, 1:]).sum() / c.pow(2).sum().clamp(min=1e-12)


class LatentTargetMixin:
    """What the latent diffusion is trained on: the VAE posterior *sample* or its
    *mean*. Shared by the EDM (`CaloHadronic_Attention`) and MeanFlow
    (`CaloHadronic_FM`) models, which differ only in the generative objective.

    'sample' is the posterior draw z = mu + sigma * eps. At kl_weight 1e-2 that draw
    is 92-96% eps (sigma ~0.97 against std(mu) ~0.2) and a fresh eps is drawn at every
    encoding, so the model can only learn the N(0,1) part: measured 2026-09-09, its
    samples decode no better than a prior draw, while all the information sits in the
    per-token structure of mu. 'mu' is the posterior mean divided by `latent_scale`,
    its measured std, so sigma_data 1 still holds; `latent_to_decoder` then maps a
    sample back to latent_scale * x + latent_noise_std * eps, the noisy input the
    decoder was trained on (decode(mu + sigma eps) reproduces REC; decode(mu) alone
    leaves the hit marginals slightly too narrow).

    Code default 'sample' keeps archived runs and their state dicts loading.
    """

    def _init_latent_target(self, args):
        """Read the latent-target config and register the buffers it needs. Call
        once `self.net` is set. The keys live under `latent:`; archived configs carry
        them under `mean_flow:` and are still honoured."""
        cfg = getattr(args, "latent", None)
        if _cfg_get(cfg, "target", None) is None:
            mf = getattr(args, "mean_flow", None)
            target = _cfg_get(mf, "latent_target", "sample")
            scale = _cfg_get(mf, "latent_scale", "auto")
            noise = _cfg_get(mf, "latent_noise", "auto")
            batches = _cfg_get(mf, "latent_stats_batches", 8)
        else:
            target = _cfg_get(cfg, "target", "sample")
            scale = _cfg_get(cfg, "scale", "auto")
            noise = _cfg_get(cfg, "noise", "auto")
            batches = _cfg_get(cfg, "stats_batches", 8)

        self.latent_target = str(target)
        if self.latent_target not in ("sample", "mu"):
            raise NotImplementedError(
                f"latent.target must be 'sample' or 'mu', got {self.latent_target!r}"
            )
        if self.latent_target == "mu" and self.net != "LatentDiff":
            raise NotImplementedError(
                "latent.target 'mu' reads the VAE posterior mean, so it only applies "
                f"to the latent diffusion; net is {self.net!r}"
            )
        if self.latent_target != "mu":
            return
        self.latent_stats_batches = int(batches)
        self._latent_scale_auto = scale == "auto"
        self._latent_noise_auto = noise == "auto"
        # Buffers so the measured values travel with the checkpoint.
        self.register_buffer(
            "latent_scale",
            torch.tensor(1.0 if self._latent_scale_auto else float(scale)),
        )
        self.register_buffer(
            "latent_noise_std",
            torch.full(
                (args.transformer.input_feats,),
                0.0 if self._latent_noise_auto else float(noise),
            ),
        )
        self.register_buffer(
            "latent_stats_ready",
            torch.tensor(
                0 if (self._latent_scale_auto or self._latent_noise_auto) else 1
            ),
        )

    def encode_latent_target(
        self, vae_mod, events, ctx=None, y=None, mint_per_layer=None, padding=None
    ):
        """The tensor the model is trained on, from the (unwrapped) VAE: the posterior
        sample for latent target 'sample', mu / latent_scale for 'mu'."""
        if self.latent_target == "sample":
            return vae_mod.encode_for_latent(
                events, ctx=ctx, y=y, mint_per_layer=mint_per_layer, padding=padding
            )
        mu, _ = vae_mod.encode_for_latent(
            events,
            ctx=ctx,
            y=y,
            mint_per_layer=mint_per_layer,
            padding=padding,
            return_params=True,
        )
        return mu / self.latent_scale

    def latent_to_decoder(self, x):
        """Map a sample back to what the VAE decoder was trained on (see the class
        docstring). Identity for 'sample'."""
        if self.latent_target == "sample":
            return x
        return self.latent_scale * x + self.latent_noise_std * torch.randn_like(x)

    @torch.no_grad()
    def measure_latent_stats(self, vae_mod, loader, device, logger=None):
        """Fill the 'auto' latent buffers from the first `latent_stats_batches` batches
        of `loader` (valid latent tokens only): latent_scale = std of mu over all
        elements, latent_noise_std = per-dim mean of the posterior sigma, and with
        analytic_skip_per_dim the skip's sigma_data_dim = per-dim std of
        mu / latent_scale. Under DDP every rank must call this; the sums are
        all-reduced so all ranks agree."""
        if self.latent_target != "mu" or bool(self.latent_stats_ready):
            return
        D = self.latent_noise_std.shape[0]
        # n, sum, sum^2 | sigma sums (D) | per-dim mu sums (D) | per-dim mu^2 sums (D)
        acc = torch.zeros(3 + 3 * D, dtype=torch.float64, device=device)
        for batch in islice(loader, self.latent_stats_batches):
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key][0].to(device)
            mu, logvar = vae_mod.encode_for_latent(
                batch["events"],
                ctx=batch["cond_feats"],
                y=batch["layer_cond"],
                mint_per_layer=batch["mint_per_layer"],
                padding=batch["padding_mask"],
                return_params=True,
            )
            P = vae_mod.effective_p_dim or mu.shape[1]
            m = mu[:, :P].reshape(-1, D).double()
            s = torch.exp(0.5 * logvar[:, :P]).reshape(-1, D).double()
            acc[0] += m.shape[0]
            acc[1] += m.sum()
            acc[2] += m.pow(2).sum()
            acc[3 : 3 + D] += s.sum(0)
            acc[3 + D : 3 + 2 * D] += m.sum(0)
            acc[3 + 2 * D :] += m.pow(2).sum(0)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(acc)
        n = acc[0] * D  # elements
        mean = acc[1] / n
        std = (acc[2] / n - mean.pow(2)).clamp(min=0).sqrt()
        if self._latent_scale_auto:
            self.latent_scale.fill_(float(std))
        if self._latent_noise_auto:
            self.latent_noise_std.copy_((acc[3 : 3 + D] / acc[0]).float())
        if getattr(self, "mf_skip_per_dim", False):
            mean_d = acc[3 + D : 3 + 2 * D] / acc[0]
            std_d = (acc[3 + 2 * D :] / acc[0] - mean_d.pow(2)).clamp(min=0).sqrt()
            # std of the training target mu / latent_scale, per dim; the floor keeps
            # s(tau) away from 0 at tau = 0 for a dim with no spread at all.
            sigma_d = (std_d / float(self.latent_scale)).clamp(min=1e-3)
            self.diffusion.sigma_data_dim.copy_(sigma_d.float())
        self.latent_stats_ready.fill_(1)
        if logger is not None:
            logger.info(
                "latent target 'mu': %d tokens -> latent_scale %.4f (mu mean %.4f), "
                "latent_noise_std mean %.4f [%.3f, %.3f]",
                int(acc[0]),
                float(self.latent_scale),
                float(mean),
                float(self.latent_noise_std.mean()),
                float(self.latent_noise_std.min()),
                float(self.latent_noise_std.max()),
            )
            if getattr(self, "mf_skip_per_dim", False):
                sd = self.diffusion.sigma_data_dim
                logger.info(
                    "MeanFlow analytic_skip_per_dim: sigma_data_dim min %.4f median %.4f "
                    "max %.4f, %d of %d dims below 0.1",
                    float(sd.min()),
                    float(sd.median()),
                    float(sd.max()),
                    int((sd < 0.1).sum()),
                    int(sd.numel()),
                )


class CaloHadronic_Attention(LatentTargetMixin, nn.Module):
    def __init__(self, args, net, distillation=False):
        super().__init__()
        self.args = args
        self.distillation = distillation
        self.net = net
        # What the diffusion is trained on (latent.target); see LatentTargetMixin.
        self._init_latent_target(args)
        # Token-order diagnostics, the EDM counterpart of CaloHadronic_FM's mf/*
        # block. training.py logs whatever lands in `mf_metrics`, keyed by name, so
        # nothing there needs to know which model produced them.
        self.mf_metric_iter = int(getattr(args, "log_iter", 2000))
        self.mf_metrics = {}
        network = PointCloudNet(args)

        self.diffusion = Denoiser(
            network,
            point_dim=args.transformer.input_feats,
            sigma_data=args.model_sigma.sigma_data,
            device=args.device,
            distillation=distillation,
            diffusion_loss=args.diffusion_loss,
            diffusion_way=args.diffusion_way,
        )
        self.MASK = BlockMaskCache(
            neighbors=30,  # does not matter
            p_dim=self.args.downsample_factor,
            device=args.device,
        )
        if args.device == "cuda":
            patch_no_tensor_aliasing_reason()
            self.diffusion.set_compiled_model(torch.compile(network, dynamic=True))
        self.kld = KLDloss()

        gmm_cfg = getattr(args, "gmm_prior", None)
        if gmm_cfg is not None and gmm_cfg.enabled:
            self.gmm_prior = ConditionalGMMPrior(
                cond_dim=args.transformer.cond_feats,
                z_dim=args.transformer.input_feats,
                n_components=gmm_cfg.n_components,
                hidden_dim=gmm_cfg.hidden_dim,
            ).to(args.device)
        else:
            self.gmm_prior = None

    def __calculate_block_mask(
        self,
        mask,
        **kwargs,
    ) -> None:
        B = mask.shape[0]

        # Compute per-sample valid point counts
        valid_counts = torch.sum(~mask, dim=1)  # [B]

        # Dynamic p_dim: cap at actual min valid points in batch, but don't exceed downsample_factor (p_dim for VAE)
        self.effective_p_dim = min(
            int(valid_counts.min().item()), self.args.downsample_factor
        )
        pad_learnable = torch.ones(B, self.effective_p_dim, dtype=torch.bool).to(
            self.args.device
        )

        #  mask for latent
        self.block_mask = self.MASK.get(
            pad_learnable,
            pad_learnable,
            self.effective_p_dim,
            self.effective_p_dim,
            type="layer",
            **kwargs,
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
        it=0,
        **kwargs,
    ):
        """
        Args:
            x:  Input point clouds, (B, N, d).
            noise: Noise point cloud (B, N, d).
            sigma: Time (B, ).
            cond_feats: conditioning features (B, C)
        """
        z = cond_feats

        self.block_mask = None
        if self.net == "LatentDiff":
            self.__calculate_block_mask(padding_mask, **kwargs)

        loss_diffusion = self.diffusion.loss(
            x,
            noise,
            sigma,
            context=z,
            block_mask=self.block_mask if self.net == "LatentDiff" else None,
            effective_p_dim=self.effective_p_dim if self.net == "LatentDiff" else None,
            y=y,
            mint_per_layer=mint_per_layer,
            padding=padding_mask,  # so the layer-occupancy profile can skip pad hits
        ).mean()  # diffusion loss
        loss = loss_diffusion
        if self.gmm_prior is not None:
            loss = loss + self.args.gmm_prior.nll_weight * self.gmm_prior.nll(x, z)
        if (
            self.net == "LatentDiff"
            and self.mf_metric_iter > 0
            and (it % self.mf_metric_iter == 0 or it == 10)
        ):
            self._order_metrics(x, z, y, mint_per_layer, padding_mask)
        return loss

    @torch.no_grad()
    def _order_metrics(self, x, cond_feats, y, mint_per_layer, padding_mask):
        """Token-order diagnostics for the latent diffusion: the lag-1 correlation
        along the latent token index of a SAMPLE against the same statistic on the
        training target. See `_lag1_autocorr` for why this matters -- the VAE latent
        is an ordered sequence, the decoder reads that order, and neither the loss
        (flat at ~230 for a whole run) nor the pooled latent histograms can see a
        model that produces the right multiset of tokens in random order.

        The MeanFlow counterpart is mf/autocorr_1step, which is free because one-step
        sampling is a single forward pass. EDM has no one-step sample, so this runs
        the configured sampler (num_steps / sampler / sigma_min / sigma_max) on the
        training batch. That is ~2 * num_steps network evaluations once every
        `log_iter` training steps, i.e. well under 1% of the step budget.
        """
        # `sample()` rebuilds block_mask / effective_p_dim for its own call and the
        # training step's values must survive it; dropout is off for the sample.
        saved = (self.block_mask, self.effective_p_dim)
        was_training = self.training
        try:
            self.eval()
            sample = self.sample(
                cond_feats,
                x.shape[1],
                self.args,
                y=y,
                mint_per_layer=mint_per_layer,
                padding_mask=padding_mask,
            )
        finally:
            self.block_mask, self.effective_p_dim = saved
            self.train(was_training)

        B, N, _ = x.shape
        valid = (
            (~_token_padding(self.effective_p_dim, B, N, x.device))
            .unsqueeze(-1)
            .to(x.dtype)
        )
        self.mf_metrics = {
            "edm/autocorr_sample": _lag1_autocorr(sample, valid).item(),
            "edm/autocorr_target": _lag1_autocorr(x, valid).item(),
        }

    def append_dims(self, x, target_dims):
        """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
        dims_to_append = target_dims - x.ndim
        if dims_to_append < 0:
            raise ValueError(
                f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
            )
        return x[(...,) + (None,) * dims_to_append]

    def to_d(self, x, sigma, denoised):
        """Converts a denoiser output to a Karras ODE derivative."""
        return (x - denoised) / self.append_dims(sigma, x.ndim)

    @torch.no_grad()
    def sample_midpoint(
        self,
        model,
        x,
        sigmas,
        extra_args=None,
        callback=None,
        disable=None,
        s_churn=0.0,
        s_tmin=0.0,
        s_tmax=float("inf"),
        s_noise=1.0,
    ):
        """
        Implements the midpoint (2nd-order Runge-Kutta) method for sampling,
        similar to Karras et al. (2022).
        """

        def append_dims(x, target_dims):
            """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
            dims_to_append = target_dims - x.ndim
            if dims_to_append < 0:
                raise ValueError(
                    f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
                )
            return x[(...,) + (None,) * dims_to_append]

        def to_d(x, sigma, denoised):
            """Converts a denoiser output to a Karras ODE derivative."""
            return (x - denoised) / append_dims(sigma, x.ndim)

        extra_args = {} if extra_args is None else extra_args
        s_in = x.new_ones([x.shape[0]])

        for i in trange(len(sigmas) - 1, disable=disable):
            # Optional noise / churn
            gamma = (
                min(s_churn / (len(sigmas) - 1), 2**0.5 - 1)
                if s_tmin <= sigmas[i] <= s_tmax
                else 0.0
            )
            eps = torch.randn_like(x) * s_noise
            sigma_hat = sigmas[i] * (gamma + 1)
            if gamma > 0:
                x = x + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5

            # derivative at current point
            denoised = model(x, sigma_hat * s_in, **extra_args)
            d = to_d(x, sigma_hat, denoised)

            dt = sigmas[i + 1] - sigma_hat

            # Midpoint estimate
            x_mid = x + d * (dt / 2)
            denoised_mid = model(x_mid, (sigma_hat + dt / 2) * s_in, **extra_args)
            d_mid = to_d(x_mid, sigma_hat + dt / 2, denoised_mid)

            # full step using midpoint derivative
            x = x + d_mid * dt

            if callback is not None:
                callback(
                    {
                        "x": x,
                        "i": i,
                        "sigma": sigmas[i],
                        "sigma_hat": sigma_hat,
                        "denoised": denoised,
                        "denoised_mid": denoised_mid,
                    }
                )

        return x

    @torch.no_grad()
    def sample(
        self,
        cond_feats,
        num_points,
        config,
        y=None,
        mint_per_layer=None,
        padding_mask=None,
    ):
        z = cond_feats  # B, C
        if self.gmm_prior is not None:
            eps = self.gmm_prior.sample_eps(
                torch.empty(
                    z.size(0),
                    num_points,
                    config.transformer.input_feats,
                    device=z.device,
                ),
                cond_feats=z,
            )
        else:
            eps = torch.randn(
                [z.size(0), num_points, config.transformer.input_feats], device=z.device
            )
        if self.net == "LatentDiff":
            # EDM assumes sigma_max >> sigma_data; here sigma_max is only ~2x the latent
            # std, so start from the forward marginal std
            # sqrt(sigma_max^2 + latent_std^2), otherwise showers come out
            # systematically narrow.
            # latent_std is 1 for a KL-to-N(0,1) VAE.
            latent_std = getattr(config, "latent_std", 1.0)
            x_T = eps * (config.sigma_max**2 + latent_std**2) ** 0.5
        else:
            x_T = eps * config.sigma_max
        if self.net == "LatentDiff":
            kwargs = _n_points_and_top_points_per_layer(y, padding_mask)
            self.__calculate_block_mask(padding_mask, **kwargs)
        extra_args = {
            "context": z,
            "block_mask": self.block_mask if self.net == "LatentDiff" else None,
            "effective_p_dim": self.effective_p_dim
            if self.net == "LatentDiff"
            else None,
            "y": y,
            "mint_per_layer": mint_per_layer,
            "padding": padding_mask,
        }
        # The ctx/layer conditioning is constant along the sampling trajectory, so
        # compute it once here instead of once per denoiser call (NFE times).
        extra_args["static_cond"] = self.diffusion.inner_model.compute_static_cond(
            z,
            y=y,
            effective_p_dim=extra_args["effective_p_dim"],
            N=num_points,
            padding=padding_mask,
            mint_per_layer=mint_per_layer,
        )
        if not self.distillation:
            sigmas = K.sampling.get_sigmas_karras(
                config.num_steps,
                config.sigma_min,
                config.sigma_max,
                rho=config.rho,
                device=z.device,
            )
            if config.sampler == "euler":
                x_0 = K.sampling.sample_euler(
                    self.diffusion, x_T, sigmas, extra_args=extra_args, disable=True
                )
            elif config.sampler == "heun":
                x_0 = K.sampling.sample_heun(
                    self.diffusion,
                    x_T,
                    sigmas,
                    extra_args=extra_args,
                    s_churn=config.s_churn,
                    s_noise=config.s_noise,
                    disable=True,
                )
            elif config.sampler == "midpoint":
                x_0 = self.sample_midpoint(
                    self.diffusion,
                    x_T,
                    sigmas,
                    extra_args=extra_args,
                    disable=True,
                )
            elif config.sampler == "dpmpp_2m":
                x_0 = K.sampling.sample_dpmpp_2m(
                    self.diffusion, x_T, sigmas, extra_args=extra_args, disable=True
                )
            elif config.sampler == "dpmpp_2s_ancestral":
                x_0 = K.sampling.sample_dpmpp_2s_ancestral(
                    self.diffusion, x_T, sigmas, extra_args=extra_args, disable=True
                )
            elif config.sampler == "sample_euler_ancestral":
                x_0 = K.sampling.sample_euler_ancestral(
                    self.diffusion, x_T, sigmas, extra_args=extra_args, disable=True
                )
            elif config.sampler == "sample_lms":
                x_0 = K.sampling.sample_lms(
                    self.diffusion, x_T, sigmas, extra_args=extra_args, disable=True
                )
            elif config.sampler == "sample_dpmpp_2m_sde":  # this should be midpoint
                x_0 = K.sampling.sample_dpmpp_2m_sde(
                    self.diffusion,
                    x_T,
                    sigmas,
                    extra_args=extra_args,
                    disable=True,
                    solver_type="midpoint",
                )
            else:
                raise NotImplementedError("Sampler not implemented")

        else:  # one step for consistency model
            x_0 = self.diffusion.forward(x_T, config.sigma_max, context=z, y=y)

        return x_0


def _cfg_get(cfg, key, default):
    """Read `key` off a config node, falling back to `default` when the node or
    the key is absent (the MeanFlow block is optional in the yaml configs)."""
    if cfg is None:
        return default
    value = getattr(cfg, key, None)
    return default if value is None else value


class MeanFlowNet(nn.Module):
    """Average-velocity network u_theta(z, r, t) for MeanFlow (arXiv:2505.13447).

    Thin wrapper around `PointCloudNet` -- the same transformer used by the EDM
    denoiser -- that adds the second time input. `PointCloudNet` embeds a single
    time through c_noise = log(sigma)/4; MeanFlow needs the *pair* (r, t), so both
    are embedded here and injected via the `static_cond` hook, which is added to
    every token exactly like the network's own timestep embedding. The network is
    then fed sigma = 1 (c_noise = 0) so its internal timestep embedding degenerates
    into a constant bias and the EDM time parametrisation is bypassed entirely.
    """

    def __init__(
        self,
        inner_model,
        time_dim,
        d_model,
        fourier_scale=1.0,
        analytic_skip=False,
        sigma_data=1.0,
        per_dim_sigma=False,
        out_feats=None,
        out_scale=1.0,
        zero_init_head=False,
    ):
        super().__init__()
        self.inner_model = inner_model
        # Same trick as `Denoiser`: keep the callable used in the forward pass out
        # of the module tree so a torch.compile wrapper does not prefix the
        # state_dict keys with "_orig_mod.".
        self._net = [inner_model]
        self.analytic_skip = analytic_skip
        self.sigma_data = sigma_data
        # Per-dimension sigma_data for the skip (see skip_coeff): a (D,) buffer,
        # `sigma_data` in every dim until `CaloHadronic_FM.measure_latent_stats`
        # fills it with the per-dim std of the training target. Only registered
        # when asked for, so archived state dicts keep loading strict.
        self.per_dim_sigma = bool(per_dim_sigma)
        if self.per_dim_sigma:
            if out_feats is None:
                raise ValueError("MeanFlowNet: per_dim_sigma needs out_feats (D)")
            self.register_buffer(
                "sigma_data_dim", torch.full((int(out_feats),), float(sigma_data))
            )
        # EDM's c_out: the network output is scaled by out_scale before the skip is
        # added. With the analytic skip only a small residual is left; at O(1) output
        # scale the head learns to cancel it and the body stops receiving gradient.
        self.out_scale = float(out_scale)
        if zero_init_head:
            # F = 0 exactly at init (DiT-style), so the model starts at the analytic
            # Gaussian solution instead of having to unlearn a random 0.2-rms output.
            head = inner_model.decoder[-1]
            with torch.no_grad():
                head.weight.zero_()
                if head.bias is not None:
                    head.bias.zero_()

        def _time_embedding():
            return nn.Sequential(
                K.layers.FourierFeatures(1, time_dim, std=fourier_scale),
                nn.Linear(time_dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, d_model),
                # Pins the magnitude at which t (and r) reach the tokens, since they are
                # summed raw into `cond`; otherwise their t-dependence can shrink until
                # the net is blind to them. mf/t_sensitivity measures whether the
                # information content survives.
                nn.LayerNorm(d_model, eps=1e-6),
            )

        self.t_embeding = _time_embedding()
        self.r_embeding = _time_embedding()

    def set_compiled_model(self, compiled_model):
        """Route the forward passes through `compiled_model` (a torch.compile
        wrapper of `inner_model`), leaving the registered module tree untouched."""
        self._net[0] = compiled_model

    def skip_coeff(self, r, t):
        """Closed-form average velocity of the flow for a Gaussian latent, as a
        scalar per sample multiplying z_t.

        Along z_tau = (1 - tau) x + tau eps the marginal variance is
        s(tau) = (1 - tau)^2 sigma_data^2 + tau^2, and the probability-flow ODE of
        the matching Gaussian integrates in closed form to
        z_r = z_t sqrt(s(r) / s(t)), so the average velocity over [r, t] is

            u_gauss(z_t, r, t) = (1 - sqrt(s(r) / s(t))) / (t - r) * z_t.

        Handing this to the network as a skip leaves it only the non-Gaussian
        residual to predict. Without it the network has to *manufacture* the
        cancellation between the contraction (tau < 0.5) and re-expansion
        (tau > 0.5) phases of the path: at (r=0, t=1) the two endpoint marginals
        coincide, s(0) = s(1) = sigma_data^2 = 1, so u_gauss is identically zero
        and the whole answer is the residual, a ~10x cancellation of an O(1) field.
        That corner is the only point one-step sampling evaluates, and it is where
        the un-skipped parametrisation collapses.

        With `per_dim_sigma` the same formula runs with sigma_data_dim[d] per latent
        dimension: on the mu target the per-dim std runs from ~0.01 to ~4 around the
        scalar 1, so a scalar skip is wrong in every dim and the network has to
        carry the trivial per-dim Gaussian rescaling itself (measured 2026-09-10 on
        the 09-09 run: F_mid 1.2, of which ~1.2 is exactly this term, and the ~100
        dead dims come out 5-10x too broad at one step). In the sigma -> 0 limit
        the per-dim skip is u = z_t / t, i.e. the one-step sample is 0 there
        exactly, and F is left with the non-Gaussian residual only.

        Returns (B, 1, 1) for the scalar sigma_data, (B, 1, D) per dim; either
        broadcasts against z_t (B, N, D).
        """
        if self.per_dim_sigma:
            sd2 = self.sigma_data_dim.pow(2).reshape(1, -1)  # (1, D)
            r, t = r.reshape(-1, 1), t.reshape(-1, 1)  # (B, 1)
        else:
            sd2 = self.sigma_data**2

        def s(tau):
            return (1.0 - tau) ** 2 * sd2 + tau**2

        s_r, s_t = s(r), s(t)
        gap = t - r
        # r == t (75% of a training batch at the default flow_ratio) is the
        # removable singularity of the expression above; its limit is the
        # instantaneous Gaussian velocity coefficient s'(t) / (2 s(t)).
        safe_gap = torch.where(gap.abs() < 1e-6, torch.ones_like(gap), gap)
        k = torch.where(
            gap.abs() < 1e-6,
            (t - (1.0 - t) * sd2) / s_t,
            (1.0 - torch.sqrt(s_r / s_t)) / safe_gap,
        )
        return k.reshape(k.shape[0], 1, -1)

    def forward(
        self,
        z,
        r,
        t,
        context,
        y=None,
        mint_per_layer=None,
        block_mask=None,
        effective_p_dim=None,
        static_cond=None,
        padding=None,
    ):
        """
        Args:
            z: point cloud on the flow path at time t, (B, N, d).
            r: start time of the averaging interval, (B, ).
            t: end time of the averaging interval, (B, ).
            context: conditioning features, (B, F).
        Returns:
            u: average velocity, (B, N, d).
        """
        batch_size, N, _ = z.shape
        if static_cond is None:
            static_cond = self.inner_model.compute_static_cond(
                context,
                y=y,
                effective_p_dim=effective_p_dim,
                N=N,
                padding=padding,
                mint_per_layer=mint_per_layer,
            )
        # (t, t - r), not (t, r): MeanFlow paper Table 1c. The gap multiplies du/dt in
        # the identity. `r_embeding` keeps its name for state-dict compatibility; it
        # embeds the interval, in [0, 1] since sample_t_r gives 0 <= r <= t <= 1.
        cond = (
            static_cond
            + self.t_embeding(t.reshape(batch_size, 1, 1))
            + self.r_embeding((t - r).reshape(batch_size, 1, 1))
        )
        out, _ = self._net[0](
            z,
            None,  # sigma: unused, the net is built with use_sigma_embedding=False
            context=context,
            y=y,
            mint_per_layer=mint_per_layer,
            block_mask=block_mask,
            effective_p_dim=effective_p_dim,
            static_cond=cond,
            padding=padding,
        )
        if self.out_scale != 1.0:
            out = out * self.out_scale
        if self.analytic_skip:
            out = out + self.skip_coeff(r, t) * z
        return out


class CaloHadronic_FM(LatentTargetMixin, nn.Module):
    """MeanFlow counterpart of `CaloHadronic_Attention` (arXiv:2505.13447).

    Same transformer, conditioning, block-mask handling and gmm prior as the EDM
    version; only the generative objective changes. Instead of a denoiser the
    network parametrises the *average* velocity of the flow-matching path

        z_t = (1 - t) x + t eps,   v(z_t, t) = eps - x,   t = 0 data, t = 1 noise
        u(z_t, r, t) = 1/(t - r) int_r^t v(z_tau, tau) dtau

    trained with the MeanFlow identity

        u(z_t, r, t) = v(z_t, t) - (t - r) d/dt u(z_t, r, t)

    where d/dt u = (grad_z u) v + partial_t u is a directional derivative. It is
    estimated by finite differences by default: flex attention has no forward-mode
    AD rule, so `torch.func.jvp` (mean_flow.use_jvp) is unusable here and is force-
    disabled when the network is compiled. Because u is an *average* velocity, the
    ODE integrates exactly over a finite interval, so sampling needs one (or a few)
    network evaluations instead of the ~2 * num_steps of the Karras samplers.

    Drop-in for `CaloHadronic_Attention`: same `__init__(args, net)`, same
    `get_loss(...)` / `sample(...)` signatures, and the trainable module is again
    exposed as `self.diffusion` (training.py builds the optimizer from it).
    """

    def __init__(self, args, net):
        super().__init__()
        self.args = args
        self.net = net
        # MeanFlow carries its own times through t_embeding / r_embeding, so the EDM
        # sigma embedding is not used.
        network = PointCloudNet(args, use_sigma_embedding=False)

        mf_cfg = getattr(args, "mean_flow", None)
        # (t, r) are drawn from a logit-normal, as in the paper; t = max, r = min.
        self.mf_t_mean = float(_cfg_get(mf_cfg, "t_mean", -0.4))
        self.mf_t_std = float(_cfg_get(mf_cfg, "t_std", 1.0))
        # Fraction of the batch kept with r != t. The remainder collapses to r = t,
        # for which the objective reduces to plain (instantaneous) flow matching.
        self.mf_flow_ratio = float(_cfg_get(mf_cfg, "flow_ratio", 0.25))
        # Adaptive loss weighting w = 1 / (||delta||^2 + c)^p, stop-gradient on w.
        self.mf_adaptive_p = float(_cfg_get(mf_cfg, "adaptive_p", 0.5))
        self.mf_adaptive_c = float(_cfg_get(mf_cfg, "adaptive_c", 1e-3))
        # Number of sampling steps (1 = single network evaluation).
        self.mf_num_steps = int(_cfg_get(mf_cfg, "num_steps", 1))
        # Fourier scale of the (r, t) embeddings. The EDM timestep embedding uses 16,
        # which is tuned for log(sigma)/4 inputs; on t in [0, 1] that oscillates ~16x
        # across the whole range, leaving no smooth structure in t to extrapolate
        # along -- and one-step sampling evaluates u at (r=0, t=1), a corner the
        # logit-normal draw almost never visits.
        self.mf_fourier_scale = float(_cfg_get(mf_cfg, "time_fourier_scale", 1.0))
        # 'uniform' or 'logit_normal' (the paper's default). Uniform covers the r->0 /
        # t->1 corner that one-step sampling evaluates; logit-normal(-0.4, 1.0) puts <1%
        # of draws near either end.
        self.mf_t_dist = str(_cfg_get(mf_cfg, "t_dist", "uniform"))
        self.mf_use_jvp = bool(_cfg_get(mf_cfg, "use_jvp", False))
        self.mf_fd_eps = float(_cfg_get(mf_cfg, "fd_eps", 1e-3))
        self.mf_compile = bool(_cfg_get(mf_cfg, "compile", True))
        if self.mf_compile and self.mf_use_jvp:
            # flex attention has no forward-mode AD rule (verified on torch 2.10:
            # torch.func.jvp raises "attempted to apply meta conversion to view
            # functorch tensor"), and a compiled graph has none either. Disable jvp
            # up front rather than burning an iteration on the runtime fallback.
            logger.warning(
                "MeanFlow: mean_flow.compile=true is incompatible with "
                "mean_flow.use_jvp=true; falling back to the finite-difference "
                "estimate of du/dt."
            )
            self.mf_use_jvp = False

        # Hand the network the closed-form Gaussian average velocity as a skip and let
        # it learn only the residual (see MeanFlowNet.skip_coeff). Default off so
        # configs without the key keep the parametrisation they were trained under.
        self.mf_analytic_skip = bool(_cfg_get(mf_cfg, "analytic_skip", False))
        # Per-dimension sigma_data in that skip, measured from the training target
        # (see MeanFlowNet.skip_coeff). Only meaningful on the mu target, whose
        # per-dim std is far from 1. Default off: archived state dicts have no
        # sigma_data_dim buffer.
        self.mf_skip_per_dim = bool(_cfg_get(mf_cfg, "analytic_skip_per_dim", False))
        # Output preconditioning (see MeanFlowNet). Code defaults are raw output and
        # default init; the repo config sets out_scale 0.15 / zero_init_head true.
        self.mf_out_scale = float(_cfg_get(mf_cfg, "out_scale", 1.0))
        self.mf_zero_init_head = bool(_cfg_get(mf_cfg, "zero_init_head", False))

        # What the flow is trained on (latent.target); see LatentTargetMixin.
        self._init_latent_target(args)
        if self.mf_skip_per_dim and self.latent_target != "mu":
            raise NotImplementedError(
                "mean_flow.analytic_skip_per_dim needs latent target 'mu' "
                "(the per-dim std is measured together with latent_scale)"
            )
        self.diffusion = MeanFlowNet(
            network,
            time_dim=args.transformer.time_dim,
            d_model=args.transformer.d_model,
            fourier_scale=self.mf_fourier_scale,
            analytic_skip=self.mf_analytic_skip,
            sigma_data=float(_cfg_get(args.model_sigma, "sigma_data", 1.0)),
            per_dim_sigma=self.mf_skip_per_dim,
            out_feats=args.transformer.input_feats,
            out_scale=self.mf_out_scale,
            zero_init_head=self.mf_zero_init_head,
        )
        self.MASK = BlockMaskCache(
            neighbors=30,  # does not matter
            p_dim=self.args.downsample_factor,
            device=args.device,
        )
        # Both forward passes of the finite-difference estimate go through the
        # compiled network, so the eager fallback below is only for use_jvp=true.
        if args.device == "cuda" and self.mf_compile:
            patch_no_tensor_aliasing_reason()
            self.diffusion.set_compiled_model(torch.compile(network, dynamic=True))
        self.kld = KLDloss()
        self.effective_p_dim = None
        self.block_mask = None

        self.mf_metric_iter = int(getattr(args, "log_iter", 2000))
        self.mf_metrics = {}  # one-step diagnostics, read by training.py

        gmm_cfg = getattr(args, "gmm_prior", None)
        if gmm_cfg is not None and gmm_cfg.enabled:
            self.gmm_prior = ConditionalGMMPrior(
                cond_dim=args.transformer.cond_feats,
                z_dim=args.transformer.input_feats,
                n_components=gmm_cfg.n_components,
                hidden_dim=gmm_cfg.hidden_dim,
            ).to(args.device)
        else:
            self.gmm_prior = None

    def __calculate_block_mask(
        self,
        mask,
        **kwargs,
    ) -> None:
        B = mask.shape[0]

        # Compute per-sample valid point counts
        valid_counts = torch.sum(~mask, dim=1)  # [B]

        # Dynamic p_dim: cap at actual min valid points in batch, but don't exceed downsample_factor (p_dim for VAE)
        self.effective_p_dim = min(
            int(valid_counts.min().item()), self.args.downsample_factor
        )
        pad_learnable = torch.ones(B, self.effective_p_dim, dtype=torch.bool).to(
            self.args.device
        )

        #  mask for latent
        self.block_mask = self.MASK.get(
            pad_learnable,
            pad_learnable,
            self.effective_p_dim,
            self.effective_p_dim,
            type="layer",
            **kwargs,
        )

    def _latent_padding(self, B, N, device):
        """See the module-level `_token_padding`."""
        return _token_padding(self.effective_p_dim, B, N, device)

    @torch.no_grad()
    def _one_step_metrics(self, x, noise, net_kwargs):
        """Diagnostics for the (r=0, t=1) corner, the only point `num_steps=1`
        evaluates and the one the loss cannot see: on the 2026-09-04 run |u| there
        decayed to ~0 -- one-step sampling returning its own input, a prior draw --
        while the loss improved monotonically.

        With the scalar sigma_data = 1, skip_coeff(0, 1) is exactly 0, so `u_corner`
        measures the network residual with or without `mean_flow.analytic_skip` and
        stays comparable across runs. With the per-dim skip the corner coefficient is
        1 - sigma_d, so `u_corner` is the full average velocity there (~1 on the mu
        target, where most dims are dead) and `F_corner` is the residual.
        """
        B, N, _ = x.shape
        ones = torch.ones(B, device=x.device, dtype=x.dtype)
        zeros = torch.zeros_like(ones)

        # z_t at t = 1 is the noise end of the path, where the sampler starts.
        # Dropout is left as the caller has it; toggling eval() would compile a
        # second graph, and these average over B * N * d.
        u_corner = self.diffusion(noise, zeros, ones, **net_kwargs)
        u_diag = self.diffusion(noise, ones, ones, **net_kwargs)

        # The network's own contribution on a real path point, away from the corner: u
        # minus the closed-form Gaussian part, at (r=0, t=0.5). u_corner alone cannot
        # tell "collapsed at the corner only" from "the network outputs nothing
        # anywhere".
        halves = torch.full_like(ones, 0.5)
        z_mid = 0.5 * x + 0.5 * noise
        u_mid = self.diffusion(z_mid, zeros, halves, **net_kwargs)
        f_mid = u_mid - self.diffusion.skip_coeff(zeros, halves) * z_mid
        f_corner = u_corner - self.diffusion.skip_coeff(zeros, ones) * noise

        valid = (~self._latent_padding(B, N, x.device)).unsqueeze(-1).to(x.dtype)
        n_valid = valid.expand_as(x).sum().clamp(min=1)

        def rms(a):
            return ((a.pow(2) * valid).sum() / n_valid).sqrt()

        def pooled_std(a):
            pooled = (a * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)  # (B, d)
            return pooled.std(dim=0).mean()

        std_data = pooled_std(x)
        std_1step = pooled_std(noise - u_corner)  # z_0 = z_1 - (1 - 0) * u

        self.mf_metrics = {
            "mf/u_corner": rms(u_corner).item(),  # ~0.10 healthy, ~0 collapsed
            # (scalar skip on a unit-variance latent; == F_corner there)
            "mf/F_corner": rms(
                f_corner
            ).item(),  # residual at (0, 1): the collapse alarm
            "mf/u_diag": rms(u_diag).item(),  # ~0.95 control, held through the collapse
            "mf/F_mid": rms(f_mid).item(),  # ~0.01 = network contributes nothing
            # Weak signal (0.90 when collapsed): a prior draw already reproduces
            # most of this latent's pooled spread. Read u_corner for the alarm.
            "mf/std_ratio_1step": (std_1step / std_data.clamp(min=1e-8)).item(),
            # std of the attention logits, recorded by MHFlexAttention during the
            # forward passes above. ~1-3 is healthy; far above, the softmax is a hard
            # argmax and everything downstream stops moving. Watch this, not the total
            # grad norm.
            "mf/qk_logit_scale": self._qk_logit_scale(),
            # How much the conditioning varies with t and r, relative to its own
            # constant part. Under ~0.1 the residual F is pinned to an
            # (r, t)-independent field, so 1-step, multi-step and a prior draw decode
            # alike. Needs no data.
            "mf/t_sensitivity": self._cond_sensitivity(
                self.diffusion.t_embeding, x.device
            ),
            "mf/r_sensitivity": self._cond_sensitivity(
                self.diffusion.r_embeding, x.device
            ),
            # Token order along the latent index; see _lag1_autocorr.
            "mf/autocorr_1step": _lag1_autocorr(noise - u_corner, valid).item(),
            "mf/autocorr_target": _lag1_autocorr(x, valid).item(),
        }

    @torch.no_grad()
    def _cond_sensitivity(self, embedding, device, n_grid: int = 101):
        """RMS variation of a time embedding across t in [0, 1], divided by the RMS
        of its constant part. 1 = the t-dependent and constant parts are comparable;
        0 = the embedding is a constant vector and the network cannot see t."""
        t = torch.linspace(0.0, 1.0, n_grid, device=device).reshape(-1, 1, 1)
        e = embedding(t).reshape(n_grid, -1)  # (n_grid, d_model)
        const = e.mean(0)
        var = (e - const).pow(2).mean(1).sqrt().mean()
        const_rms = const.norm() / const.numel() ** 0.5
        return float(var / const_rms.clamp(min=1e-8))

    def _qk_logit_scale(self):
        """Largest attention logit scale over the denoiser blocks, read out of the
        non-persistent buffers `MHFlexAttention` fills when `track_qk_scale` is on."""
        scales = [
            float(m.qk_logit_scale)
            for m in self.diffusion.inner_model.modules()
            if getattr(m, "track_qk_scale", False)
        ]
        return max(scales) if scales else 0.0

    def sample_t_r(self, batch_size, device):
        """Draw the MeanFlow time pair (t, r) with r <= t, both in [0, 1]."""
        if self.mf_t_dist == "uniform":
            u = torch.rand(batch_size, 2, device=device)
        else:
            u = torch.sigmoid(
                torch.randn(batch_size, 2, device=device) * self.mf_t_std
                + self.mf_t_mean
            )
        t = u.max(dim=1).values
        r = u.min(dim=1).values
        keep = torch.rand(batch_size, device=device) < self.mf_flow_ratio
        r = torch.where(keep, r, t)
        return t, r

    def _u_and_dudt(self, z_t, r, t, v, net_kwargs):
        """u(z_t, r, t) together with its total derivative along the path,
        d/dt u = (grad_z u) v + partial_t u."""

        def fn(z_in, t_in):
            return self.diffusion(z_in, r, t_in, **net_kwargs)

        if self.mf_use_jvp:
            try:
                return torch.func.jvp(fn, (z_t, t), (v, torch.ones_like(t)))
            except Exception as exc:  # noqa: BLE001
                self.mf_use_jvp = False
                logger.warning(
                    "MeanFlow: forward-mode AD (torch.func.jvp) failed (%s). "
                    "Falling back to a finite-difference estimate of du/dt.",
                    exc,
                )

        # Finite differences along (v, 1); step backwards where t + h would leave [0, 1].
        h = torch.full_like(t, self.mf_fd_eps)
        h = torch.where(t + h > 1.0, -h, h)
        h_b = h.reshape(-1, 1, 1)

        # The two evaluations MUST see the same dropout masks (EncoderBlock does not
        # forward `dropout`, so attention always runs at p=0.1): a mask difference
        # divided by h blows up the bootstrap target. Rewinding the RNG shares the
        # masks, as a single forward-mode pass would; both calls run with grad enabled
        # so they take the same path through the compiled graph.
        rng_cpu = torch.get_rng_state()
        rng_cuda = torch.cuda.get_rng_state(z_t.device) if z_t.is_cuda else None
        u = fn(z_t, t)
        torch.set_rng_state(rng_cpu)
        if rng_cuda is not None:
            torch.cuda.set_rng_state(rng_cuda, z_t.device)
        u_shift = fn(z_t + h_b * v, t + h)

        dudt = (u_shift.detach() - u.detach()) / h_b
        return u, dudt

    def _net_kwargs(self, cond_feats, N, y, mint_per_layer, padding_mask):
        """Conditioning that is constant along the sampling trajectory / across the
        jvp, computed once (as `sample()` does for the EDM model)."""
        static_cond = self.diffusion.inner_model.compute_static_cond(
            cond_feats,
            y=y,
            effective_p_dim=self.effective_p_dim,
            N=N,
            padding=padding_mask,
            mint_per_layer=mint_per_layer,
        )
        return {
            "context": cond_feats,
            "static_cond": static_cond,
            "block_mask": self.block_mask if self.net == "LatentDiff" else None,
            "effective_p_dim": self.effective_p_dim
            if self.net == "LatentDiff"
            else None,
            "y": y,
            "mint_per_layer": mint_per_layer,
            "padding": padding_mask,
        }

    def get_loss(
        self,
        x,
        noise,
        sigma,
        cond_feats,
        padding_mask=None,
        y=None,
        mint_per_layer=None,
        it=0,
        **kwargs,
    ):
        """
        Args:
            x:  Input point clouds, (B, N, d).
            noise: Noise point cloud (B, N, d) -- the eps end of the flow path.
            sigma: unused, kept so the signature matches `CaloHadronic_Attention`;
                MeanFlow draws its own time pair (t, r).
            cond_feats: conditioning features (B, C)
        """
        z_ctx = cond_feats
        B, N, _ = x.shape

        self.block_mask = None
        self.effective_p_dim = None
        if self.net == "LatentDiff":
            self.__calculate_block_mask(padding_mask, **kwargs)

        t, r = self.sample_t_r(B, x.device)
        t_b = t.reshape(B, 1, 1)
        r_b = r.reshape(B, 1, 1)

        z_t = (1.0 - t_b) * x + t_b * noise
        v = noise - x  # instantaneous velocity of the path

        net_kwargs = self._net_kwargs(z_ctx, N, y, mint_per_layer, padding_mask)
        u, dudt = self._u_and_dudt(z_t, r, t, v, net_kwargs)

        if self.mf_metric_iter > 0 and (it % self.mf_metric_iter == 0 or it == 10):
            self._one_step_metrics(x, noise, net_kwargs)

        # MeanFlow identity; the target is a bootstrap, hence stop-gradient.
        target = (v - (t_b - r_b) * dudt).detach()
        err = u - target

        valid = (~self._latent_padding(B, N, x.device)).unsqueeze(-1).to(err.dtype)
        n_valid = valid.expand_as(err).flatten(1).sum(1).clamp(min=1)
        err_sq = (err.pow(2) * valid).flatten(1).sum(1) / n_valid  # (B, )

        # Adaptive weighting w = 1 / (||delta||^2 + c)^p, treated as a constant.
        weights = 1.0 / (err_sq.detach() + self.mf_adaptive_c).pow(self.mf_adaptive_p)
        loss = (weights * err_sq).mean()

        if self.gmm_prior is not None:
            loss = loss + self.args.gmm_prior.nll_weight * self.gmm_prior.nll(x, z_ctx)
        return loss

    @torch.no_grad()
    def sample(
        self,
        cond_feats,
        num_points,
        config,
        y=None,
        mint_per_layer=None,
        padding_mask=None,
    ):
        z_ctx = cond_feats  # B, C
        if self.gmm_prior is not None:
            eps = self.gmm_prior.sample_eps(
                torch.empty(
                    z_ctx.size(0),
                    num_points,
                    config.transformer.input_feats,
                    device=z_ctx.device,
                ),
                cond_feats=z_ctx,
            )
        else:
            eps = torch.randn(
                [z_ctx.size(0), num_points, config.transformer.input_feats],
                device=z_ctx.device,
            )
        # t = 1 is the noise end of the flow path itself, so x_T is the raw draw:
        # unlike EDM there is no sigma_max scaling to apply here.
        z = eps
        B = z.shape[0]

        self.block_mask = None
        self.effective_p_dim = None
        if self.net == "LatentDiff":
            kwargs = _n_points_and_top_points_per_layer(y, padding_mask)
            self.__calculate_block_mask(padding_mask, **kwargs)

        net_kwargs = self._net_kwargs(
            z_ctx, num_points, y, mint_per_layer, padding_mask
        )

        num_steps = max(
            int(
                _cfg_get(
                    getattr(config, "mean_flow", None), "num_steps", self.mf_num_steps
                )
            ),
            1,
        )
        # Uniform grid from noise (t = 1) to data (t = 0); each step is exact for the
        # average velocity, so one step already lands on the data end.
        times = torch.linspace(1.0, 0.0, num_steps + 1, device=z.device)
        for i in range(num_steps):
            t = torch.full((B,), float(times[i]), device=z.device, dtype=z.dtype)
            r = torch.full((B,), float(times[i + 1]), device=z.device, dtype=z.dtype)
            u = self.diffusion(z, r, t, **net_kwargs)
            z = z - (t - r).reshape(B, 1, 1) * u

        return z
