"""
Full generation-time benchmark for the CaloCHILL (PointCountFM point-count +
VAE + LMF) pipeline, matching the scope of AllShowers_private's
allshowers/unified_sampling.py `time_generation()` benchmark (used for the
"AllShowers" rows of the CPU/GPU timing table): each timed call samples the
per-layer point counts / min-times from the distilled one-step PointCountFM
student *and* then runs the diffusion sampling + VAE decode, using synthetic, uniform
conditioning batches (fixed incident energy per batch) rather than real showers.
This makes the "CaloCHILL" and "AllShowers" rows apples-to-apples: both
include their respective point-count submodel in the timed region.

Times cover only the two models' inference; they do not include host<->device
transfer beyond what the point-count model itself requires, or projecting hits
into the detector geometry.

Times at 9 fixed incident energies (10-90 GeV in steps of 10 GeV) by default. The
first --warmup batches at each (energy, NFE, batch size) point are discarded before
timing, since JIT compilation / memory allocation make them longer and more erratic.

Example:
    python scripts/evaluation/timing.py --device cuda --steps 1 50 --bs 1 16
    python scripts/evaluation/timing.py --device cpu --steps 1 50 --bs 1 16

--toy-points drops the point-count model out of the timed region
altogether: instead of drawing hit counts from it, every shower in the batch is given
exactly the requested total, spread over the layers with the shape of a real drawn
shower, so the axis can be walked in even steps all the way to max_points_pions. Since
the point-count stage costs the same at every shower size, it is timed once and added
back as a constant (the pointcount_time_per_sample_s column).

The point-count stage is the distilled PointCountFM student, which is a single
network evaluation by construction -- there is no step count to sweep, so the old
--pointcount-steps / --pointcount-count-steps flags are gone along with the
ShowerFlow path they configured.
(run from the repo root, e.g. via slurm_scripts/timing_evaluate.sh, since config paths are relative)
"""

import argparse
import csv
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from utils.constants import (
    Xmax_pions,
    Xmin_pions,
    Zmax_pions,
    Zmin_pions,
    max_points_per_layer_pions,
    max_points_pions,
    maximum_time,
)
from utils.dataset import get_layers
from utils.gen_utils import load_configs, load_models, resolve_edm_dir
from utils.misc import seed_all

path_to_repo = "./"
EDM_RUN_DEFAULT = "lmf_large_ds256"  # large LMF, d_s = 256 (paper)

# Distilled one-step PointCountFM student, the same checkpoint the evaluation uses
# clone of https://github.com/FLC-QU-hep/LayerFM, weights from https://huggingface.co/FLC-QU-hep/LayerFM
PCFM_ROOT = os.environ.get("LAYERFM_ROOT", "../LayerFM")
PCFM_CKPT = os.environ.get(
    "LAYERFM_CKPT",
    PCFM_ROOT + "/results/20260619_124529_LayerFM_PandT_/checkpoint_distill.pt",
)


def load_pointcount_model(device):
    """The per-layer point-count / min-time model: the distilled one-step
    PointCountFM student (NFE=1), i.e. the CaloCHILL counterpart to
    AllShowers' PointCountFM.

    Loading re-fits the input transforms on the full training set, so this costs
    ~30 s of start-up before any timing begins."""
    if PCFM_ROOT not in sys.path:
        sys.path.insert(0, PCFM_ROOT)
    from layerfm.inference_cond_file import load_from_checkpoint

    flow = load_from_checkpoint(
        PCFM_CKPT, distill=True
    )  # --pcfm-ckpt overrides PCFM_CKPT
    flow.eval()
    return flow.to(device)


def sample_point_counts(pointcount_model, cond_E, max_retries=5):
    """Sample per-layer hit counts / min-times from the distilled PointCountFM
    student, conditioned on (normalized) incident energy `cond_E` ([bs, 1]).

    The student takes raw GeV and returns physical, already inverse-transformed
    values, so min-time is divided by `maximum_time` to match the normalization the
    VAE / LMF conditioning expects (the same thing gen_utils.py does on its
    PointCountFM2 path)."""
    e_gev = (cond_E + 1) / 2 * 100

    def draw(energies_gev):
        with torch.no_grad():
            res = pointcount_model(energies_gev)
        hits = np.round(res[0].cpu().numpy()).clip(0).astype(int)
        # points-only LayerFM (no-time models) returns no min-times; they are unused there
        mint = (
            res[1].cpu().numpy() / maximum_time
            if len(res) > 1
            else np.zeros(hits.shape, np.float32)
        )
        return hits, mint

    hits_per_layer, mint_per_layer = draw(e_gev)

    # reject/resample showers that are zero total hits, exceed the VAE's fixed max
    # sequence length (max_points_pions), or have any single layer overflow the
    # per-layer positional-embedding table (max_points_per_layer_pions) -- all three
    # break batch construction / the VAE decode path if left unhandled
    for _ in range(max_retries):
        totals = hits_per_layer.sum(axis=1)
        per_layer_over = hits_per_layer.max(axis=1) > max_points_per_layer_pions
        invalid = np.where(
            (totals == 0) | (totals > max_points_pions) | per_layer_over
        )[0]
        if len(invalid) == 0:
            break
        retry_hits, retry_mint = draw(e_gev[invalid])
        hits_per_layer[invalid] = retry_hits
        mint_per_layer[invalid] = retry_mint

    # last resort if retries didn't converge: clip per-layer counts first (the hard
    # constraint from the positional-embedding table size), then scale down any
    # still-oversized shower total -- scaling only shrinks values further, so it
    # cannot undo the per-layer clip
    np.clip(hits_per_layer, 0, max_points_per_layer_pions, out=hits_per_layer)
    totals = hits_per_layer.sum(axis=1)
    over = totals > max_points_pions
    if over.any():
        scale = max_points_pions / totals[over]
        hits_per_layer[over] = np.floor(hits_per_layer[over] * scale[:, None]).astype(
            int
        )

    return hits_per_layer, mint_per_layer


def toy_point_counts(profile, total_points, cap=max_points_per_layer_pions):
    """Per-layer hit counts whose per-shower total is exactly `total_points`.

    `profile` is a batch of real per-layer counts (drawn once from the point-count
    model, outside any timed region); only its *shape* along the layer axis is kept,
    and it is rescaled to the requested total. Layers that hit the per-layer ceiling
    `cap` -- the size of the positional-embedding table -- hand their excess to the
    layers that still have room, which is what lets the total reach max_points_pions
    instead of saturating the way a plain multiplicative scale factor does.

    The showers are toy: a real shower of this size would have a different
    longitudinal profile. The tensor shapes, and so the cost, are those of a real
    shower with that many points, which is all the benchmark reads."""
    weights = np.asarray(profile, dtype=np.float64)
    bs, n_layers = weights.shape
    if total_points > n_layers * cap:
        raise ValueError(
            f"{total_points} points do not fit in {n_layers} layers of at most {cap}"
        )
    counts = np.zeros((bs, n_layers), dtype=np.int64)
    for _ in range(200):
        remaining = total_points - counts.sum(axis=1)
        if not (remaining > 0).any():
            break
        free = cap - counts
        w = np.where(free > 0, weights, 0.0)
        norm = w.sum(axis=1, keepdims=True)
        # a row whose profile is zero everywhere it still has room: fall back to
        # spreading the rest flat over the layers that are not full
        flat = (norm[:, 0] == 0) & (remaining > 0)
        if flat.any():
            w[flat] = (free[flat] > 0).astype(np.float64)
            norm = w.sum(axis=1, keepdims=True)
        share = w / norm * remaining[:, None]
        add = np.minimum(np.floor(share).astype(np.int64), free)
        # flooring can place nothing at all once the remainder is smaller than the
        # number of layers sharing it; give the largest share one point so the loop
        # always advances
        stuck = (add.sum(axis=1) == 0) & (remaining > 0)
        if stuck.any():
            idx = np.argmax(np.where(free > 0, share, -1.0)[stuck], axis=1)
            add[np.where(stuck)[0], idx] = 1
        counts += add
    totals = counts.sum(axis=1)
    if not (totals == total_points).all():
        raise RuntimeError(
            f"toy counts did not converge: totals {totals.min()}-{totals.max()} "
            f"for a target of {total_points}"
        )
    return counts


def synthetic_energy(energy_gev, bs, device):
    """A batch of `bs` identical incident energies, matching AllShowers'
    time_generation() (torch.full((batch_size, 1), energy))."""
    cond_E_gev = torch.full((bs, 1), float(energy_gev), device=device)
    return (cond_E_gev / 100) * 2 - 1


def build_batch(hits_per_layer, mint_per_layer, cfg_compressor, device):
    mint_b = torch.tensor(mint_per_layer, dtype=torch.float32, device=device)

    max_num_clusters = hits_per_layer.sum(axis=1).max()
    padding_mask = np.zeros((hits_per_layer.shape[0], max_num_clusters), dtype=bool)
    col_indices = np.arange(max_num_clusters)
    padding_mask[col_indices >= hits_per_layer.sum(axis=1)[:, None]] = True

    cond_N = torch.tensor(
        hits_per_layer.sum(axis=1) / max_points_pions,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(-1)

    y_cond = get_layers(hits_per_layer, 100)
    for j in range(y_cond.shape[0]):
        y_cond[j][padding_mask[j]] = 0
    cond_layer = torch.tensor(y_cond, dtype=torch.float32, device=device)

    n_points = int((~padding_mask).sum(axis=1).min())
    if cfg_compressor.vae.downsampling_flag and n_points > cfg_compressor.vae.p_dim:
        n_points = cfg_compressor.vae.p_dim

    padding_mask_t = torch.tensor(padding_mask, device=device)
    return cond_N, cond_layer, mint_b, padding_mask_t, n_points


def generate_batch(
    model,
    vae,
    cfg,
    num_points,
    cond_E,
    cond_N,
    cond_layer,
    mint_per_layer,
    padding_mask,
    stages=False,
):
    """Returns the shower and, with `stages`, the time the latent sampling ended at
    (after a synchronisation, so the two stages can be timed apart), else None."""
    cond_feats = torch.cat([cond_E, cond_N], dim=-1)
    t_latent = None
    with torch.no_grad():
        latent = model.sample(
            cond_feats,
            num_points,
            cfg,
            y=cond_layer.to(torch.long),
            mint_per_layer=mint_per_layer,
            padding_mask=padding_mask.bool(),
        )
        if getattr(model, "latent_target", "sample") == "mu":
            # mean_flow trained on the posterior mean: map back to the decoder input
            latent = model.latent_to_decoder(latent)
        if stages:
            if cond_feats.is_cuda:
                torch.cuda.synchronize()
            t_latent = time.perf_counter()
        shower, _ = vae.sample(
            z=latent,
            N=cond_layer.shape[1],
            c=cond_feats,
            y=cond_layer.to(torch.long),
            mint_per_layer=mint_per_layer,
            padding=padding_mask.bool(),
            device=cond_feats.device,
        )
    return shower, t_latent


def to_physical(shower, padding_mask, cfg_data):
    """gen_utils' post-processing of a generated batch (exp of the log energy / time, x_y_shift to mm,
    padding zeroed), on the device: AllShowers' timed call ends with the same step (Generator.forward
    inverse transforms + mask), so both timers stop at physical-unit points on the GPU (2026-10-14)."""
    if cfg_data.log_energy:
        shower[:, :, 2] = torch.exp(
            shower[:, :, 2] * cfg_data.log_var + cfg_data.log_mean
        )
    if shower.shape[-1] > 3:  # with time
        shower[:, :, 3] = torch.exp(
            shower[:, :, 3] * cfg_data.log_var_time + cfg_data.log_mean_time
        )
    shower[:, :, 0] = (shower[:, :, 0] + 1) / 2 * (Xmax_pions - Xmin_pions) + Xmin_pions
    shower[:, :, 1] = (shower[:, :, 1] + 1) / 2 * (Zmax_pions - Zmin_pions) + Zmin_pions
    shower[padding_mask.bool()] = 0
    return shower


# Geant4 mean number of points per shower at each timed incident energy [GeV]: local
# linear fit of hit count vs. energy within +/-2 GeV of each energy, over dataset
# files 1-10 (all_interactions_pions_regular_ECAL+HCAL_10-90GeV_{1..10}); the
# uncertainty on each value is at most ~11 points
GEANT4_MEAN_POINTS = {
    10.0: 510.9,
    20.0: 968.2,
    30.0: 1368.7,
    40.0: 1752.6,
    50.0: 2132.6,
    60.0: 2488.5,
    70.0: 2840.0,
    80.0: 3160.0,
    90.0: 3493.6,
}
# Geant4 mean shower size is close to linear in incident energy over 10-90 GeV; the inverted
# fit gives the incident energy a toy shower of N points is conditioned on (past ~3.5k points
# this extrapolates beyond the training range).
_G4_SLOPE, _G4_INTERCEPT = np.polyfit(
    sorted(GEANT4_MEAN_POINTS),
    [GEANT4_MEAN_POINTS[e] for e in sorted(GEANT4_MEAN_POINTS)],
    1,
)


def toy_energy_for_points(n_points):
    return float((n_points - _G4_INTERCEPT) / _G4_SLOPE)


def timed_call(
    pointcount_model,
    model,
    vae,
    cfg,
    cfg_compressor,
    cfg_data,
    cond_E,
    device,
    prebuilt=None,
    stages=False,
):
    """One full (point-count + shower generation) call. Returns
    (point_count_s, latent_s, decode_s, total_s); without `stages` the latent and
    decode stages are not timed apart, and latent_s holds the whole generation.

    `prebuilt` is a (hits_per_layer, mint_per_layer) pair prepared outside the timed
    region (--toy-points). The point-count model is then not called at all and the
    first returned number is ~0: only the LMF sampling + VAE decode is timed."""
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    if prebuilt is None:
        hits_per_layer, mint_per_layer = sample_point_counts(pointcount_model, cond_E)
    else:
        hits_per_layer, mint_per_layer = prebuilt

    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    cond_N, cond_layer, mint_b, padding_mask, n_points = build_batch(
        hits_per_layer, mint_per_layer, cfg_compressor, device
    )
    shower, t_latent = generate_batch(
        model,
        vae,
        cfg,
        n_points,
        cond_E,
        cond_N,
        cond_layer,
        mint_b,
        padding_mask,
        stages,
    )
    if not stages:  # physical units inside the timer, as AllShowers (the --stages breakdown leaves this out)
        to_physical(shower, padding_mask, cfg_data)

    if device == "cuda":
        torch.cuda.synchronize()
    t2 = time.perf_counter()
    if t_latent is None:
        t_latent = t2

    return t1 - t0, t_latent - t1, t2 - t_latent, t2 - t0


# Denoiser evaluations each sampler spends per solver step, measured by counting
# calls into Denoiser.forward (scripts/models/calochill.py sampler dispatch):
#   euler / dpmpp_2m  1 eval per step          -> NFE = num_steps
#   midpoint          2 evals per step         -> NFE = 2 * num_steps
#   heun              2 per step, 1 on the last -> NFE = 2 * num_steps - 1
EVALS_PER_STEP = {
    "euler": 1,
    "dpmpp_2m": 1,
    "sample_lms": 1,
    "sample_euler_ancestral": 1,
    "midpoint": 2,
    "heun": 2,
    "dpmpp_2s_ancestral": 2,
    "sample_dpmpp_2m_sde": 1,
}


def nfe_to_num_steps(nfe, sampler):
    """config.num_steps that spends exactly `nfe` denoiser evaluations.

    The two are NOT the same number: a 2nd-order sampler evaluates the network twice
    per step, so `--nfe 8` on midpoint is num_steps=4, not 8. Raises if the requested
    NFE is not reachable with this sampler.
    """
    per_step = EVALS_PER_STEP.get(sampler)
    if per_step is None:
        raise ValueError(f"unknown evals-per-step for sampler {sampler!r}")
    if sampler == "heun":  # 2n-1: the final step drops to Euler at sigma=0
        if nfe == 1:
            return 1
        if nfe % 2 == 0:
            raise ValueError(
                f"sampler 'heun' spends an odd number of evaluations (2*steps-1), "
                f"so NFE={nfe} is unreachable; try {nfe - 1} or {nfe + 1}"
            )
        return (nfe + 1) // 2
    if nfe % per_step:
        raise ValueError(
            f"sampler {sampler!r} spends {per_step} evaluations per step, so NFE={nfe} "
            f"is unreachable; use a multiple of {per_step} (nearest: {nfe - 1} or "
            f"{nfe + 1}), or a 1st-order sampler such as --sampler euler / dpmpp_2m "
            f"if you specifically need NFE={nfe}"
        )
    return nfe // per_step


def outlier_count(times, factor=2.0):
    """Repeats far above the median. Timing noise here is one-sided: a call is only
    ever slowed down (by a compile for a newly-seen padded length, or by contention),
    never sped up, so a high tail is the only kind that occurs."""
    med = np.median(times)
    return int((times > factor * med).sum()) if med > 0 else 0


def aggregate(per_sample, stat, repeats):
    """Per-column (point-count, latent, decode, total) value and its uncertainty, per sample.

    `mean` matches the AllShowers reference timings, which quote mean +/- sigma/sqrt(n).
    `median` is the default because a single first-occurrence torch.compile inside the
    timed region adds seconds to one repeat and drags the mean far off the typical
    call; the quoted uncertainty is then the equivalent robust spread,
    1.4826 * MAD / sqrt(n).
    """
    if stat == "mean":
        return per_sample.mean(axis=0), per_sample.std(axis=0) / np.sqrt(repeats)
    med = np.median(per_sample, axis=0)
    return med, 1.4826 * np.median(np.abs(per_sample - med), axis=0) / np.sqrt(repeats)


def time_calls(fn, warmup, repeats):
    for _ in range(warmup):
        fn()
    return np.array(
        [fn() for _ in range(repeats)]
    )  # (repeats, 4): pointcount, latent, decode, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--edm-dir",
        default=EDM_RUN_DEFAULT,
        help="LMF run folder holding the configs and ckpt_latest.pt: a name "
        "under logs/, or an absolute path",
    )
    parser.add_argument(
        "--nfe",
        "--steps",
        dest="nfe",
        type=int,
        nargs="+",
        default=[2, 100],
        help="number of denoiser evaluations (NFE) to time, e.g. --nfe 8 32. These "
        "are true network evaluations, not solver steps: with the default 2nd-order "
        "midpoint sampler each step costs 2 evaluations, so --nfe 8 runs "
        "config.num_steps=4, and odd NFE is unreachable. The default 2 and 100 are "
        "the old --steps 1 and 50 expressed in evaluations",
    )
    parser.add_argument("--bs", type=int, nargs="+", default=[1, 16])
    parser.add_argument(
        "--pcfm-ckpt",
        default=None,
        help="distilled LayerFM checkpoint (default PCFM_CKPT, points and time); for models without time "
        "use the points-only one, results/20260619_192637_LayerFM_P/checkpoint_distill.pt",
    )
    parser.add_argument("--sampler", default="midpoint")
    parser.add_argument(
        "--energies",
        type=float,
        nargs="+",
        default=None,
        help="incident energies [GeV] to time at (default: 10..90 in steps of 10)",
    )
    parser.add_argument(
        "--toy-points",
        type=int,
        nargs="+",
        default=[],
        help="time toy showers of exactly these total point counts instead of "
        "sweeping incident energies. The point-count model is not called inside the "
        "timed region at all: every shower in the batch is built with the requested "
        "number of points, spread over the layers with the shape of a real drawn "
        "shower and capped per layer at max_points_per_layer_pions, so the vs-points "
        f"axis can be stepped evenly up to max_points_pions ({max_points_pions}). Its "
        "constant cost is timed once and added back, so the y axis stays the full "
        "pipeline. Each shower is conditioned on the incident energy a real shower of "
        "that size would carry, which above ~3.5k points is "
        "past the top of the training range -- these showers are toy, and only their "
        "tensor shapes, hence their cost, are meant to be real",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="batches discarded before timing at each (energy, NFE, bs) point",
    )
    parser.add_argument(
        "--repeats", type=int, default=20, help="timed batches kept after warmup"
    )
    parser.add_argument("--seed", type=int, default=144)
    parser.add_argument(
        "--geant4-baseline",
        type=float,
        default=2.09,
        help="CPU Geant4 s/sample, for the speed-factor column",
    )
    parser.add_argument(
        "--stat",
        default="median",
        choices=["median", "mean"],
        help="how the timed repeats are reduced to one number: 'median' (default) is "
        "robust to the seconds-long torch.compile a newly-seen padded length costs if "
        "it lands inside the timed region; 'mean' matches the AllShowers reference "
        "timings but is dragged off by a single such call",
    )
    parser.add_argument("--cpu-threads", type=int, default=None)
    parser.add_argument(
        "--matmul-precision",
        default="high",
        choices=["highest", "high", "medium"],
        help="fp32 matmul kernel precision (torch.set_float32_matmul_precision): "
        "'highest' is true fp32, 'high' uses TF32 tensor cores, 'medium' uses bf16 "
        "inputs and is the fastest. The default matches what AllShowers' "
        "unified_sampling.py sets for the reference timings, and what "
        "generate.py generates with, so the comparison "
        "stays apples-to-apples; use 'highest' to reproduce fp32 timings",
    )
    parser.add_argument(
        "--stages",
        action="store_true",
        help="time the latent sampling and the VAE decompressor apart (one extra "
        "synchronisation between them; the conversion to physical units is left out of "
        "the timer) and write their columns, for the per-stage figure",
    )
    parser.add_argument(
        "--n-refinement-blocks",
        type=int,
        default=None,
        help="keep only the first N refinement blocks (last layers) of the VAE "
        "decompressor, for the ablation of the per-stage figure",
    )
    parser.add_argument("--out", default=None, help="CSV output path")
    args = parser.parse_args()
    energies = args.energies or list(range(10, 91, 10))
    toy_points = sorted(set(args.toy_points))
    if toy_points and max(toy_points) > max_points_pions:
        raise SystemExit(
            f"--toy-points above max_points_pions ({max_points_pions}): "
            f"{[n for n in toy_points if n > max_points_pions]}. The VAE was never "
            f"trained on longer showers and the decode path assumes that ceiling"
        )

    edm_dir = resolve_edm_dir(args.edm_dir)
    print(f"edm_dir: {edm_dir}")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")
    if args.cpu_threads is not None:
        torch.set_num_threads(args.cpu_threads)

    seed_all(args.seed)
    cfg, cfg_data, cfg_compressor = load_configs(
        edm_dir, args.device, args.matmul_precision
    )
    # decoder padding follows the batch, so its cost scales with shower size (the model pads to max_points otherwise)
    cfg_compressor.vae.dynamic_max_len = True
    cfg.sampler = args.sampler
    is_mean_flow = getattr(cfg.model, "generative_model", "edm") == "mean_flow"

    def set_nfe(nfe):
        """Point the sampler at `nfe` network evaluations; returns the step count."""
        if is_mean_flow:
            # LMF.sample reads mean_flow.num_steps, not cfg.num_steps /
            # cfg.sampler: one network evaluation per step, so NFE = num_steps
            cfg.mean_flow.num_steps = nfe
            cfg.num_steps = nfe
        else:
            cfg.num_steps = nfe_to_num_steps(nfe, args.sampler)
        return cfg.num_steps

    model, model_vae = load_models(cfg, cfg_data, cfg_compressor, edm_dir, args.device)
    compressor_mod = getattr(model_vae.model, "_orig_mod", model_vae.model)
    if args.n_refinement_blocks is not None:
        # reset dynamo so the shortened module list is traced fresh rather than
        # reusing the graph for the full stack
        torch._dynamo.reset()
        compressor_mod.out_refinement = torch.nn.ModuleList(
            list(compressor_mod.out_refinement)[: args.n_refinement_blocks]
        )
    if args.pcfm_ckpt:
        global PCFM_CKPT
        PCFM_CKPT = (
            args.pcfm_ckpt
            if args.pcfm_ckpt.startswith("/")
            else PCFM_ROOT + "/" + args.pcfm_ckpt
        )
    pointcount_model = load_pointcount_model(args.device)

    gpu_name = torch.cuda.get_device_name(0) if args.device == "cuda" else None
    print(
        f"device={args.device}  sampler={args.sampler}  incident energies: {energies} GeV"
    )
    if gpu_name is not None:
        print(f"GPU: {gpu_name}")
    if is_mean_flow:
        print(
            f"NFE: {args.nfe} (mean_flow: num_steps = NFE, --sampler ignored)  "
            f"batch sizes: {args.bs}  warmup={args.warmup}  repeats={args.repeats}"
        )
    else:
        print(
            f"NFE: {args.nfe} (= num_steps {[nfe_to_num_steps(n, args.sampler) for n in args.nfe]} "
            f"at {EVALS_PER_STEP[args.sampler]} eval(s)/step for sampler {args.sampler!r})  "
            f"batch sizes: {args.bs}  warmup={args.warmup}  repeats={args.repeats}"
        )
    print(f"point-count model: distilled PointCountFM student (NFE 1)  {PCFM_CKPT}")
    print(
        f"matmul precision: {args.matmul_precision if args.device == 'cuda' else 'n/a (cpu)'}  "
        f"decoder padding length: per batch"
    )
    hardware = f"GPU ({gpu_name})" if gpu_name else "CPU"
    # warm up
    for j in range(2):
        print(f"warmup {j + 1}/2")
        set_nfe(args.nfe[0])
        cond_E = synthetic_energy(energies[0], 128, args.device)
        timed_call(
            pointcount_model,
            model,
            model_vae,
            cfg,
            cfg_compressor,
            cfg_data,
            cond_E,
            args.device,
        )

    if toy_points:
        work = [(toy_energy_for_points(n), n) for n in toy_points]
        profile_energy = max(energies)
        print(
            f"toy sweep: {toy_points} points per shower, layer shape from "
            f"{profile_energy} GeV draws, conditioned at the energy each size implies "
            f"({', '.join(f'{toy_energy_for_points(n):.0f}' for n in toy_points)} GeV)"
        )
    else:
        work = [(e, 0) for e in energies]
    toy_profiles, toy_pc_cost = {}, {}

    rows = []
    for energy, toy_n in work:
        for bs in args.bs:
            cond_E = synthetic_energy(energy, bs, args.device)
            prebuilt = None
            if toy_n:
                if bs not in toy_profiles:
                    # one draw for the per-layer shape, and the point-count model's
                    # own cost at this batch size -- both outside the timed region,
                    # both the same at every toy size
                    prof_E = synthetic_energy(profile_energy, bs, args.device)
                    toy_profiles[bs] = sample_point_counts(pointcount_model, prof_E)

                    def pc_only(prof_E=prof_E):
                        if args.device == "cuda":
                            torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        sample_point_counts(pointcount_model, prof_E)
                        if args.device == "cuda":
                            torch.cuda.synchronize()
                        t1 = time.perf_counter()
                        return 0.0, 0.0, 0.0, t1 - t0

                    pc_rec = time_calls(pc_only, args.warmup, args.repeats) / bs
                    pc_val, pc_errs = aggregate(pc_rec, args.stat, args.repeats)
                    pc_t, pc_e = pc_val[3], pc_errs[3]
                    toy_pc_cost[bs] = (pc_t, pc_e)
                    print(
                        f"  point-count stage at bs={bs}: {pc_t:.4g}s/sample "
                        f"(+/-{pc_e:.2g}), constant across shower size, added to every "
                        f"toy point"
                    )
                prof_hits, prof_mint = toy_profiles[bs]
                prebuilt = (toy_point_counts(prof_hits, toy_n), prof_mint)
            for nfe in args.nfe:
                set_nfe(nfe)

                def fn(prebuilt=prebuilt):
                    return timed_call(
                        pointcount_model,
                        model,
                        model_vae,
                        cfg,
                        cfg_compressor,
                        cfg_data,
                        cond_E,
                        args.device,
                        prebuilt=prebuilt,
                        stages=args.stages,
                    )

                records = time_calls(fn, args.warmup, args.repeats)
                # shower size for the vs-points x axis, drawn outside the timed region.
                # Batch min/max too: the latent length follows the smallest shower and
                # the VAE decompressor pads to the largest.
                if toy_n:  # the toy counts are exact and already in hand
                    probe_totals = prebuilt[0].sum(axis=1)
                else:
                    probe_hits, _ = sample_point_counts(pointcount_model, cond_E)
                    probe_totals = probe_hits.sum(axis=1)
                mean_n_hits = float(probe_totals.mean())
                per_sample = records / bs
                val, err = aggregate(per_sample, args.stat, args.repeats)
                pc_mean, lat_mean, dec_mean, tot_mean = val
                pc_err, tot_std = err[0], err[3]
                gen_mean = lat_mean + dec_mean
                if toy_n:
                    # only LMF + decode were timed; put the point-count stage's
                    # measured constant back so the y axis is still the full pipeline
                    pc_mean, pc_err = toy_pc_cost[bs]
                    gen_mean, tot_mean = tot_mean, tot_mean + pc_mean
                    tot_std = float(np.hypot(tot_std, pc_err))
                n_out = outlier_count(per_sample[:, 3])
                if n_out:
                    print(
                        f"  note: {n_out}/{args.repeats} repeat(s) above 2x the median "
                        f"at E={energy} NFE={nfe} -- a first-occurrence torch.compile "
                        f"for a new padded length; excluded by --stat median"
                    )
                speed_factor = args.geant4_baseline / tot_mean
                rows.append(
                    {
                        "hardware": hardware,
                        "model": "CaloCHILL",
                        "energy_gev": energy,
                        "toy_points": toy_n,
                        "nfe": nfe,
                        "num_steps": cfg.num_steps,
                        "batch_size": bs,
                        "mean_n_hits": mean_n_hits,
                        "min_n_hits": int(probe_totals.min()),
                        "max_n_hits": int(probe_totals.max()),
                        "pointcount_time_per_sample_s": pc_mean,
                        "generation_time_per_sample_s": gen_mean,
                        "time_per_sample_s": tot_mean,
                        "time_per_sample_std_s": tot_std,
                        "speed_factor": speed_factor,
                    }
                )
                if args.stages:
                    rows[-1].update(
                        {
                            "n_refinement_blocks": len(compressor_mod.out_refinement),
                            "pointcount_time_per_sample_std_s": pc_err,
                            "latent_time_per_sample_s": lat_mean,
                            "latent_time_per_sample_std_s": err[1],
                            "decode_time_per_sample_s": dec_mean,
                            "decode_time_per_sample_std_s": err[2],
                        }
                    )
                print(
                    f"E={energy:>4.0f} GeV  "
                    + (f"toy{toy_n:<5d}" if toy_n else "        ")
                    + f"NFE={nfe:>3}  bs={bs:>3}  "
                    f"points={mean_n_hits:.0f} "
                    f"[{probe_totals.min()}-{probe_totals.max()}]  "
                    f"point-count={pc_mean:.4g}s  generation={gen_mean:.4g}s  "
                    f"total/sample={tot_mean:.4g}s (+/-{tot_std:.2g})  "
                    f"speed-factor={speed_factor:.1f}x"
                )
                if args.device == "cuda":
                    torch.cuda.empty_cache()

    # average over the full incident-energy range, per (NFE, batch size)
    by_combo = defaultdict(list)
    for r in rows:
        by_combo[(r["nfe"], r["batch_size"])].append(r["time_per_sample_s"])

    span = (
        f"toy sizes {toy_points[0]}-{toy_points[-1]} points"
        if toy_points
        else f"{energies[0]}-{energies[-1]} GeV"
    )
    print(f"\n--- average over {span} ---")
    summary_rows = []
    for (nfe_val, bs), vals in by_combo.items():
        vals = np.array(vals)
        avg, std = vals.mean(), vals.std()
        summary_rows.append(
            {
                "hardware": hardware,
                "model": "CaloCHILL",
                "nfe": nfe_val,
                "batch_size": bs,
                "avg_time_per_sample_s": avg,
                "std_across_energies_s": std,
                "speed_factor": args.geant4_baseline / avg,
            }
        )
        print(
            f"NFE={nfe_val:>3}  bs={bs:>3}  "
            f"avg time/sample={avg:.4g}s (+/-{std:.2g})  speed-factor={args.geant4_baseline / avg:.1f}x"
        )

    hardware_tag = f"GPU_{gpu_name.replace(' ', '')}" if gpu_name else "CPU"
    tag = f"{hardware_tag}_{int(time.time())}"
    out = args.out or (path_to_repo + f"files/timing/CaloCHILL_{tag}.csv")
    summary_out = os.path.splitext(out)[0] + "_summary.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(summary_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"saved: {out}")
    print(f"saved: {summary_out}")


if __name__ == "__main__":
    main()
