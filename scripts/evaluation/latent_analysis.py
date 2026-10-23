"""Latent-space study of the paper (figures 9 and 10, appendix on the latent information):

    python scripts/evaluation/latent_analysis.py encode --edm-dir <LMF run> --out <grid dir> [options]
    python scripts/evaluation/latent_analysis.py lda-ds-grid --run "ds N=<grid dir>" ... --out <dir> [options]

encode writes <grid dir>/tsne_latents.npz (name kept so the existing caches still load);
lda-ds-grid draws the LDA figures from one such file per latent size."""

import argparse
import os
import sys

import h5py
import matplotlib
import matplotlib.font_manager
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

matplotlib.use("Agg")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.constants import (  # noqa: E402
    Xmax_pions,
    Xmin_pions,
    max_points_pions,
    maximum_time,
)
from utils.dataset import (  # noqa: E402
    get_preprocessing_events,
    get_sorted_events_layercond,
)
from utils.misc import seed_all  # noqa: E402


def valid_mask(z):
    """[B, N] True where the token is not an all-zero padding slot."""
    return np.abs(z).sum(axis=-1) > 0


N_BLOCKS = 16


def block_features(z, n_blocks=N_BLOCKS):
    """
    [B, N, d] -> [B, n_blocks * d]: split each shower's *valid* tokens into
    n_blocks contiguous equal groups and average each.

    Blocks cover the whole latent at coarse resolution (the interior carries more
    energy information than the first and last tokens alone). Splitting on
    valid tokens rather than slots keeps it well defined when the latent length
    varies between batches (p_dim above the batch-min hit count).
    """
    z = np.asarray(z)
    B, N, d = z.shape
    m = valid_mask(z)
    out = np.zeros((B, n_blocks, d), np.float32)
    for i in range(B):
        idx = np.flatnonzero(m[i])
        if idx.size == 0:
            continue
        for b, part in enumerate(np.array_split(idx, n_blocks)):
            if part.size:
                out[i, b] = z[i, part].mean(0)
    return out.reshape(B, n_blocks * d)


def feature_of(d, key, feature):
    """Latent `key` (mu, sample) of an encode npz as the chosen --feature."""
    b = d[f"blocks_{key}"].reshape(len(d["energy_grid"]), N_BLOCKS, -1)
    # 'middle' drops the two outermost blocks at each end, which is
    # roughly the first and last tokens
    return (b if feature == "blocks" else b[:, 2:-2]).reshape(len(b), -1)


def load_pairs(d, feature, log):
    """
    Return [(name, X_enc, X_samp, enc_label, samp_label)], the energy each shower
    is labelled with, the group index used to split the encoded side, and the
    true incident energy (which is not the grid value: each grid point takes the
    showers nearest to it, and at 10 GeV the dataset only starts at 10.14 so the
    bin reaches upward).
    """
    if "blocks_mu" in d:  # encode, balanced grid of distinct showers

        def take(k):
            return feature_of(d, k, feature)

        e = np.asarray(d["energy_grid"], dtype=float)
        e_true = np.asarray(d["energy_true"], dtype=float)
        log(f"format: balanced energy grid, {len(e)} distinct showers")
        for lv in np.unique(e):
            sel = e == lv
            log(
                f"  {lv:>5.0f} GeV label: {sel.sum():>5d} showers, "
                f"true energy [{e_true[sel].min():.2f}, {e_true[sel].max():.2f}] GeV"
            )
        # every shower is distinct here, so each is its own group
        groups = np.arange(len(e))
        pairs = [
            (
                "train_space",
                take("mu"),
                take("sample"),
                r"$\mathbf{z}_\mu$ (encoded)",
                r"$\mathbf{z}$ (sampled)",
            ),
        ]
        # both sides are the same showers: the sampled latent is drawn with the
        # conditioning of the shower the encoded one comes from, index for index
        return pairs, e, e, groups, e_true, True
    raise SystemExit(
        "needs the balanced-grid npz with block features from latent_analysis.py encode"
    )


def build_labels(kind, e_true, e_grid, n_pts, n_bins, log):
    """
    Return (y, levels, level_names, target) where `target` is the continuous
    quantity the LDA axis is correlated against.

      energy         the grid point each shower was selected for
      npoints        quantile bins of the real hit count. Hits and energy are
                     ~0.88 correlated, so this largely re-finds the energy axis
    """
    if kind == "energy":
        levels = np.unique(np.round(e_grid).astype(int))
        y = np.searchsorted(levels, np.round(e_grid).astype(int))
        return y, levels, [f"{lv:g} GeV" for lv in levels], e_true

    if n_pts is None:
        raise SystemExit(
            "this npz has no n_points; regenerate it with latent_analysis.py encode"
        )
    n_pts = np.asarray(n_pts, dtype=float)

    levels = np.arange(n_bins)
    edges = np.quantile(n_pts, np.linspace(0, 1, n_bins + 1))
    y = np.clip(np.searchsorted(edges[1:-1], n_pts, side="right"), 0, n_bins - 1)
    names = []
    for b in levels:
        v = n_pts[y == b]
        names.append(f"{v.min():.0f}-{v.max():.0f} hits" if len(v) else f"bin {b}")
    log(f"  {kind}: {n_bins} bins, sizes {np.bincount(y, minlength=n_bins).tolist()}")
    return y, levels, names, n_pts


def make_lda(shrinkage):
    """The block feature has 16*z_dim (2048) dimensions, more than the number of
    showers per class, so the plain SVD solver overfits the
    within-class covariance. The eigen solver with Ledoit-Wolf shrinkage is the
    stable choice at this ratio and still supports transform(); pass
    --shrinkage none to get the unregularised SVD fit back."""
    if shrinkage == "none":
        return LDA(n_components=2)
    value = "auto" if shrinkage == "auto" else float(shrinkage)
    return LDA(n_components=2, solver="eigen", shrinkage=value)


def half_split(groups, seed):
    """[n] bool, True for the rows of a random half of the groups (the LDA fit set),
    so repeats of one shower never land on both sides."""
    uniq = np.unique(groups)
    fit = np.random.default_rng(seed).permutation(uniq)[: len(uniq) // 2]
    return np.isin(groups, fit)


# Helvetica Neue first, then the metric-compatible fallbacks, so the figure keeps
# its look on a machine that has the real font and degrades to Nimbus Sans (the
# URW Helvetica clone) where it does not. mathtext takes one family name rather
# than a fallback list, so resolve the first that is actually installed.
_STACK = [
    "Helvetica Neue",
    "HelveticaNeue",
    "Helvetica",
    "Nimbus Sans",
    "Arial",
    "DejaVu Sans",
]
_INSTALLED = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
_MATH = next((f for f in _STACK if f in _INSTALLED), "DejaVu Sans")

_LDA_RC = dict(
    {
        "font.family": "sans-serif",
        "font.sans-serif": _STACK,
        "mathtext.fontset": "custom",
        "mathtext.rm": _MATH,
        "mathtext.it": f"{_MATH}:italic",
        "mathtext.bf": f"{_MATH}:bold",
        "font.size": 9,
        "axes.labelsize": 14,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
    }
)


def row_projection(X_enc, X_samp, y, groups, seed, shrinkage, target):
    """
    Fit one LDA on half the encoded showers and project both sides through it.

    An LDA axis is an eigenvector, so its sign is arbitrary: the same structure
    can come back mirrored from one fit to the next, or between two rows of the
    same figure, for no physical reason. Energy and hit count are ~0.88
    correlated, so the energy-labelled and hit-count-labelled fits recover
    essentially the same plane (cross-row |corr| 0.94 on axis 1 and 0.78 on
    axis 2) -- and without a convention they are drawn mirror-imaged.

    The convention fixes that at the source: orient *every* axis so that it
    increases with the row's own target. It needs nothing from the other row,
    it is deterministic, and it makes the rows agree automatically.
    """
    in_fit = half_split(groups, seed)
    lda = make_lda(shrinkage).fit(X_enc[in_fit], y[in_fit])
    P_enc, P_samp = lda.transform(X_enc[~in_fit]), lda.transform(X_samp[~in_fit])

    sign = np.ones(P_enc.shape[1])
    for k in range(P_enc.shape[1]):
        r = np.corrcoef(P_enc[:, k], target[~in_fit])[0, 1]
        sign[k] = np.sign(r) if r else 1.0
    P_enc *= sign
    P_samp *= sign

    acc = tuple(
        float((lda.predict(X[~in_fit]) == y[~in_fit]).mean()) for X in (X_enc, X_samp)
    )
    return P_enc, P_samp, y[~in_fit], acc, ~in_fit, sign


def bin_colorbar(fig, ax, fraction, tgt, y_held, n_levels, label):
    """Plasma colourbar over the ordered class bins, each tick at its bin's mean."""
    ticks = [tgt[y_held == b].mean() for b in range(n_levels) if (y_held == b).any()]
    sm = plt.cm.ScalarMappable(
        cmap=plt.get_cmap("plasma"), norm=plt.Normalize(vmin=0, vmax=n_levels - 1)
    )
    cb = fig.colorbar(sm, ax=ax, fraction=fraction, pad=0.015)
    cb.outline.set_linewidth(0.6)
    cb.ax.tick_params(width=0.6, length=2.5, labelsize=7.5)
    cb.set_ticks(range(n_levels))
    cb.set_ticklabels([f"{t:.0f}" for t in ticks])
    cb.set_label(label, fontsize=9)


# ---------------------------------------------------------------- encode
_DOC_ENCODE = """
latent_analysis.py encode

Encoded and generated latents of one LMF run on a balanced incident-energy grid, the
input of lda-ds-grid:

  * `--energies 10 20 ... 90`, `--per-energy 500`: the showers of the held-out file
    whose true incident energy is closest to each grid point, so every energy carries
    the same weight; they are distinct showers.
  * only showers with at least `--min-points` points enter (default p_dim). The latent
    length is min(batch-min hits, p_dim); pin `--min-points` (256 in the paper) to get
    the same showers at every ds.
  * both latents come from the *same* showers and the same conditioning: the posterior
    mean z_mu (the space the LMF is trained on, mu / latent_scale for latent_target: mu)
    and the LMF sample.

Writes <out>/tsne_latents.npz: blocks_mu, blocks_sample (block_features), energy_true,
energy_grid, n_points.

    python scripts/evaluation/latent_analysis.py encode --edm-dir <LMF run> \\
        --energies 10 20 30 40 50 60 70 80 90 --per-energy 500 --min-points 256 \\
        --out gen_utils_plot/lda_grid_ds256_10to90_500each
(run from the repo root, the config paths are relative)
"""


# the ds256 mean_flow run
EDM_RUN_DEFAULT = "lmf_ds256"


def select_showers(path, energies, per_energy, min_points):
    """The `per_energy` showers closest in true incident energy to each grid point,
    among those with at least `min_points` hits. Returns the raw shower array in the
    layout gen_utils uses, their true energies, the grid point each belongs to, and
    their real hit counts."""
    with h5py.File(path, "r") as f:
        energy = np.array(f["energy"])[:, 0]
        n_points = np.array(f["n_points"]).reshape(-1).astype(int)
        big = np.flatnonzero(n_points >= min_points)
        picked, targets = [], []
        for e_target in energies:
            order = big[np.argsort(np.abs(energy[big] - e_target))[:per_energy]]
            if len(order) < per_energy:
                raise SystemExit(
                    f"{path}: only {len(order)} showers with >= {min_points} points "
                    f"near {e_target} GeV, asked for {per_energy}"
                )
            picked.append(np.sort(order))
            targets.append(np.full(len(order), float(e_target)))
            print(
                f"  {e_target:>4.0f} GeV: {len(order)} showers, true energy "
                f"[{energy[order].min():.2f}, {energy[order].max():.2f}] GeV, "
                f"points {n_points[order].min()}-{n_points[order].max()}"
            )
        idx = np.concatenate(picked)
        assert len(np.unique(idx)) == len(idx), "a shower landed in two energy bins"
        # h5py fancy indexing wants an increasing index list, so read in that order
        # and put the rows back in energy-bin order afterwards
        order = np.argsort(idx)
        events = np.array(f["events"][idx[order]])[np.argsort(order)]

    events = np.moveaxis(events, -1, -2)
    events[:, :, [1, 2]] = events[:, :, [2, 1]]
    events[:, :, 3] *= 1000
    events[events[:, :, 3] == 0] = 0
    return events, energy[idx], np.concatenate(targets), n_points[idx]


def prepare_batch(showers, e_gev, cfg, cfg_data, device, include_time=True):
    """The preprocessing gen_utils.get_reco_shower does before it calls the VAE, plus
    the conditioning the LatentDiff sampler needs. One batch of raw showers in, the
    tensors both panels are built from out."""
    showers, points_per_layer, mint_per_layer = get_preprocessing_events(
        showers.copy(), Xmin_pions, Xmax_pions, Xmin_pions, Xmax_pions, layers_axis=2
    )
    showers, layer_cond, n_points, padding_mask = get_sorted_events_layercond(
        showers, 100, points_per_layer, max_points_pions, y_idx=1, layer_idx=2
    )

    showers = showers.to(device).float()
    e = torch.from_numpy(e_gev / 100 * 2 - 1).unsqueeze(1).float()
    cond_feats = torch.cat([e, n_points], -1).to(device).float()

    if cfg_data.log_energy:
        showers[:, :, 3] = torch.log(showers[:, :, 3] + 1e-20)
        showers[:, :, 3] = (showers[:, :, 3] - cfg_data.log_mean) / cfg_data.log_var
    if include_time:
        showers[:, :, 4] = torch.log(showers[:, :, 4] + 1e-20)
        showers[:, :, 4] = (
            showers[:, :, 4] - cfg_data.log_mean_time
        ) / cfg_data.log_var_time

    padding = padding_mask.to(torch.bool).to(device)
    for j in range(showers.shape[0]):
        showers[j, :, 3][padding[j]] = 0
        if include_time:
            showers[j, :, 4][padding[j]] = 0
        layer_cond[j, :, 0][padding_mask[j].bool()] = 0

    x = showers[:, :, [0, 1, 3, 4] if include_time else [0, 1, 3]].to(torch.float32)
    y = layer_cond.to(torch.long).to(device)
    mint = (
        torch.from_numpy(mint_per_layer / maximum_time).to(torch.float32).to(device)
        if include_time
        else None
    )

    # latent length, exactly as gen_utils picks it for the generated side
    valid = (~padding).sum(dim=1)
    num_points = int(valid.min())
    if cfg.downsample_factor and num_points > cfg.downsample_factor:
        num_points = int(cfg.downsample_factor)
    return x, cond_feats, y, padding, mint, num_points


def main_encode():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edm-dir", default=EDM_RUN_DEFAULT)
    ap.add_argument(
        "--energies",
        type=float,
        nargs="+",
        default=list(range(10, 91, 10)),
        help="incident-energy grid [GeV] (default 10..90 in steps of 10)",
    )
    ap.add_argument(
        "--per-energy",
        type=int,
        default=150,
        help="distinct showers taken at each grid point",
    )
    ap.add_argument(
        "--file",
        type=int,
        default=99,
        help="dataset file the showers come from (99 is held out of training)",
    )
    ap.add_argument(
        "--min-points",
        type=int,
        default=None,
        help="minimum hits a shower needs to be selected (default p_dim, which "
        "makes every latent exactly p_dim tokens). Set it explicitly to compare "
        "two p_dim settings on the same showers -- the latent is then "
        "min(batch-min hits, p_dim), which is the regime the model normally runs in",
    )
    ap.add_argument("--num-steps", type=int, default=1, help="sampler steps (NFE)")
    ap.add_argument("--bs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True, help="output directory")
    args = ap.parse_args()

    # here, not at the top: gen_utils restyles matplotlib on import, which lda-ds-grid must not get
    from utils.gen_utils import load_configs, load_models, resolve_edm_dir

    os.makedirs(args.out, exist_ok=True)
    edm_dir = resolve_edm_dir(args.edm_dir)
    print(f"edm_dir: {edm_dir}")
    seed_all(args.seed)

    # production generation runs on TF32, not the timing script's bf16 "medium"
    cfg, cfg_data, cfg_compressor = load_configs(edm_dir, args.device, "high")
    cfg.sampler = "midpoint"
    cfg.num_steps = args.num_steps
    is_mean_flow = getattr(cfg.model, "generative_model", "edm") == "mean_flow"
    if is_mean_flow:
        cfg.mean_flow.num_steps = args.num_steps
    p_dim = int(cfg_compressor.vae.p_dim)

    min_points = args.min_points if args.min_points is not None else p_dim
    print(
        f"{args.per_energy} showers per energy, at least {min_points} points each "
        f"(p_dim {p_dim}):"
    )
    events, e_true, e_grid, n_pts = select_showers(
        cfg_data.dataset_path.format(args.file),
        args.energies,
        args.per_energy,
        min_points,
    )
    print(f"selected {len(events)} showers, {events.shape[1]} slots")

    model, vae = load_models(cfg, cfg_data, cfg_compressor, edm_dir, args.device)
    latent_target = getattr(model, "latent_target", "sample")
    print(f"latent target: {latent_target}  |  sampler steps: {args.num_steps}")

    mu_list, samp_list, tok_list = [], [], []
    for start in range(0, len(events), args.bs):
        sl = slice(start, min(start + args.bs, len(events)))
        x, cond_feats, y, padding, mint, num_points = prepare_batch(
            events[sl],
            e_true[sl],
            cfg,
            cfg_data,
            args.device,
            include_time=cfg_compressor.vae.include_time,
        )
        with torch.no_grad():
            mu = model.encode_latent_target(
                vae, x, ctx=cond_feats, y=y, mint_per_layer=mint, padding=padding
            )
            _, z = vae.reco(x, c=cond_feats, y=y, mint_per_layer=mint, padding=padding)
            latent = model.sample(
                cond_feats,
                num_points,
                cfg,
                y=y,
                mint_per_layer=mint,
                padding_mask=padding,
            )
            decoder_in = model.latent_to_decoder(latent)
        # the posterior draw z and latent_to_decoder are not stored, but both draw from the
        # global torch RNG, so they stay to keep the sampled latents of the paper bit-identical
        del z, decoder_in
        mu_list.append(
            mu.squeeze(1).float().cpu().numpy()
            if mu.dim() == 4
            else mu.float().cpu().numpy()
        )
        samp_list.append(latent.float().cpu().numpy())
        tok_list.append(num_points)
        print(
            f"  {sl.start:>5}-{sl.stop:<5} latent tokens {num_points}  "
            f"mu {tuple(mu_list[-1].shape)}  sample {tuple(samp_list[-1].shape)}"
        )

    print(f"latent tokens per batch: min {min(tok_list)}, max {max(tok_list)}")

    def stack(chunks):
        """
        Concatenate the per-batch latents. The latent length is
        min(batch-min hits, p_dim), so it is constant only when every shower has
        at least p_dim hits -- with --min-points below p_dim the batches come
        back at different lengths and a plain concatenate raises. Pad the short
        ones with zeros, which block_features() reads as padding.
        """
        width = max(c.shape[1] for c in chunks)
        return np.concatenate(
            [
                c
                if c.shape[1] == width
                else np.pad(c, ((0, 0), (0, width - c.shape[1]), (0, 0)))
                for c in chunks
            ]
        )

    z_mu = stack(mu_list)
    z_samp = stack(samp_list)

    blocks = {name: block_features(z) for name, z in (("mu", z_mu), ("sample", z_samp))}
    npz = os.path.join(args.out, "tsne_latents.npz")
    np.savez(
        npz,
        blocks_mu=blocks["mu"],
        blocks_sample=blocks["sample"],
        energy_true=e_true.astype(np.float32),
        energy_grid=e_grid.astype(np.float32),
        n_points=n_pts.astype(np.int32),
    )
    print(f"saved: {npz}")


# ---------------------------------------------------------------- lda-ds-grid
_DOC_LDA_DS_GRID = """
latent_analysis.py lda-ds-grid

The training-space LDA side by side for several latent sizes: one figure per label
(incident energy, points per shower), rows = encoded latent (the flow's target) and
mean-flow sample, one column per ds. Each column has its own LDA, fitted on half the
encoded showers and applied unchanged to both panels, LDA 1 oriented to increase with
the label.

    python scripts/evaluation/latent_analysis.py lda-ds-grid \
        --run "ds 64=gen_utils_plot/lda_grid_ds64_10to90_500each" \
        --run "ds 256=gen_utils_plot/lda_grid_ds256_10to90_500each" \
        --run "ds 1024=gen_utils_plot/lda_grid_ds1024_10to90_500each" \
        --out gen_utils_plot/lda_ds_comparison
"""


LABELS = {
    "energy": ("incident energy [GeV]", "lda_energy_ds.pdf"),
    "npoints": ("points per shower", "lda_npoints_ds.pdf"),
}


def main_lda_ds_grid():
    plt.rcParams.update(_LDA_RC)
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run",
        action="append",
        required=True,
        help='"label=grid dir holding tsne_latents.npz", in column order',
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shrinkage", default="auto")
    ap.add_argument("--feature", default="blocks", choices=["blocks", "middle"])
    ap.add_argument("--n-bins", type=int, default=9)
    ap.add_argument("--ticks", action="store_true", help="keep the LDA axis ticks")
    ap.add_argument(
        "--common-limits",
        action="store_true",
        help="one x/y range for every panel of both figures (the 0.5-99.5%% "
        "range of all projected points; the few beyond it are off-scale)",
    )
    ap.add_argument(
        "--limit-pct",
        type=float,
        default=0.5,
        help="--common-limits percentile: range [p, 100-p] of all points",
    )
    ap.add_argument(
        "--vertical",
        action="store_true",
        help="one row per ds, columns = encoded / sampled (one-column paper figure)",
    )
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    runs = []
    for spec in args.run:
        name, folder = spec.split("=", 1)
        d = np.load(os.path.join(folder, "tsne_latents.npz"))
        log(f"== {name}: {folder}")
        pairs, e_enc, _, groups, e_true, paired = load_pairs(d, args.feature, log)
        if not paired:
            raise SystemExit(f"{folder}: needs the balanced-grid npz")
        _, X_enc, X_samp, lbl_enc, lbl_samp = pairs[0]
        runs.append(
            dict(
                name=name,
                X_enc=np.asarray(X_enc),
                X_samp=np.asarray(X_samp),
                e_enc=e_enc,
                groups=groups,
                e_true=e_true,
                n_pts=np.asarray(d["n_points"]),
                lbl_enc=lbl_enc,
                lbl_samp=lbl_samp,
            )
        )
    # the columns are only comparable if every ds saw the same showers
    same = all(
        np.array_equal(r["e_true"], runs[0]["e_true"])
        and np.array_equal(r["n_pts"], runs[0]["n_pts"])
        for r in runs
    )
    log(f"same showers in every column: {same}")

    # every projection first, so that --common-limits can see all panels of both figures
    proj = {}
    for label in LABELS:
        for c, r in enumerate(runs):
            y, levels, _, target = build_labels(
                label, r["e_true"], r["e_enc"], r["n_pts"], args.n_bins, log
            )
            proj[label, c] = (levels, target) + row_projection(
                r["X_enc"],
                r["X_samp"],
                y,
                r["groups"],
                args.seed,
                args.shrinkage,
                target,
            )
    limits = None
    if args.common_limits:
        pts = np.concatenate(
            [np.concatenate([v[2][:, :2], v[3][:, :2]]) for v in proj.values()]
        )
        lo, hi = (
            np.percentile(pts, args.limit_pct, axis=0),
            np.percentile(pts, 100 - args.limit_pct, axis=0),
        )
        pad = 0.04 * (hi - lo)
        limits = (lo - pad, hi + pad)
        off = ((pts < limits[0]) | (pts > limits[1])).any(axis=1).mean()
        log(
            f"common limits: LDA 1 [{limits[0][0]:.1f}, {limits[1][0]:.1f}], LDA 2 "
            f"[{limits[0][1]:.1f}, {limits[1][1]:.1f}], {100 * off:.2f}% of points off-scale"
        )

    for label, (cbar_label, fname) in LABELS.items():
        ncol = len(runs)
        if args.vertical:  # one row per ds, one column per latent kind
            fig, axes = plt.subplots(ncol, 2, figsize=(5.6, 2.45 * ncol))
            fig.subplots_adjust(hspace=0.12, wspace=0.08)
            at = lambda row, c: axes[c, row]
        else:
            fig, axes = plt.subplots(2, ncol, figsize=(2.9 * ncol, 5.0))
            fig.subplots_adjust(hspace=0.12, wspace=0.28 if args.ticks else 0.08)
            at = lambda row, c: axes[row, c]
        for c, r in enumerate(runs):
            levels, target, P_enc, P_samp, y_held, acc, held, sign = proj[label, c]
            log(
                f"{label:<8} {r['name']:<8} enc {acc[0]:.3f}  samp {acc[1]:.3f}  "
                f"(chance {1 / len(levels):.3f})"
            )
            colours = plt.get_cmap("plasma")(np.linspace(0.05, 0.92, len(levels)))
            at(1, c).sharex(at(0, c))
            at(1, c).sharey(at(0, c))
            for row, (P, a) in enumerate([(P_enc, acc[0]), (P_samp, acc[1])]):
                ax = at(row, c)
                for b in range(len(levels)):
                    sel = y_held == b
                    if sel.any():
                        ax.scatter(
                            P[sel, 0],
                            P[sel, 1],
                            s=2.2,
                            alpha=0.45,
                            color=colours[b],
                            linewidths=0,
                        )
                ax.text(
                    0.03,
                    0.96,
                    f"{len(levels)}-class accuracy {a:.3f}",
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=9,
                )
                if limits is not None:
                    ax.set_xlim(limits[0][0], limits[1][0])
                    ax.set_ylim(limits[0][1], limits[1][1])
                if args.ticks:
                    ax.tick_params(labelsize=8, length=2.5, width=0.6)
                else:
                    ax.set_xticks([])
                    ax.set_yticks([])
            ds_label = r"$d_s = %s$" % r["name"].split()[-1]
            if args.vertical:
                at(0, c).set_ylabel(f"{ds_label}\nLDA 2")
                at(1, c).tick_params(
                    labelleft=False
                )  # same y range as its left neighbour
            else:
                at(0, c).set_title(ds_label, fontsize=12)
                at(1, c).set_xlabel("LDA 1")
            tgt = target[held]
        if args.vertical:
            at(0, 0).set_title(runs[0]["lbl_enc"], fontsize=12)
            at(1, 0).set_title(runs[0]["lbl_samp"], fontsize=12)
            at(0, ncol - 1).set_xlabel("LDA 1")
            at(1, ncol - 1).set_xlabel("LDA 1")
        else:
            at(0, 0).set_ylabel(f"{runs[0]['lbl_enc']}\nLDA 2")
            at(1, 0).set_ylabel(f"{runs[0]['lbl_samp']}\nLDA 2")

        # one colourbar: the bins are the same showers in every column
        bin_colorbar(
            fig, axes.ravel().tolist(), 0.02, tgt, y_held, len(levels), cbar_label
        )

        out = os.path.join(args.out, fname)
        fig.savefig(out, bbox_inches="tight")
        fig.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        log(f"saved: {out}")
    with open(os.path.join(args.out, "lda_ds_report.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


COMMANDS = {
    "encode": main_encode,
    "lda-ds-grid": main_lda_ds_grid,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        sys.exit(
            f"usage: {sys.argv[0]} {{{','.join(COMMANDS)}}} [options]  (--help per command)"
        )
    COMMANDS[sys.argv.pop(1)]()
