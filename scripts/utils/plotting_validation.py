import warnings

import matplotlib.pyplot as plt
import numpy as np

# ── Color palette ─────────────────────────────────────────────────────────────
_C_SIM = "#bbbbbb"  # grey   — ground truth
_C_REC = "#1f77b4"  # blue   — reconstruction
_C_SAMP = "#2ca02c"  # green  — sampled generation (e.g. LMF)
_C_VAE = "#ff7f0e"  # orange — sampled directly from the VAE prior (no diffusion)


# ── helpers ──────────────────────────────────────────────────────────────────


def _log_figure(experiment, name, fig, step):
    """Log and close a figure if an experiment tracker is provided."""
    if experiment is not None:
        experiment.log_figure(name, figure=fig, step=step)
    plt.close(fig)


def _style_ax(ax, ylabel=None):
    """Apply clean paper-style formatting to one axis."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.5, alpha=0.55, color="grey")
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)


def _hist_pair(
    ax,
    real,
    fake,
    bins,
    xlabel,
    xscale="linear",
    yscale="log",
    range_=None,
    fake2=None,
    fake3=None,
    label="REC",
    label2="SAMP",
    label3="SAMP_VAE",
    **kwargs,
):
    """Overlay SIM (grey fill), REC (blue step), optional SAMP (green dashed step),
    optional third series (orange dotted step) e.g. a direct VAE-prior sample."""
    common = dict(bins=bins, range=range_)
    ax.hist(real, color=_C_SIM, alpha=0.75, label="SIM", **common, **kwargs)
    ax.hist(
        fake,
        histtype="step",
        color=_C_REC,
        linewidth=1.8,
        label=label,
        **common,
        **kwargs,
    )
    if fake2 is not None:
        ax.hist(
            fake2,
            histtype="step",
            color=_C_SAMP,
            linewidth=1.8,
            linestyle="--",
            label=label2,
            **common,
            **kwargs,
        )
    if fake3 is not None:
        ax.hist(
            fake3,
            histtype="step",
            color=_C_VAE,
            linewidth=1.8,
            linestyle=":",
            label=label3,
            **common,
            **kwargs,
        )
    ax.set_xlabel(xlabel, fontsize=10)
    ax.legend(fontsize=8, framealpha=0.7)
    ax.set_xscale(xscale)
    _style_ax(ax, ylabel="counts")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        try:
            ax.set_yscale(yscale)
        except ValueError:
            pass


def _mean_energy_hist(
    ax,
    real_pos,
    real_e,
    fake_pos,
    fake_e,
    bins,
    range_=None,
    xlabel="",
    xscale="linear",
    fake2_pos=None,
    fake2_e=None,
    fake3_pos=None,
    fake3_e=None,
    label="REC",
    label2="SAMP",
    label3="SAMP_VAE",
):
    """Mean hit energy per bin: SIM (grey fill), REC (blue), optional SAMP (green dashed),
    optional third series (orange dotted) e.g. a direct VAE-prior sample."""
    edges = (
        np.linspace(range_[0], range_[1], bins + 1)
        if isinstance(bins, int)
        else np.asarray(bins)
    )

    def _bin_mean(pos, e):
        pos, e = np.asarray(pos).flatten(), np.asarray(e).flatten()
        cnt, _ = np.histogram(pos, bins=edges)
        total, _ = np.histogram(pos, bins=edges, weights=e)
        result = np.zeros_like(total)
        np.divide(total, cnt, out=result, where=cnt > 0)
        return result

    ax.stairs(
        _bin_mean(real_pos, real_e),
        edges,
        fill=True,
        color=_C_SIM,
        alpha=0.75,
        label="SIM",
    )
    ax.stairs(
        _bin_mean(fake_pos, fake_e), edges, color=_C_REC, linewidth=1.8, label=label
    )
    if fake2_pos is not None:
        ax.stairs(
            _bin_mean(fake2_pos, fake2_e),
            edges,
            color=_C_SAMP,
            linewidth=1.8,
            linestyle="--",
            label=label2,
        )
    if fake3_pos is not None:
        ax.stairs(
            _bin_mean(fake3_pos, fake3_e),
            edges,
            color=_C_VAE,
            linewidth=1.8,
            linestyle=":",
            label=label3,
        )
    ax.set_xlabel(xlabel, fontsize=10)
    ax.legend(fontsize=8, framealpha=0.7)
    ax.set_xscale(xscale)
    _style_ax(ax, ylabel="mean energy")
    try:
        ax.set_yscale("log")
    except ValueError:
        pass


# ── public API ────────────────────────────────────────────────────────────────


def plot_during_train(
    x2,
    fake_shower,
    samp_shower=None,
    samp_shower_vae=None,
    others=None,
    plot_name="",
    experiment=None,
    it=0,
):
    """Diagnostic plots comparing SIM (x2), REC (fake_shower), optionally SAMP
    (samp_shower), and optionally a direct VAE-prior sample (samp_shower_vae) —
    useful to compare LMF-generated latents against plain VAE-prior latents.
    """
    k = 2  # example shower index for scatter plots
    has_samp = samp_shower is not None
    has_rec = fake_shower is not None
    has_samp_vae = samp_shower_vae is not None

    def _scatter(ax, data, title, color):
        ax.set_title(title, fontsize=10, fontweight="bold", pad=5)
        ax.set_ylim(-1, 78)
        ax.set_xlim(-1.1, 1.1)
        ax.set_xlabel("X", fontsize=9)
        ax.set_ylabel("Layer", fontsize=9)
        _style_ax(ax)
        mask = data[k, :, 3] > 0
        if not mask.any():
            return
        x = data[k, :, 0][mask]
        y = data[k, :, 1][mask]
        e = data[k, :, 3][mask]
        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(e)
        if not valid.any():
            return
        x, y, e = x[valid], y[valid], e[valid]
        emax = float(e.max())
        s = np.clip(e / emax * 50, 0.5, 50) if emax > 0 else np.full(len(e), 5.0)
        ax.scatter(x, y, s=s, c=color, alpha=0.55, linewidths=0)

    # ── Figure 1: scatter + energy histograms ────────────────────────────────
    n_scatter = 1 + int(has_rec) + int(has_samp) + int(has_samp_vae)
    fig1, axes = plt.subplots(1, n_scatter + 2, figsize=(5.5 * (n_scatter + 2), 5.5))
    scatter_axes = axes[:n_scatter]
    ax_ehist = axes[n_scatter]
    ax_esum = axes[n_scatter + 1]

    _scatter(scatter_axes[0], x2, "SIM", _C_SIM)
    si = 1
    if has_rec:
        _scatter(scatter_axes[si], fake_shower, "REC", _C_REC)
        si += 1
    if has_samp:
        _scatter(scatter_axes[si], samp_shower, "SAMP", _C_SAMP)
        si += 1
    if has_samp_vae:
        _scatter(scatter_axes[si], samp_shower_vae, "SAMP_VAE", _C_VAE)

    energy_bins = np.logspace(np.log10(8e-5), np.log10(4e2), 60)
    _hist_pair(
        ax_ehist,
        x2[:, :, 3][x2[:, :, 3] > 0].flatten(),
        fake_shower[:, :, 3][fake_shower[:, :, 3] > 0].flatten()
        if has_rec
        else np.array([]),
        bins=energy_bins,
        xlabel="visible energy",
        xscale="log",
        fake2=samp_shower[:, :, 3][samp_shower[:, :, 3] > 0].flatten()
        if has_samp
        else None,
        fake3=samp_shower_vae[:, :, 3][samp_shower_vae[:, :, 3] > 0].flatten()
        if has_samp_vae
        else None,
    )
    _hist_pair(
        ax_esum,
        x2[:, :, 3].sum(axis=1).flatten(),
        fake_shower[:, :, 3].sum(axis=1).flatten() if has_rec else np.array([]),
        bins=10,
        range_=[0, 2000],
        xlabel="energy sum",
        fake2=samp_shower[:, :, 3].sum(axis=1).flatten() if has_samp else None,
        fake3=samp_shower_vae[:, :, 3].sum(axis=1).flatten() if has_samp_vae else None,
    )

    fig1.suptitle(
        f"Shower diagnostics  —  {plot_name}", fontsize=11, fontweight="bold", y=1.01
    )
    fig1.tight_layout()
    _log_figure(experiment, plot_name, fig1, it)

    # ── Figure 2: X / last-feature / Z distributions ─────────────────────────
    fig2, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19, 5.5))
    xy_range = [-1, 1]
    bins_xyz = 33
    mask_r = x2[:, :, 3] > 0
    mask_f = fake_shower[:, :, 3] > 0 if has_rec else np.zeros_like(mask_r)
    mask_s = samp_shower[:, :, 3] > 0 if has_samp else None
    mask_sv = samp_shower_vae[:, :, 3] > 0 if has_samp_vae else None

    for ax, feat_idx, xlabel in [(ax1, 0, "X"), (ax3, 2, "Z")]:
        _mean_energy_hist(
            ax,
            x2[:, :, feat_idx][mask_r].flatten(),
            x2[:, :, 3][mask_r].flatten(),
            fake_shower[:, :, feat_idx][mask_f].flatten() if has_rec else np.array([]),
            fake_shower[:, :, 3][mask_f].flatten() if has_rec else np.array([]),
            bins=bins_xyz,
            range_=xy_range,
            xlabel=xlabel,
            fake2_pos=samp_shower[:, :, feat_idx][mask_s].flatten()
            if has_samp
            else None,
            fake2_e=samp_shower[:, :, 3][mask_s].flatten() if has_samp else None,
            fake3_pos=samp_shower_vae[:, :, feat_idx][mask_sv].flatten()
            if has_samp_vae
            else None,
            fake3_e=samp_shower_vae[:, :, 3][mask_sv].flatten()
            if has_samp_vae
            else None,
        )

    has_time = x2.shape[-1] > 4
    last_feat_bins = np.logspace(np.log10(8e-6), np.log10(350), 70)
    if not has_time:
        # no time column: the last feature is the energy, already shown above
        ax2.set_visible(False)
    else:
        _mean_energy_hist(
            ax2,
            x2[:, :, -1][mask_r].flatten(),
            x2[:, :, 3][mask_r].flatten(),
            fake_shower[:, :, -1][mask_f].flatten() if has_rec else np.array([]),
            fake_shower[:, :, 3][mask_f].flatten() if has_rec else np.array([]),
            bins=last_feat_bins,
            xlabel="Last feature",
            xscale="log",
            fake2_pos=samp_shower[:, :, -1][mask_s].flatten() if has_samp else None,
            fake2_e=samp_shower[:, :, 3][mask_s].flatten() if has_samp else None,
            fake3_pos=samp_shower_vae[:, :, -1][mask_sv].flatten()
            if has_samp_vae
            else None,
            fake3_e=samp_shower_vae[:, :, 3][mask_sv].flatten()
            if has_samp_vae
            else None,
        )

    fig2.suptitle(
        f"Spatial distributions  —  {plot_name}", fontsize=11, fontweight="bold", y=1.01
    )
    fig2.tight_layout()
    _log_figure(experiment, f"XYZ {plot_name}", fig2, it)

    # ── Figure 3: ECal / HCal energy split ───────────────────────────────────
    fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    ecal_bins = np.logspace(np.log10(8e-4), np.log10(4e2), 60)

    ecal_mask_r = (x2[:, :, 3] > 0) & (x2[:, :, 1] <= 29)
    ecal_mask_f = (
        (fake_shower[:, :, 3] > 0) & (fake_shower[:, :, 1] <= 29) if has_rec else None
    )
    hcal_mask_r = (x2[:, :, 3] > 0) & (x2[:, :, 1] > 29)
    hcal_mask_f = (
        (fake_shower[:, :, 3] > 0) & (fake_shower[:, :, 1] > 29) if has_rec else None
    )

    _hist_pair(
        ax1,
        x2[:, :, 3][ecal_mask_r].flatten(),
        fake_shower[:, :, 3][ecal_mask_f].flatten() if has_rec else np.array([]),
        bins=ecal_bins,
        xlabel="visible energy ECal",
        xscale="log",
        fake2=(
            samp_shower[:, :, 3][
                (samp_shower[:, :, 3] > 0) & (samp_shower[:, :, 1] <= 29)
            ].flatten()
            if has_samp
            else None
        ),
        fake3=(
            samp_shower_vae[:, :, 3][
                (samp_shower_vae[:, :, 3] > 0) & (samp_shower_vae[:, :, 1] <= 29)
            ].flatten()
            if has_samp_vae
            else None
        ),
    )
    _hist_pair(
        ax2,
        x2[:, :, 3][hcal_mask_r].flatten(),
        fake_shower[:, :, 3][hcal_mask_f].flatten() if has_rec else np.array([]),
        bins=ecal_bins,
        xlabel="visible energy HCal",
        xscale="log",
        fake2=(
            samp_shower[:, :, 3][
                (samp_shower[:, :, 3] > 0) & (samp_shower[:, :, 1] > 29)
            ].flatten()
            if has_samp
            else None
        ),
        fake3=(
            samp_shower_vae[:, :, 3][
                (samp_shower_vae[:, :, 3] > 0) & (samp_shower_vae[:, :, 1] > 29)
            ].flatten()
            if has_samp_vae
            else None
        ),
    )

    fig3.suptitle(
        f"ECal / HCal  —  {plot_name}", fontsize=11, fontweight="bold", y=1.01
    )
    fig3.tight_layout()
    _log_figure(experiment, f"Energies ECal/HCal {plot_name}", fig3, it)

    # ── Optional extras (posteriors, diffusion latents) ──────────────────────
    if others is None:
        return

    _plot_posteriors(others, plot_name, experiment, it)
    _plot_priors(others, plot_name, experiment, it)
    if others.get("priors") is not None and others.get("posteriors") is not None:
        _plot_posteriors_and_priors(others, plot_name, experiment, it)
    _plot_diffusion_latents(others, plot_name, experiment, it)


def plot_energy_per_layer(
    x2, fake_showers, samp_shower=None, samp_shower_vae=None, experiment=None, it=0
):
    """Plot mean deposited energy per calorimeter layer for SIM, REC, optionally
    SAMP, and optionally a direct VAE-prior sample (samp_shower_vae)."""
    layers = np.arange(0, 78)
    n_real = x2.shape[0]
    n_fake = fake_showers.shape[0] if fake_showers is not None else 0
    n_samp = samp_shower.shape[0] if samp_shower is not None else 0
    n_samp_vae = samp_shower_vae.shape[0] if samp_shower_vae is not None else 0

    def _energy_per_layer(data, n_events):
        if data is None or n_events == 0:
            return np.zeros(len(layers))
        result = []
        for layer in layers:
            mask = (
                (data[:, :, 1] >= layer)
                & (data[:, :, 1] < layer + 1)
                & (data[:, :, 3] > 0)
            )
            result.append(data[:, :, 3][mask].sum())
        return np.array(result) / n_events

    energy_real = _energy_per_layer(x2, n_real)
    energy_fake = _energy_per_layer(fake_showers, n_fake)
    energy_samp = _energy_per_layer(samp_shower, n_samp)
    energy_samp_vae = _energy_per_layer(samp_shower_vae, n_samp_vae)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.step(layers, energy_real, linewidth=2.5, alpha=0.75, label="SIM", color=_C_SIM)
    if n_fake > 0:
        ax.step(layers, energy_fake, linewidth=2.0, label="REC", color=_C_REC)
    if n_samp > 0:
        ax.step(
            layers,
            energy_samp,
            linewidth=2.0,
            linestyle="--",
            label="SAMP",
            color=_C_SAMP,
        )
    if n_samp_vae > 0:
        ax.step(
            layers,
            energy_samp_vae,
            linewidth=2.0,
            linestyle=":",
            label="SAMP_VAE",
            color=_C_VAE,
        )
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Mean energy", fontsize=11)
    ax.set_yscale("log")
    ax.legend(fontsize=9, framealpha=0.7)
    _style_ax(ax)
    ax.xaxis.grid(True, linestyle=":", linewidth=0.5, alpha=0.55, color="grey")
    fig.tight_layout()
    _log_figure(experiment, "Energy per Layer", fig, it)


def plot_DS_xye(x2_in, fs_in, downsampled: bool, experiment=None, it=0):
    """Compare coordinate and hit-count distributions."""
    en_idx = 2
    xy_range = [-1, 1]
    title = "Downsampled points" if downsampled else "NOT Downsampled points"
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(28, 6))
    for ax, feat_idx, xlabel in [(ax1, 0, "X"), (ax3, 1, "Y")]:
        _hist_pair(
            ax,
            x2_in[:, :, feat_idx][x2_in[:, :, en_idx] > 0].flatten(),
            fs_in[:, :, feat_idx][fs_in[:, :, en_idx] > 0].flatten(),
            bins=31,
            range_=xy_range,
            xlabel=xlabel,
        )
    if x2_in.shape[-1] > 3:
        _hist_pair(
            ax2,
            x2_in[:, :, -1][x2_in[:, :, en_idx] > 0],
            fs_in[:, :, -1][fs_in[:, :, en_idx] > 0],
            bins=np.logspace(np.log10(8e-4), np.log10(350), 70),
            xlabel="Last feature",
            xscale="log",
        )
    else:
        # no time column: the last feature is the energy, not a separate observable
        ax2.set_visible(False)
    n_hits_real = (x2_in[:, :, en_idx] > 0).sum(axis=1).flatten()
    n_hits_fake = (fs_in[:, :, en_idx] > 0).sum(axis=1).flatten()
    hits_max = float(np.percentile(n_hits_real, 99)) if len(n_hits_real) > 0 else 1.0
    _hist_pair(
        ax4,
        n_hits_real,
        n_hits_fake,
        bins=30,
        range_=[0, max(hits_max * 1.2, 1)],
        xlabel="# hits",
    )
    fig.suptitle(title, fontsize=11, fontweight="bold", y=1.01)
    fig.tight_layout()
    _log_figure(experiment, title, fig, it)


# ── private helpers for plot_during_train ────────────────────────────────────


def _plot_posteriors(others, plot_name, experiment, it, n_dims_to_plot=6):
    if others.get("posteriors") is None:
        return
    # (B, p_dim, [mixture,] z_dim, 3) where the last axis is [eps, mu, logvar]
    last_layer = others["posteriors"][:, -1]

    # mask_DS marks padded downsampled points (1/True = padded) — drop them so
    # histograms aren't dominated by zeroed-out padding.
    mask_ds = others.get("mask_DS")
    valid = None
    if mask_ds is not None and mask_ds.shape[:2] == last_layer.shape[:2]:
        valid = ~mask_ds.bool()

    mixture_colors = [
        "green",
        "blue",
        "red",
        "purple",
        "orange",
        "cyan",
        "gray",
        "black",
        "magenta",
    ]
    labels = ["eps", "mu", "logvar"]
    n_mixtures = int(others.get("gaussian_mixture", 1))
    z_dim = last_layer.shape[-2]  # z_dim is always the second-to-last axis
    dim_idx = sorted(
        set(
            np.linspace(0, z_dim - 1, num=min(n_dims_to_plot, z_dim))
            .round()
            .astype(int)
        )
    )

    fig, axes = plt.subplots(
        len(dim_idx), 3, figsize=(15, 4 * len(dim_idx)), squeeze=False
    )
    for row, dim in enumerate(dim_idx):
        for col, label in enumerate(labels):
            ax = axes[row, col]
            if n_mixtures > 1:
                # last_layer: (B, p_dim, mixture, z_dim, 3)
                x = last_layer[:, :, :, dim, col]  # B, p_dim, mixture
                for j in range(n_mixtures):
                    vals = x[:, :, j]
                    if valid is not None:
                        vals = vals[valid]
                    ax.hist(
                        vals.flatten().cpu().numpy(),
                        bins=100,
                        range=[-5, 5],
                        alpha=0.7,
                        color=mixture_colors[j],
                        label=f"mix {j}",
                    )
                ax.legend()
            else:
                # last_layer: (B, p_dim, z_dim, 3)
                vals = last_layer[:, :, dim, col]  # B, p_dim
                if valid is not None:
                    vals = vals[valid]
                ax.hist(
                    vals.flatten().cpu().numpy(),
                    bins=100,
                    range=[-5, 5],
                )
            ax.set_xlabel(f"{label} | z-dim {dim}")
            ax.set_ylabel("counts")
            ax.set_yscale("log")
            ax.grid(True)
            if row == 0:
                ax.set_title(label, fontsize=14)
    fig.tight_layout()
    _log_figure(experiment, f"Posterior {plot_name}", fig, it)


def _plot_posteriors_and_priors(others, plot_name, experiment, it):
    if others.get("posteriors") is None or others.get("priors") is None:
        return
    post = others["posteriors"][:, -1]
    prior = others["priors"][:, -1]
    n_mixtures = int(others.get("gaussian_mixture", 1))
    n_dims = post.shape[-2] if n_mixtures == 1 else post.shape[-3]

    post_labels = ["mu", "logvar"]
    fig, axes = plt.subplots(n_dims, 2, figsize=(20, 5 * n_dims))
    if n_dims == 1:
        axes = axes[np.newaxis, :]

    for dim in range(n_dims):
        for col, label in enumerate(post_labels):
            ax = axes[dim, col]
            post_idx = col + 1
            if n_mixtures > 1:
                for j in range(n_mixtures):
                    ax.hist(
                        post[:, dim, j, post_idx].flatten().cpu().numpy(),
                        bins=100,
                        range=[-5, 5],
                        alpha=0.5,
                        label=f"post mix {j}",
                    )
                    ax.hist(
                        prior[:, dim, j, col].flatten().cpu().numpy(),
                        bins=100,
                        range=[-5, 5],
                        alpha=0.5,
                        linestyle="dashed",
                        label=f"prior mix {j}",
                    )
            else:
                ax.hist(
                    post[:, dim, post_idx].flatten().cpu().numpy(),
                    bins=100,
                    range=[-5, 5],
                    alpha=0.6,
                    label="posterior",
                    color="blue",
                )
                ax.hist(
                    prior[:, dim, col].flatten().cpu().numpy(),
                    bins=100,
                    range=[-5, 5],
                    alpha=0.6,
                    label="prior",
                    color="orange",
                )
            ax.set_xlabel(f"{label} | dim {dim}")
            ax.set_ylabel("counts")
            ax.set_yscale("log")
            ax.legend()
            ax.grid(True)
            if dim == 0:
                ax.set_title(label, fontsize=14)
    fig.tight_layout()
    _log_figure(experiment, f"Posterior vs Prior {plot_name}", fig, it)


def _plot_priors(others, plot_name, experiment, it):
    if others.get("priors") is None:
        return
    last_layer = others["priors"][:, -1]
    mixture_colors = [
        "green",
        "blue",
        "red",
        "purple",
        "orange",
        "cyan",
        "gray",
        "black",
        "magenta",
    ]
    labels = ["mu", "logvar"]
    n_mixtures = int(others.get("gaussian_mixture", 1))
    n_dims = last_layer.shape[-2] if n_mixtures == 1 else last_layer.shape[-3]

    fig, axes = plt.subplots(n_dims, 2, figsize=(20, 5 * n_dims))
    if n_dims == 1:
        axes = axes[np.newaxis, :]

    for dim in range(n_dims):
        for col, label in enumerate(labels):
            ax = axes[dim, col]
            if n_mixtures > 1:
                x = last_layer[:, dim, :, col]
                for j in range(n_mixtures):
                    ax.hist(
                        x[:, j].flatten().cpu().numpy(),
                        bins=100,
                        range=[-5, 5],
                        alpha=0.7,
                        color=mixture_colors[j],
                        label=f"mix {j}",
                    )
                ax.legend()
            else:
                ax.hist(
                    last_layer[:, dim, col].flatten().cpu().numpy(),
                    bins=100,
                    range=[-5, 5],
                )
            ax.set_xlabel(f"{label} | dim {dim}")
            ax.set_ylabel("counts")
            ax.set_yscale("log")
            ax.grid(True)
            if dim == 0:
                ax.set_title(label, fontsize=14)
    fig.tight_layout()
    _log_figure(experiment, f"Prior {plot_name}", fig, it)


def _plot_diffusion_latents(others, plot_name, experiment, it):
    if others.get("lmf_vae") is None:
        return

    B, N, F = others["lmf_vae"].shape
    z_vae = others["lmf_vae"].reshape(-1, F)
    z_diff = others["lmf_sample"].reshape(-1, F)
    # Drop the cross-batch zero padding (training.py pads every batch's latent
    # to the longest effective_p_dim); it would add a spike at 0 to both
    # histograms in every dimension.
    mask = others.get("lmf_mask")
    if mask is not None:
        mask = np.asarray(mask).reshape(-1)
        z_vae = z_vae[mask]
        z_diff = z_diff[mask]

    ncols = int(np.ceil(np.sqrt(F)))
    nrows = int(np.ceil(F / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(35, 20))
    for i, ax in enumerate(axes.flatten()):
        if i >= F:
            ax.set_visible(False)
            continue
        combined = np.concatenate([z_vae[:, i], z_diff[:, i]])
        finite = combined[np.isfinite(combined)]
        if len(finite) == 0:
            ax.set_xlabel(f"latent {i}")
            continue
        lo, hi = float(np.percentile(finite, 1)), float(np.percentile(finite, 99))
        if lo == hi:
            lo, hi = lo - 1, hi + 1
        range_ = [lo, hi]
        ax.hist(z_vae[:, i].flatten(), bins=31, range=range_, color="lightgray")
        ax.hist(
            z_diff[:, i].flatten(),
            bins=31,
            range=range_,
            histtype="step",
            linewidth=3,
            color="green",
        )
        try:
            ax.set_yscale("log")
        except ValueError:
            pass
        ax.set_xlabel(f"latent {i}")

    fig.tight_layout()
    _log_figure(experiment, f"Latent space of Diffusion {plot_name}", fig, it)


# ── t-SNE latent space plot (VAE and LMF) ──────────────────────


def plot_tsne(others, g4, net, experiment, it):
    """Run shower-level t-SNE on the VAE/LMF latent space."""
    try:
        from utils.plotting_t_SNE import plot_vae_tsne_conditioning
    except ImportError:
        from plotting_t_SNE import plot_vae_tsne_conditioning

    if net == "LMF":
        z_enc_t = others.get("lmf_vae")
        z_samp_t = others.get("lmf_sample")
        cond_e_t = others.get("cond_e")
        if z_enc_t is None or z_samp_t is None or cond_e_t is None:
            return
        z_enc = (
            z_enc_t.detach().cpu().numpy()
            if hasattr(z_enc_t, "detach")
            else np.asarray(z_enc_t)
        )
        z_samp = (
            z_samp_t.detach().cpu().numpy()
            if hasattr(z_samp_t, "detach")
            else np.asarray(z_samp_t)
        )
        cond_e = (
            cond_e_t.detach().cpu().numpy()
            if hasattr(cond_e_t, "detach")
            else np.asarray(cond_e_t)
        )

    elif net == "VAE":
        latent_t = others.get("latent")
        samp_latent_t = others.get("sampled_latent")
        cond_feats_t = others.get("cond_feats")
        if latent_t is None or samp_latent_t is None or cond_feats_t is None:
            return
        latent = latent_t[:, -1]  # last decoder layer: [B, p_dim, z_dim, 1]
        samp_latent = samp_latent_t[:, -1]  # same shape
        B, N, F, D = latent.shape
        z_enc = latent.reshape(B, N, F * D).detach().cpu().numpy()
        z_samp = samp_latent.reshape(B, N, F * D).detach().cpu().numpy()
        cf = (
            cond_feats_t.detach().cpu().numpy()
            if hasattr(cond_feats_t, "detach")
            else np.asarray(cond_feats_t)
        )
        cond_e = cf[:, 0]

    else:
        return

    if not (np.isfinite(z_enc).all() and np.isfinite(z_samp).all()):
        print("[t-SNE] Skipping: non-finite values in latent vectors.")
        return

    B = z_enc.shape[0]
    if B < 10:
        print(f"[t-SNE] Skipping: only {B} samples (need ≥ 10).")
        return

    n_pts = (g4[:, :, 3] > 0).sum(axis=1).astype(float)
    layers = ((g4[:, :, 1] + 1) / 2 * 78).clip(0, 77).astype(int)
    perp = max(5, min(30, int(np.sqrt(B)) - 1))

    old_rc = {
        k: plt.rcParams[k] for k in ("font.family", "font.size") if k in plt.rcParams
    }
    try:
        plot_vae_tsne_conditioning(
            z_enc,
            z_samp,
            incident_energy=np.asarray(cond_e, dtype=float),
            num_points=n_pts,
            layer_index=layers,
            flag_shower_level=True,
            flag_point_level=False,
            perplexity=perp,
            energy_log_scale=True,
            points_log_scale=True,
            layer_log_scale=True,
            experiment=experiment,
            step=it,
        )
    finally:
        plt.rcParams.update(old_rc)
