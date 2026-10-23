"""
plot_vae_tsne_conditioning.py

Publication-ready t-SNE plots for a conditional VAE, coloured by three
conditioning variables: incident energy, number of points, and layer index.

One figure per conditioning variable → three figures total, each showing
z_encoded (left) and z_sampled (right) side-by-side.
"""

import h5py
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colorbar import ColorbarBase
from sklearn.manifold import TSNE

# ── Colourmaps ────────────────────────────────────────────────────────────────
# Continuous variables  → perceptually uniform sequential cmaps
# Discrete variable     → qualitative cmap (tab10 / Set1)
CMAP_ENERGY = "plasma"  # incident energy  (continuous)
CMAP_NPOINTS = "viridis"  # number of points (continuous)
CMAP_LAYER = "tab10"  # layer index       (discrete)

# ─── Axis styling ─────────────────────────────────────────────────────────────


def _style_ax(ax: plt.Axes, title: str) -> None:
    """Apply minimal paper-style formatting to one scatter axis."""
    ax.set_title(title, fontsize=10, pad=6)
    ax.set_xlabel("t-SNE 1", fontsize=8)
    ax.set_ylabel("t-SNE 2", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)


# ─── Colorbar helpers ─────────────────────────────────────────────────────────


def _add_continuous_colorbar(ax, cmap, vmin, vmax, label, log_scale=False):
    if log_scale:
        vmin = max(vmin, 1e-10)
        vmax = max(vmax, vmin * 10)
    else:
        if vmin == vmax:
            vmin -= 1e-10
            vmax += 1e-10
    norm = (
        mcolors.LogNorm(vmin=vmin, vmax=vmax)
        if log_scale
        else mcolors.Normalize(vmin=vmin, vmax=vmax)
    )
    cax = ax.inset_axes([1.03, 0.0, 0.04, 1.0])
    cb = ColorbarBase(cax, cmap=cmap, norm=norm, orientation="vertical")
    cb.set_label(label, fontsize=8)
    cb.ax.tick_params(labelsize=7)


def _add_discrete_colorbar(ax, unique_vals, cmap_name, label):
    n = len(unique_vals)
    cmap = cm.get_cmap(cmap_name, n)
    norm = mcolors.BoundaryNorm(np.arange(-0.5, n + 0.5, 1), ncolors=n)
    cax = ax.inset_axes([1.03, 0.0, 0.04, 1.0])
    cb = ColorbarBase(
        cax, cmap=cmap, norm=norm, orientation="vertical", ticks=np.arange(n)
    )
    cb.set_ticklabels([str(int(v)) for v in unique_vals])
    cb.set_label(label, fontsize=8)
    cb.ax.tick_params(labelsize=7)


# ─── t-SNE helpers ────────────────────────────────────────────────────────────


def _run_tsne(
    data: np.ndarray, perplexity: int, n_iter: int, random_state: int
) -> np.ndarray:
    """Fit t-SNE on *data* [M, d] and return embedding [M, 2]."""
    return TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=n_iter,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
    ).fit_transform(data)


def _tsne_per_shower(z_enc, z_samp, perplexity, n_iter, random_state):
    """
    Mean-pool [B, N, d] → [B, d], concatenate enc+samp, run one t-SNE.
    Returns tsne_enc [B, 2] and tsne_samp [B, 2].
    """
    B = z_enc.shape[0]
    enc_pool = z_enc.mean(axis=1)  # [B, d]
    samp_pool = z_samp.mean(axis=1)  # [B, d]
    combined = np.concatenate([enc_pool, samp_pool], axis=0)  # [2B, d]
    emb = _run_tsne(combined, perplexity, n_iter, random_state)
    return emb[:B], emb[B:]


def _tsne_per_point(z_enc, z_samp, perplexity, n_iter, random_state):
    """
    Flatten [B, N, d] → [B*N, d], concatenate enc+samp, run one t-SNE.
    Returns tsne_enc [B*N, 2] and tsne_samp [B*N, 2].
    """
    B, N, d = z_enc.shape
    enc_flat = z_enc.reshape(B * N, d)
    samp_flat = z_samp.reshape(B * N, d)
    combined = np.concatenate([enc_flat, samp_flat], axis=0)  # [2BN, d]
    emb = _run_tsne(combined, perplexity, n_iter, random_state)
    return emb[: B * N], emb[B * N :]


# ─── Main Plotting script  ────────────────────────────────────────────────────────────


def plot_vae_tsne_conditioning(
    z_encoded: np.ndarray,
    z_sampled: np.ndarray,
    incident_energy: np.ndarray,
    num_points: np.ndarray,
    layer_index: np.ndarray,
    incident_energy_samp: np.ndarray = None,
    num_points_samp: np.ndarray = None,
    layer_index_samp: np.ndarray = None,
    flag_shower_level: bool = True,
    flag_point_level: bool = True,
    perplexity: int = 30,
    n_iter: int = 1000,
    random_state: int = 42,
    figsize: tuple = (12, 4.5),
    alpha: float = 0.55,
    point_size: int = 6,
    energy_log_scale: bool = True,
    points_log_scale: bool = True,
    layer_log_scale: bool = True,
    save_prefix: str = None,
    experiment=None,
    step: int = 0,
    dpi: int = 300,
) -> None:
    """
    Produce three publication-ready figures, one per conditioning variable.

    Each figure has two panels:
        left  — z_encoded  (encoder mean μ)
        right — z_sampled  (reparameterised sample)

    Args:
        z_encoded:        [B, N, d]  encoder mean.
        z_sampled:        [B, N, d]  sampled latent code.
        incident_energy:  [B]        continuous — e.g. GeV values.
        num_points:       [B]        continuous/integer — number of hits/points.
        layer_index:      [B]        discrete integer — calorimeter layer id.
        incident_energy_samp / num_points_samp / layer_index_samp:
                          the same three labels for the *sampled* latents.
                          z_sampled[i] is generated from its own conditioning, which
                          describes the same shower as z_encoded[i] only when the
                          caller kept the two index-aligned. Pass these whenever it
                          did not -- e.g. gen_utils sorts the real showers by their
                          own point count and the generated ones by the generated
                          hit count, so index i is a different shower on each side,
                          and colouring the right panel with the encoded labels
                          scrambles it. Default None -> reuse the encoded labels.
        perplexity:       t-SNE perplexity.
        n_iter:           t-SNE iterations.
        random_state:     Reproducibility seed.
        figsize:          (width, height) per figure in inches.
        alpha:            Point transparency.
        point_size:       Scatter marker size.
        energy_log_scale: Use log normalisation for the energy colorbar.
        points_log_scale:  Use log normalisation for the num_points colorbar.
        layer_log_scale:   Use log normalisation for the layer_index colorbar.
        save_prefix:      If given, saves as "{save_prefix}_energy.pdf" etc.
        dpi:              Resolution for raster saves.

    Returns:
        List of (fig, axes) tuples — one per conditioning variable.
    """
    # ── Scatter builders ──────────────────────────────────────────────────────

    def _fig_continuous(
        coords_enc,
        coords_samp,
        color_enc,
        color_samp,
        cmap,
        vmin,
        vmax,
        cb_label,
        log_scale,
        suptitle,
    ):
        fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"wspace": 0.35})
        norm = (
            mcolors.LogNorm(vmin=vmin, vmax=vmax)
            if log_scale
            else mcolors.Normalize(vmin=vmin, vmax=vmax)
        )
        for ax, coords, color, title in [
            (axes[0], coords_enc, color_enc, r"$\mathbf{z}_\mu$ (encoded)"),
            (axes[1], coords_samp, color_samp, r"$\mathbf{z}$ (sampled)"),
        ]:
            ax.scatter(
                coords[:, 0],
                coords[:, 1],
                c=color,
                cmap=cmap,
                norm=norm,
                s=point_size,
                alpha=alpha,
                linewidths=0,
                rasterized=True,
            )
            _style_ax(ax, title)
        _add_continuous_colorbar(axes[-1], cmap, vmin, vmax, cb_label, log_scale)
        fig.suptitle(suptitle, fontsize=11, y=1.02)
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig.tight_layout()
        return fig, axes

    def _fig_discrete(
        coords_enc,
        coords_samp,
        color_enc_flat,
        color_samp_flat,
        unique_vals,
        cmap_name,
        cb_label,
        suptitle,
    ):
        n = len(unique_vals)
        cmap = cm.get_cmap(cmap_name, n)
        val_to_idx = {v: i for i, v in enumerate(unique_vals)}
        norm = mcolors.BoundaryNorm(np.arange(-0.5, n + 0.5, 1), ncolors=n)
        fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"wspace": 0.35})
        for ax, coords, color_vals, title in [
            (axes[0], coords_enc, color_enc_flat, r"$\mathbf{z}_\mu$ (encoded)"),
            (axes[1], coords_samp, color_samp_flat, r"$\mathbf{z}$ (sampled)"),
        ]:
            c_idx = np.array([val_to_idx[v] for v in color_vals])
            ax.scatter(
                coords[:, 0],
                coords[:, 1],
                c=c_idx,
                cmap=cmap,
                norm=norm,
                s=point_size,
                alpha=alpha,
                linewidths=0,
                rasterized=True,
            )
            _style_ax(ax, title)
        _add_discrete_colorbar(axes[-1], unique_vals, cmap_name, cb_label)
        fig.suptitle(suptitle, fontsize=11, y=1.02)
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig.tight_layout()
        return fig, axes

    B, N, d = z_encoded.shape

    # ── Global rcParams ───────────────────────────────────────────────────────
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    if flag_shower_level:
        if not flag_point_level:
            print("Only plotting shower-level t-SNE (skipping point-level).")

        incident_energy = np.asarray(incident_energy, dtype=float)
        num_points = np.asarray(num_points, dtype=float)
        mean_layer_index = np.asarray(layer_index, dtype=int).mean(
            axis=1
        )  # [B] mean layer index per shower

        # labels for the right-hand (sampled) panel; see the docstring on why these
        # are not always the encoded ones
        energy_samp = (
            incident_energy
            if incident_energy_samp is None
            else np.asarray(incident_energy_samp, dtype=float)
        )
        npoints_samp = (
            num_points
            if num_points_samp is None
            else np.asarray(num_points_samp, dtype=float)
        )
        mean_layer_samp = (
            mean_layer_index
            if layer_index_samp is None
            else np.asarray(layer_index_samp, dtype=int).mean(axis=1)
        )

        assert incident_energy.shape == (B,), f"incident_energy must be [B={B}]"
        assert num_points.shape == (B,), f"num_points must be [B={B}]"
        assert mean_layer_index.shape == (B,), f"layer_index shape 0 must be [B={B}]"
        assert energy_samp.shape == (B,), f"incident_energy_samp must be [B={B}]"
        assert npoints_samp.shape == (B,), f"num_points_samp must be [B={B}]"
        assert mean_layer_samp.shape == (B,), f"layer_index_samp must be [B={B}]"

        # ── t-SNE run 1: per-shower (energy & num_points) ────────────────────────
        print("Running t-SNE [per shower] …")
        tsne_enc_sh, tsne_samp_sh = _tsne_per_shower(
            z_encoded, z_sampled, perplexity, n_iter, random_state
        )

        # ── Figure 1: Incident energy (per shower) ────────────────────────────────
        energy_all = np.concatenate([incident_energy, energy_samp])
        vmin_e = max(energy_all.min(), 1e-9) if energy_log_scale else energy_all.min()
        fig1, axes1 = _fig_continuous(
            tsne_enc_sh,
            tsne_samp_sh,
            color_enc=incident_energy,
            color_samp=energy_samp,
            cmap=CMAP_ENERGY,
            vmin=vmin_e,
            vmax=energy_all.max(),
            cb_label="Incident energy [GeV]",
            log_scale=energy_log_scale,
            suptitle="t-SNE coloured by incident energy  (one point per shower)",
        )

        # ── Figure 2: Number of points (per shower) ───────────────────────────────
        npoints_all = np.concatenate([num_points, npoints_samp])
        vmin_p = max(npoints_all.min(), 1e-9) if points_log_scale else npoints_all.min()
        fig2, axes2 = _fig_continuous(
            tsne_enc_sh,
            tsne_samp_sh,
            color_enc=num_points,
            color_samp=npoints_samp,
            cmap=CMAP_NPOINTS,
            vmin=vmin_p,
            vmax=npoints_all.max(),
            cb_label="Number of points",
            log_scale=points_log_scale,
            suptitle="t-SNE coloured by number of points  (one point per shower)",
        )

        # ── Figure 3: Mean Layer (per shower) ───────────────────────────────
        meanlayer_all = np.concatenate([mean_layer_index, mean_layer_samp])
        vmin_l = (
            max(meanlayer_all.min(), 1e-9) if layer_log_scale else meanlayer_all.min()
        )
        fig3, axes3 = _fig_continuous(
            tsne_enc_sh,
            tsne_samp_sh,
            color_enc=mean_layer_index,
            color_samp=mean_layer_samp,
            cmap=CMAP_NPOINTS,
            vmin=vmin_l,
            vmax=meanlayer_all.max(),
            cb_label="Mean Layer Index",
            log_scale=layer_log_scale,
            suptitle="t-SNE coloured by mean index layer  (one point per shower)",
        )

        if save_prefix:
            for fig, tag in [(fig1, "energy"), (fig2, "npoints"), (fig3, "meanlayer")]:
                path = f"{save_prefix}_{tag}.pdf"
                fig.savefig(path, dpi=dpi, bbox_inches="tight")
                print(f"Saved {path}")
        if experiment is not None:
            for fig, tag in [(fig1, "energy"), (fig2, "npoints"), (fig3, "meanlayer")]:
                experiment.log_figure("t-SNE " + tag, figure=fig, step=step)
                plt.close(fig)

    if flag_point_level:
        if not flag_shower_level:
            print("Only plotting point-level t-SNE (skipping shower-level).")
            print(
                "NOT implemeted yet, problems cause points in z are not the same as points in layer_index, not sure if it is possible to do smt"
            )

        layer_index = np.asarray(layer_index, dtype=int)
        assert layer_index.shape[0] == B, f"layer_index shape 0 must be [B={B}]"

        # ── t-SNE run 2: per-point (layer index) ─────────────────────────────────
        print("Running t-SNE [per point]  …")
        tsne_enc_pt, tsne_samp_pt = _tsne_per_point(
            z_encoded, z_sampled, perplexity, n_iter, random_state
        )

        layer_flat = layer_index.reshape(-1)  # [B*N]
        layer_samp_flat = (
            layer_flat
            if layer_index_samp is None
            else np.asarray(layer_index_samp, dtype=int).reshape(-1)
        )
        unique_layers = np.sort(
            np.unique(np.concatenate([layer_flat, layer_samp_flat]))
        )

        # ── Figure 3: Layer index (per point) ─────────────────────────────────────
        fig3, axes3 = _fig_discrete(
            tsne_enc_pt,
            tsne_samp_pt,
            color_enc_flat=layer_flat,
            color_samp_flat=layer_samp_flat,
            unique_vals=unique_layers,
            cmap_name=CMAP_LAYER,
            cb_label="Layer index",
            suptitle="t-SNE coloured by layer index  (one point per hit)",
        )

        if save_prefix:
            for fig, tag in [(fig3, "layer")]:
                path = f"{save_prefix}_{tag}.pdf"
                fig.savefig(path, dpi=dpi, bbox_inches="tight")
                print(f"Saved {path}")

    if not (flag_shower_level or flag_point_level):
        raise ValueError(
            "At least one of flag_shower_level or flag_point_level must be True."
        )
    elif flag_shower_level and not flag_point_level:
        return [(fig1, axes1), (fig2, axes2)]
    elif not flag_shower_level and flag_point_level:
        return [(fig3, axes3)]
    elif flag_shower_level and flag_point_level:
        return [(fig1, axes1), (fig2, axes2), (fig3, axes3)]


# ── Quick demo ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    B = 1000
    n_layers = 47

    dataset = "/data/dust/user/mmozzani/HGCAL_dataset/h5s-photons-PC/HGCal_showers{}.h5"
    f = h5py.File(dataset.format(99), "r")
    events = np.array(f["showers"])[:B]
    z = events[:, :, [0, 1, 3]] + rng.standard_normal((B, events.shape[1], 3)) * 0.3
    incident_energy = np.array(f["gen_info"])[:B, 0]
    num_points = (events[:, :, 3] > 0).sum(axis=1)
    layer_index = events[:, :, 2]

    plot_vae_tsne_conditioning(
        z,
        z,
        incident_energy=incident_energy,
        num_points=num_points,
        layer_index=layer_index,
        flag_shower_level=True,
        flag_point_level=False,
        perplexity=4,
        energy_log_scale=True,
        points_log_scale=True,
        layer_log_scale=True,
        save_prefix="gen_utils_plot/tsne_vae",
    )

    plt.show()
