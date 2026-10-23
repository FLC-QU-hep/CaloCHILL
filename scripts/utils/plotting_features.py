from functools import lru_cache

import matplotlib as mpl
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binned_statistic

mpl.rcParams["xtick.labelsize"] = 18
mpl.rcParams["ytick.labelsize"] = 18
mpl.rcParams["font.size"] = 20
mpl.rcParams["legend.frameon"] = False
mpl.rcParams["text.usetex"] = True
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Helvetica", "Nimbus Sans", "Arial", "DejaVu Sans"]
plt.close()


def c_eff(type="ecal"):
    c = 299.792458  # mm/ns
    if type == "ecal":
        x_cell_size = z_size = 5.0  # mm
        y_layer_size = 0.52  # mm

    elif type == "hcal":
        x_cell_size = z_size = 30.0  # mm
        y_layer_size = 3  # mm

    # c_eff = c (1 + delta_r / r): delta_r = half the cell diagonal, r = c * min_t
    min_t = 0.022159403
    return c * (
        1
        + 0.5
        * np.sqrt((x_cell_size) ** 2 + (z_size) ** 2 + (y_layer_size) ** 2)
        / (c * min_t)
    )


LAYER_BOTTOM_POS = "./files/geometry/layer_bottom_pos_ecal+hcal.npy"
LAYER_TOP_POS = "./files/geometry/layer_top_pos_ecal+hcal.npy"
GUN_POSITION = [0, 1804.7, 0]


def cell_distance_grid(s=180, n_layers=78):
    """(x, layer, z) grid of the gun-to-cell distance [mm] for the projected
    showers, with exactly the positions occ-scale.py's get_time_features uses for
    the velocity (cell index -> unshifted left edge, layer -> mean layer depth).
    The HCAL 30x30 cells sit at indices 75..104 of the merged grid
    (merge_ecal_and_hcal pads 75 empty cells on each side); the padding never
    holds HCAL hits and is NaN."""
    layer_mean_pos = (np.load(LAYER_BOTTOM_POS) + np.load(LAYER_TOP_POS)) / 2
    xz_ecal = np.arange(-450, 450 + 5, 5)[:s].astype(float)
    xz_hcal = np.full(s, np.nan)
    xz_hcal[75:105] = np.arange(-450, 450 + 30, 30)[:30]
    dist = np.full((s, n_layers, s), np.nan)
    for ll in range(n_layers):
        xz = xz_ecal if ll < 30 else xz_hcal
        dist[:, ll, :] = np.sqrt(
            (layer_mean_pos[ll] - GUN_POSITION[1]) ** 2
            + (xz[:, None] - GUN_POSITION[0]) ** 2
            + (xz[None, :] - GUN_POSITION[2]) ** 2
        )
    return dist


@lru_cache(maxsize=1)  # occ-scale.py clips every batch against it
def physical_time_floor(s=180, n_layers=78):
    """(x, layer, z) grid of the earliest causal hit time, distance(gun -> cell) /
    c_eff: a time at or above it is never counted as unphysical by the velocity
    features."""
    c = np.where(np.arange(n_layers) < 30, c_eff("ecal"), c_eff("hcal"))
    # 1e-6 margin: times are stored as float32, and rounding down would put a
    # clamped hit a hair above c_eff
    floor = cell_distance_grid(s, n_layers) / c[None, :, None] * (1 + 1e-6)
    floor.flags.writeable = False  # cached and shared
    return floor


def combined_time_floor(g4_floor=None):
    """The paper's per-cell floor for generated times (occ-scale.py --clip-time-to-geant4):
    the later of the Geant4 per-cell minimum `g4_floor` (NaN where Geant4 never hit,
    which leaves the causal floor) and physical_time_floor(); the causal floor alone
    when g4_floor is None."""
    phys = physical_time_floor()
    return phys if g4_floor is None else np.fmax(g4_floor, phys)


def legend_strip(
    labels, colors, path, g4_patch=None, handlelength=1.5, columnspacing=1.5
):
    """One-row legend, Geant4 then one line per model, saved alone to `path`, for a row
    of panels that carry none. g4_patch: Patch kwargs of the Geant4 entry; None for the
    paper's lightgrey/dimgrey. handlelength/columnspacing None take the rcParams."""
    handles = [
        mpl.patches.Patch(
            **(g4_patch or dict(facecolor="lightgrey", edgecolor="dimgrey", lw=2)),
            label="Geant4",
        )
    ] + [
        mpl.lines.Line2D([], [], color=c, lw=3, label=lab)
        for lab, c in zip(labels, colors)
    ]
    fig = plt.figure(figsize=(0.1, 0.1))
    leg = fig.legend(
        handles=handles,
        loc="center",
        ncol=len(handles),
        frameon=False,
        prop={"size": 18},
        handlelength=handlelength,
        columnspacing=columnspacing,
    )
    fig.canvas.draw()
    bbox = leg.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(path, bbox_inches=bbox)
    plt.close(fig)
    print("wrote", path)


def get_cog2(
    events, Xmin=-450, Xmax=450, cell_size=5, cell_size_hcal=30, y_boundary=30
):
    n, X, Y, Z = events.shape

    # ECal layers (Y < y_boundary) use a linear grid at cell_size. HCal layers are
    # projected at cell_size_hcal and zero-padded into the centre of the X/Z axes
    # (merge_ecal_and_hcal), so they need their own full-width grid. Both add back
    # the mean (cell/2) of the projection's Uniform(0, cell) bin-edge dither.
    hcal_bins = int(round((Xmax - Xmin) / cell_size_hcal))
    pad = (X - hcal_bins) // 2

    x_vals_ecal = np.linspace(Xmin + cell_size / 2, Xmax - cell_size / 2, X)
    x_vals_ecal = x_vals_ecal + cell_size / 2

    x_vals_hcal = np.zeros(X)
    x_vals_hcal[pad : pad + hcal_bins] = (
        Xmin
        + cell_size_hcal / 2
        + np.arange(hcal_bins) * cell_size_hcal
        + cell_size_hcal / 2
    )

    y_vals = np.arange(Y) + 0.5
    is_ecal_layer = (np.arange(Y) < y_boundary)[None, :, None]  # (1, Y, 1)

    xx_ecal, yy, zz_ecal = np.meshgrid(x_vals_ecal, y_vals, x_vals_ecal, indexing="ij")
    xx_hcal, _, zz_hcal = np.meshgrid(x_vals_hcal, y_vals, x_vals_hcal, indexing="ij")

    xx = np.where(is_ecal_layer, xx_ecal, xx_hcal)[None]
    zz = np.where(is_ecal_layer, zz_ecal, zz_hcal)[None]
    yy = yy[None]

    total_energy = events.sum(axis=(1, 2, 3), keepdims=True)

    cog_x = (events * xx).sum(axis=(1, 2, 3)) / total_energy.squeeze()
    cog_y = (events * yy).sum(axis=(1, 2, 3)) / total_energy.squeeze()
    cog_z = (events * zz).sum(axis=(1, 2, 3)) / total_energy.squeeze()

    return cog_x, cog_y, cog_z


class Configs:
    def __init__(self):
        # legend font
        self.font = font_manager.FontProperties(family="serif", size=25)
        self.text_size = 25

        self.title = " "
        self.font_ratio = 25
        self.font_labely = 25
        self.font_labelx = 25
        # error band
        self.alpha = 0.3
        self.lw = 5
        self.c_real = 299.792458  # mm/ns
        self.c = np.max([c_eff(type="ecal"), c_eff(type="hcal")])  # mm/ns

        # radial profile
        self.bins_r = np.linspace(0, 30, 31)
        self.bins_r_mm = np.linspace(
            0, 450, 15
        )  # note: at least a dimension of hcal cell size=30 of the bins
        self.origin = (0, -72)
        self.xrange_radial = [0, 30]
        self.ylim_radial = (2e-1, 2e3)
        self.ylim_radial_mm = (1e-1, 2e5)

        # occupancy
        self.occup_bins = np.linspace(-100, 1000, 50)  # for 30 GeV @ (0,-50)
        self.plot_text_occupancy = False
        self.occ_indent = 20
        # Nhits
        self.occ_bins = np.linspace(-2, 3000, 40)
        self.ylim_Nhits = None  # None: 1.3 x the tallest histogram (scales with shw)
        # e_sum
        self.e_sum_bins = np.linspace(0, 2300, 25)  # for full spectrum
        self.ylim_esum = (20, 3e3)
        # e ratio
        self.ratio_bins = np.logspace(np.log10(0.002), np.log10(0.09), 40)
        # start L
        self.startL_bins = 36
        self.plot_text_e = False
        self.plot_legend_e = True
        self.e_indent = 20

        # hits
        self.hit_bins = np.logspace(np.log10(0.01000001), np.log10(200), 60)
        self.ylim_hits = (10, 1e7)
        # x y z
        self.bins_xz = 25
        self.bins_y = 78
        self.xyz_ranges = [(-450, 450), (0, 77), (-450, 450)]

        # CoG
        self.bins_cog = 30
        # bin ranges for [X, Y, Z] coordinates, in ILD coordinate system [X', Y', Z']
        # X/Z [mm]: 50 GeV pion COG spread is ~7-12 mm std, with tails past +/-50 mm
        self.cog_ranges = [(-40, 40), (9, 76), (-40, 40)]

        # xyz featas
        self.bins_feats = 50
        # bin ranges for [X, Z, Y] coordinates, in ILD coordinate system [X', Y', Z']
        self.feats_ranges = [(-250, 250), (1800, 2011), (-250, 250)]

        # time
        self.time_bins = np.logspace(np.log10(5e-3), np.log10(800), 50)
        self.unphysical_hits_shw_bins = np.linspace(0, 50, 50)
        self.velocity_bins = np.logspace(np.log10(5e-3), np.log10(1e5), 50)

        # velocity bins are in a way that c exatly lies in between two bins
        edges = self.velocity_bins
        log_edges = np.log10(edges)
        log_mids = log_edges[1:]

        # Find the bin whose midpoint is closest to c
        closest_mid_idx = np.argmin(np.abs(10**log_mids - self.c))

        # Compute the offset to shift log_edges so that c lies exactly between two bins
        # We want c to be at the midpoint between log_edges[i+1] and log_edges[i]
        target_mid = np.log10(self.c)
        current_mid = log_mids[closest_mid_idx]
        offset = target_mid - current_mid

        shifted_log_edges = log_edges + offset
        self.velocity_bins = 10**shifted_log_edges

        self.threshold = 0.1  # MeV / half a MIP
        self.color_lines = [
            "tab:green",
            "tab:orange",
            "tab:blue",
            "tab:red",
            "tab:purple",
            "tab:pink",
        ]
        self.include_artifacts = (
            False  # include_artifacts = True -> keeps points that hit dead material
        )
        # percentile edges for occupancy based metrics
        self.layer_edges = [0, 8, 11, 13, 15, 16, 18, 19, 21, 24, 29]
        self.radial_edges = [
            0,
            6.558,
            9.849,
            12.96,
            17.028,
            23.434,
            33.609,
            40.119,
            48.491,
            68.808,
            300,
        ]


plt_config = Configs()


def ratio_plots(
    axs,
    geant4_data,
    gen_data_list,
    err_data,
    err_gen_list,
    bins,
    pos,
    plt_config=plt_config,
):
    # ratio plot on the bottom
    lims_min, lims_max = getattr(
        plt_config, "ratio_lims", (0.6, 1.4)
    )  # paper_outputs.py sets it per panel
    eps = 1e-5

    for i, gen_data in enumerate(gen_data_list):
        centers = pos
        ratios = np.clip((gen_data + eps) / (geant4_data + eps), lims_min, lims_max)
        mask = (ratios > lims_min) & (
            ratios < lims_max
        )  # mask ratios within plotting y range
        # only connect dots with adjecent points
        starts = np.argwhere(np.insert(mask[:-1], 0, False) < mask)[:, 0]
        ends = np.argwhere(np.append(mask[1:], False) < mask)[:, 0] + 1
        indexes = np.stack((starts, ends)).T
        for idxs in indexes:
            axs.stairs(
                ratios[idxs[0] : idxs[1]],
                edges=bins[idxs[0] : idxs[1] + 1],
                # stairs' default baseline=0 draws a vertical edge down to 0 at
                # both ends of every segment, which reads as a crash in the ratio
                baseline=None,
                color=plt_config.color_lines[i],
                lw=2,
            )
        # remaining points either above or below plotting y range
        mask = ratios == lims_min
        if len(ratios[mask]) > 0:
            axs.plot(
                centers[mask],
                ratios[mask],
                linestyle="",
                lw=2,
                marker="v",
                color=plt_config.color_lines[i],
                clip_on=False,
            )

        mask = ratios == lims_max
        if len(ratios[mask]) > 0:
            axs.plot(
                centers[mask],
                ratios[mask],
                linestyle="",
                lw=2,
                marker="^",
                color=plt_config.color_lines[i],
                clip_on=False,
            )

        # error band only over the in-range segments, clipped to the plotting range:
        # near-zero bins would otherwise blow up the relative error and fill the axis
        ratio_err = ratios * np.sqrt(
            ((err_gen_list[i]) / (gen_data)) ** 2 + ((err_data) / (geant4_data)) ** 2
        )
        for idxs in indexes:
            axs.stairs(
                np.clip(
                    ratios[idxs[0] : idxs[1]] + ratio_err[idxs[0] : idxs[1]],
                    lims_min,
                    lims_max,
                ),
                edges=bins[idxs[0] : idxs[1] + 1],
                baseline=np.clip(
                    ratios[idxs[0] : idxs[1]] - ratio_err[idxs[0] : idxs[1]],
                    lims_min,
                    lims_max,
                ),
                color=plt_config.color_lines[i],
                alpha=plt_config.alpha,
                fill=True,
            )

    axs.set_ylim(lims_min, lims_max)
    axs.axhline(1, linestyle="-", lw=1, color="k")


def add_zoom(ax, edges, ref, gens, colors, zoom):
    """Inset on ax: Geant4 (filled grey) and the generated histograms (lines) between zoom["x"], linear y.
    zoom = dict(x=(lo, hi), bounds=[x0, y0, w, h] in axes fraction, logx=bool)."""
    ins = ax.inset_axes(zoom["bounds"])
    ins.stairs(ref, edges=edges, color="lightgrey", fill=True)
    ins.stairs(ref, edges=edges, color="dimgrey", lw=1.5)
    for g, c in zip(gens, colors):
        ins.stairs(g, edges=edges, color=c, lw=2)
    lo, hi = zoom["x"]
    if zoom.get("logx"):
        ins.set_xscale("log")
    ins.set_xlim(lo, hi)
    centres = (np.asarray(edges[:-1]) + np.asarray(edges[1:])) / 2
    m = (centres >= lo) & (centres <= hi)
    vals = np.concatenate([np.asarray(ref)[m]] + [np.asarray(g)[m] for g in gens])
    pad = 0.08 * (vals.max() - vals.min())
    ins.set_ylim(vals.min() - pad, vals.max() + pad)
    ins.tick_params(labelsize=11)
    ins.ticklabel_format(
        axis="y", style="sci", scilimits=(-2, 3)
    )  # 9e5 as 9 x 10^5, not 900000
    ins.yaxis.get_offset_text().set_fontsize(10)
    ax.indicate_inset_zoom(ins, edgecolor="0.3")
    return ins


def plt_spinal(
    e_layers,
    e_layers_list,
    shw,
    labels,
    gen_counts=None,
    my_dir=None,
    plt_config=plt_config,
    title=plt_config.title,
    zoom=None,
):
    fig, axs = plt.subplots(2, 1, figsize=(7, 9), height_ratios=[3, 1], sharex=True)
    # for legend
    axs[0].hist(
        np.zeros(1) - 10, label=labels[0], color="lightgrey", edgecolor="dimgrey", lw=2
    )
    for i in range(len(e_layers_list)):
        axs[0].plot(
            0,
            0,
            linestyle="-",
            lw=3,
            color=plt_config.color_lines[i],
            label=labels[i + 1],
        )

    pos = np.arange(1, len(e_layers) + 1)
    bins = np.arange(0.5, len(e_layers) + 1.5)

    axs[0].stairs(e_layers, edges=bins, color="lightgrey", fill=True)

    # uncertainty band
    err_data = e_layers.std(axis=0) / np.sqrt(
        shw
    )  # std of individual measures --> std of mean
    axs[0].stairs(
        e_layers + err_data,
        edges=bins,
        baseline=e_layers - err_data,
        color="dimgrey",
        lw=2,
        hatch="///",
    )
    err_gen_list = []
    for i, e_layers_ in enumerate(e_layers_list):
        # each source's own shower count for its std-of-the-mean, since it
        # can differ from `shw` (e.g. CaloHadronic vs. the main comparison)
        n_i = gen_counts[i] if gen_counts is not None else shw
        err_gen = e_layers_.std(axis=0) / np.sqrt(n_i)
        axs[0].hist(
            pos,
            bins=bins,
            weights=e_layers_,
            histtype="step",
            linewidth=3,
            color=plt_config.color_lines[i],
        )
        axs[0].stairs(
            e_layers_ + err_gen,
            edges=bins,
            baseline=e_layers_ - err_gen,
            color=plt_config.color_lines[i],
            alpha=plt_config.alpha,
            lw=plt_config.lw,
            fill=True,
        )
        err_gen_list.append(err_gen)
    # ratio plot on the bottom
    ratio_plots(
        axs[1],
        e_layers,
        e_layers_list,
        err_data,
        err_gen_list,
        bins,
        pos,
        plt_config=plt_config,
    )
    axs[0].set_yscale("log")
    axs[0].set_ylim(1.1e-1, 2e2)
    axs[0].set_xlim(0, 79)
    if zoom is not None:
        add_zoom(axs[0], bins, e_layers, e_layers_list, plt_config.color_lines, zoom)
    plt.xlabel("layers", fontsize=plt_config.font_labelx)
    axs[0].set_ylabel("mean energy [MeV]", fontsize=plt_config.font_labely)

    plt.legend(prop=plt_config.font, loc="best")
    axs[0].set_title(title, fontsize=plt_config.font.get_size(), loc="right")
    axs[1].set_ylabel("ratio to G4", fontsize=plt_config.font_ratio)
    plt.subplots_adjust(hspace=0.1)

    if my_dir is not None:
        fig.tight_layout()
        plt.savefig(my_dir + "spinal.pdf", dpi=100, bbox_inches="tight")
    plt.close()


def plt_radial_mm(
    e_radial,
    e_radial_list,
    events,
    labels,
    gen_counts=None,
    my_dir=None,
    cfg=plt_config,
    title=plt_config.title,
):
    fig, axs = plt.subplots(2, 1, figsize=(7, 9), height_ratios=[3, 1], sharex=True)

    # for legend
    axs[0].hist(
        np.zeros(1) - 10,
        label=labels[0],
        color="lightgrey",
        edgecolor="dimgrey",
        lw=2,
        range=(0, 1),
    )
    for i in range(len(e_radial_list)):
        axs[0].plot(
            0, 0, linestyle="-", lw=3, color=cfg.color_lines[i], label=labels[i + 1]
        )
    axs[0].set_title(title, fontsize=cfg.font.get_size(), loc="right")
    axs[0].legend(prop=cfg.font, loc="upper right")

    # mean and std as binned statistic
    mean, edges, _ = binned_statistic(
        e_radial[0], e_radial[1], bins=plt_config.bins_r_mm, statistic="mean"
    )
    std, _, _ = binned_statistic(
        e_radial[0], e_radial[1], bins=plt_config.bins_r_mm, statistic="std"
    )
    count, _, _ = binned_statistic(
        e_radial[0], e_radial[1], bins=plt_config.bins_r_mm, statistic="count"
    )

    mean_shower = mean * count / events  # mean shower energy per bin
    std_shower = std / np.sqrt(
        count
    )  # std of individual event measures --> std of mean
    std_shower = std_shower * count / events  # std of shower energy per bin

    # data histogram
    axs[0].stairs(mean_shower, edges=edges, color="lightgrey", fill=True)
    # uncertainty band
    axs[0].stairs(
        mean_shower + std_shower,
        edges=edges,
        baseline=mean_shower - std_shower,
        color="dimgrey",
        lw=3,
        hatch="///",
    )
    mean_shower_gen_list, err_gen_list = [], []
    for i, e_radial_ in enumerate(e_radial_list):
        # own shower count for this source (can differ from `events`, e.g. CaloHadronic)
        events_i = gen_counts[i] if gen_counts is not None else events
        # mean and std as binned statistic
        mean, _, _ = binned_statistic(
            e_radial_[0], e_radial_[1], bins=edges, statistic="mean"
        )
        std, _, _ = binned_statistic(
            e_radial_[0], e_radial_[1], bins=edges, statistic="std"
        )
        count, _, _ = binned_statistic(
            e_radial_[0], e_radial_[1], bins=edges, statistic="count"
        )
        mean_shower_gen = mean * count / events_i  # mean shower energy per bin
        mean_shower_gen_list.append(mean_shower_gen)
        std_shower_gen = std / np.sqrt(
            count
        )  # std of individual event measures --> std of mean
        std_shower_gen = (
            std_shower_gen * count / events_i
        )  # std of shower energy per bin
        err_gen_list.append(std_shower_gen)
        # gen histogram
        axs[0].stairs(
            mean_shower_gen,
            edges=edges,
            linestyle="-",
            lw=3,
            color=plt_config.color_lines[i],
        )
        # uncertainty band
        axs[0].stairs(
            mean_shower_gen + std_shower_gen,
            edges=edges,
            baseline=mean_shower_gen - std_shower_gen,
            color=plt_config.color_lines[i],
            alpha=plt_config.alpha,
            fill=True,
        )

    # ratio plot on the bottom
    ratio_plots(
        axs[1],
        mean_shower,
        mean_shower_gen_list,
        std_shower,
        err_gen_list,
        edges,
        pos=np.array((edges[:-1] + edges[1:]) / 2),
        plt_config=plt_config,
    )

    axs[0].set_ylim(plt_config.ylim_radial_mm[0], plt_config.ylim_radial_mm[1])
    axs[0].set_yscale("log")
    plt.xlabel("radius [mm]", fontsize=plt_config.font_labelx)
    axs[0].set_ylabel("mean energy [MeV]", fontsize=plt_config.font_labely)
    axs[1].set_ylabel("ratio to G4", fontsize=plt_config.font_ratio)
    plt.subplots_adjust(hspace=0.1)

    if my_dir is not None:
        fig.tight_layout()
        plt.savefig(my_dir + "radial_mm.pdf", dpi=100, bbox_inches="tight")
    plt.close()


def plt_hit_e(
    hits,
    hits_list,
    events,
    labels,
    gen_counts=None,
    my_dir=None,
    plt_config=plt_config,
    title=plt_config.title,
    zoom=None,
):
    fig, axs = plt.subplots(2, 1, figsize=(7, 9), height_ratios=[3, 1], sharex=True)
    # for legend
    axs[0].hist(
        np.zeros(1) + 1, label=labels[0], color="lightgrey", edgecolor="dimgrey", lw=3
    )
    for i in range(len(hits_list)):
        axs[0].plot(
            0,
            0,
            linewidth=3,
            color=plt_config.color_lines[i],
            label=labels[i + 1],
        )
    plt.legend(prop=plt_config.font, loc="upper right")
    axs[0].set_title(title, fontsize=plt_config.font.get_size(), loc="right")

    h_data = axs[0].hist(
        hits, bins=plt_config.hit_bins, color="lightgrey", rasterized=True
    )

    # uncertainty band
    h_data_err = np.sqrt(h_data[0])
    axs[0].stairs(
        h_data[0] + h_data_err,
        edges=h_data[1],
        baseline=h_data[0] - h_data_err,
        color="dimgrey",
        lw=3,
        hatch="///",
    )

    data_gen_list, err_gen_list = [], []
    for i, hits_ in enumerate(hits_list):
        # rescale to the G4 shower count; gen_counts[i] can differ from `events`
        # (e.g. CaloHadronic is evaluated on fewer showers)
        scale = events / gen_counts[i] if gen_counts is not None else 1.0
        counts_raw, edges_gen = np.histogram(hits_, bins=h_data[1])
        h_gen = counts_raw * scale
        axs[0].stairs(
            h_gen,
            edges=edges_gen,
            linestyle="-",
            linewidth=3,
            color=plt_config.color_lines[i],
        )
        data_gen_list.append(h_gen)

        # uncertainty band in histogram (Poisson error on raw counts, rescaled)
        h_gen_err = np.sqrt(counts_raw) * scale
        axs[0].stairs(
            h_gen + h_gen_err,
            edges=edges_gen,
            baseline=h_gen - h_gen_err,
            color=plt_config.color_lines[i],
            alpha=plt_config.alpha,
            fill=True,
        )
        err_gen_list.append(h_gen_err)

    # ratio plot on the bottom
    ratio_plots(
        axs[1],
        h_data[0],
        data_gen_list,
        h_data_err,
        err_gen_list,
        h_data[1],
        pos=np.array((h_data[1][:-1] + h_data[1][1:]) / 2),
        plt_config=plt_config,
    )

    axs[0].set_xlim(h_data[1].min(), h_data[1].max())
    axs[0].set_ylim(plt_config.ylim_hits[0], plt_config.ylim_hits[1])
    axs[0].set_yscale("log")
    axs[0].set_xscale("log")
    if zoom is not None:
        add_zoom(
            axs[0],
            h_data[1],
            h_data[0],
            data_gen_list,
            plt_config.color_lines,
            {**zoom, "logx": True},
        )
    axs[0].set_ylabel("# cells", fontsize=plt_config.font_labelx)
    axs[1].set_ylabel(
        "ratio to G4", fontsize=plt_config.font_ratio
    )  # as plt_spinal / plt_radial_mm
    plt.xlabel("visible cell energy [MeV]", fontsize=plt_config.font_labelx)

    if my_dir is not None:
        fig.tight_layout()
        plt.savefig(my_dir + "hits.pdf", dpi=100, bbox_inches="tight")
    plt.close()
