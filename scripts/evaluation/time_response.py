"""Time of the first hit in a cell (paper fig. 12), Geant4 vs this work (AllShowers and CaloHadronic generate no time),
on the full-range 100k samples of fig. 5 (paper_outputs.SETS), Geant4, ds 128 and the large ds 256.

  first_hit_100k  per fig. 5 source, time-of-first-hit spectrum of the ECal and HCal cells -> files/time_response/first_hit_100k.npz
                  (N_MAX: number of showers, default 100000)
  plot_paper      the figure from that npz -> <out>/4_first_hit_100k.{pdf,png}
  linearity       paper fig. 15, linearity and resolution of the visible energy of the fixed-energy 10k samples
                  (occ-scale caches e_sum_list / inc_en.npy) -> <out>/linearity_resolution_models.{pdf,png}

    .venv/bin/python scripts/evaluation/time_response.py first_hit_100k
    .venv/bin/python scripts/evaluation/time_response.py plot_paper --out figs/time_response
    .venv/bin/python scripts/evaluation/time_response.py linearity --out figs/time_response [--horizontal]
"""

import os
import sys
from multiprocessing import Pool

import h5py
import numpy as np

ROOT = "."
PROJ = ROOT + "/files/projected_array/"
OUT_NPZ = ROOT + "/files/time_response/"


def floored(name, m, t):
    """Generated cell times raised to the paper's per-cell floor (occ-scale --clip-time-to-geant4: the later of the
    Geant4 per-cell minimum and distance/c_eff); the projected hdf5 stores them unclamped. Geant4 untouched."""
    global FLOOR
    if name == "g4":
        return t
    if FLOOR is None:
        sys.path.append(ROOT + "/scripts")
        from utils.plotting_features import combined_time_floor

        FLOOR = combined_time_floor(np.load(PROJ + "Geant4/time_floor.npy"))
    return np.where(t > 0, np.fmax(t, FLOOR[m]), t)


FLOOR = None
E_THR = 1e-2  # MeV: the 10 keV cell threshold of the paper (occ-scale threshold); the projected hdf5 keeps cells to ~1 keV
FIG5 = "ds_0GeV_calibEfull_large_do01"  # paper fig. 5 (paper_outputs.SETS): the full-range 100k samples


def fig5_sources():
    """[(key, label, projected folder, colour)] of the paper's fig. 5, Geant4 first."""
    import matplotlib

    sys.path.insert(0, ROOT + "/scripts")
    with matplotlib.rc_context():  # paper_outputs sets its own rcParams on import
        from evaluation.paper_outputs import SETS
    g4, src, _ = SETS[FIG5]
    return [("g4", "Geant4", g4, "lightgrey")] + [
        (f"m{i}", lab, f, c) for i, (lab, f, c, _) in enumerate(src)
    ]


def first_hit_100k(args):
    """Showers i0..i1 of one fig. 5 sample, ECal and HCal: time-of-first-hit spectrum (counts)."""
    key, folder, i0, i1 = args
    out = {c: np.zeros(len(TBINS) - 1) for c in ("ecal", "hcal")}
    with h5py.File(PROJ + folder + "/events_100000.hdf5", "r") as f:
        en, tm = f["projected_showers"], f["projected_time_showers"]
        for i in range(i0, i1):
            e = en[i]
            m = e > 0
            t = floored(key, m, tm[i][m])
            y = np.nonzero(m)[1]
            e = e[m]
            ok = np.isfinite(t) & (t >= 0) & (e >= E_THR)
            for c, sel in (("ecal", ok & (y < 30)), ("hcal", ok & (y >= 30))):
                out[c] += np.histogram(t[sel], TBINS)[0]
    return key, out


TBINS = np.concatenate([[0.0], np.logspace(-2, np.log10(200.0), 120)])  # ns


def plot_paper(out):
    """4_first_hit_100k: the full-range 100k samples of fig. 5, Geant4, ds 128 and the large ds 256 only (as fig. 6),
    ECal + HCal / ECal / HCal, ratio to Geant4 below each panel with its statistical band."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Nimbus Sans", "Arial", "DejaVu Sans"],
            "text.latex.preamble": r"\usepackage{helvet}\usepackage{sansmath}\sansmath",  # sans numbers too
            "font.size": 22,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.frameon": False,
        }
    )
    os.makedirs(out, exist_ok=True)

    def save(fig, name):
        for ext in ("pdf", "png"):
            fig.savefig(f"{out}/{name}.{ext}", bbox_inches="tight", dpi=150)
        plt.close(fig)
        print("wrote", f"{out}/{name}.pdf")

    def hist_panel(
        ax, axr, edges, h, herr, e, rlim=(0.5, 1.5), ls="-", g4_fill=True, cols=None
    ):
        """Geant4 filled, models as lines; ratio with its statistical band, black line at 1 (ratio_plots style)."""
        if g4_fill:
            ax.stairs(h["g4"], edges, fill=True, color="lightgrey")
        else:
            ax.stairs(h["g4"], edges, color="dimgrey", lw=2.5, ls=ls, baseline=None)
        for k, col in cols.items():
            ax.stairs(h[k], edges, color=col, lw=2.5, ls=ls, baseline=None)
            ok = h["g4"] > 0
            r = np.where(ok, h[k] / np.where(ok, h["g4"], 1), np.nan)
            re = r * np.hypot(
                herr[k] / np.where(h[k] > 0, h[k], 1),
                herr["g4"] / np.where(ok, h["g4"], 1),
            )
            axr.stairs(np.clip(r, *rlim), edges, color=col, lw=2, baseline=None)
            axr.stairs(
                np.clip(r + re, *rlim),
                edges,
                baseline=np.clip(r - re, *rlim),
                color=col,
                alpha=0.3,
                fill=True,
            )
        axr.axhline(1, color="grey", lw=1)
        axr.set_ylim(*rlim)
        axr.set_ylabel(f"{e} GeV\nratio")

    F = np.load(
        OUT_NPZ + "first_hit_100k.npz"
    )  # Geant4, ds 128 and the large ds 256 only, as fig. 6
    src = [
        x
        for x in fig5_sources()
        if x[1] in ("Geant4", r"$d_s = 128$", r"$d_s = 256$, large")
    ]
    cols = {k: c for k, _, _, c in src[1:]}
    # one column: ECal + HCal on top, ECal | HCal smaller below; ratio panels a quarter of each block
    fig = plt.figure(figsize=(10, 13))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.25, 1.1], hspace=0.28, top=0.92)
    top = outer[0].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.06)
    bot = outer[1].subgridspec(2, 2, height_ratios=[3, 1], hspace=0.06, wspace=0.42)
    panels = [(fig.add_subplot(top[0]), "both", "ECal + HCal")] + [
        (fig.add_subplot(bot[0, j]), p, name)
        for j, (p, name) in enumerate((("ecal", "ECal"), ("hcal", "HCal")))
    ]
    ratios = [fig.add_subplot(top[1], sharex=panels[0][0])] + [
        fig.add_subplot(bot[1, j], sharex=panels[1 + j][0]) for j in range(2)
    ]
    handles = [Line2D([], [], color="dimgrey", lw=3, label="Geant4")] + [
        Line2D([], [], color=c, lw=3, label=lab) for _, lab, _, c in src[1:]
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        bbox_to_anchor=(0.5, 0.925),
        fontsize=20,
        handlelength=1.5,
        columnspacing=1.2,
    )
    for (ax, p, name), axr in zip(panels, ratios):
        n = {
            k: F[f"{k}_ecal"] + F[f"{k}_hcal"] if p == "both" else F[f"{k}_{p}"]
            for k, *_ in src
        }
        h = {k: v / F["n"] for k, v in n.items()}
        he = {k: np.sqrt(v) / F["n"] for k, v in n.items()}
        hist_panel(
            ax, axr, TBINS, h, he, None, rlim=(0, 2), g4_fill=False, cols=cols
        )  # G4 unfilled step
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.tick_params(labelbottom=False)
        axr.set_xlim(0.05, 200)
        ax.set_ylim(5e-2, 3 * max(v.max() for v in h.values()))
        ax.text(0.05, 0.94, name, transform=ax.transAxes, va="top")
        ax.set_ylabel("Mean number of cells")
        axr.set_ylabel("Ratio")
        axr.set_yticks([0.5, 1.0, 1.5])
        axr.set_xlabel("Time of first hit in a cell [ns]")
    for (ax, _, _), axr in zip(
        panels[1:], ratios[1:]
    ):  # small panels: shorter label, fewer ratio ticks
        ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(numticks=4))
        axr.set_yticks([0.5, 1.5])
        axr.set_xlabel("First-hit time [ns]")
    save(fig, "4_first_hit_100k")


ENERGIES = list(range(10, 100, 10))
G4_10K = {
    E: f"Geant4/{E}GeV_g4win{'9.4-10.6' if E == 10 else f'{E - 1}.5-{E}.5'}_10000"
    for E in ENERGIES
}
P = "pions_pions_HGx9_LatentDiff_SWD_zdim128_"
LINEARITY_MODELS = [  # (label, projected folder, colour): the paper's fig. 5 / 6 models and colours
    ("CaloHadronic", "HGx9_CNF_ECAL_HCAL_30", "#7b3fa0"),
    ("AllShowers NFE 32", "AllShowers_mid32", "#c41e3a"),
    ("AllShowers NFE 4", "AllShowers_mid4", "#eb6834"),
    (
        r"$d_s = 128$",
        P
        + "ds128_2LastattnLayer_2026_09_16__16_14_55__mean_flow_2026_09_22__10_40_48_calibEfull__1",
        "#3987e5",
    ),
    (
        r"$d_s = 256$, large LMF",
        P
        + "ds256_KL1e-2_2LastattnLayer_2026_08_24__13_18_40__mean_flow_2026_10_06__22_25_31_it1199k_calib100k__1",
        "#2e9e5b",
    ),
]


def linearity(out, vertical=True):
    """Linearity and resolution of the rescaled visible energy (as occ-scale_MultipleEnergies.plotting_linearity_rescaled_ds):
    per energy the core-90 of the visible energy, mu90 = its mean, sigma90 = its std, errors from 200
    bootstrap resamplings of the showers; one factor k (least-squares k * mu90_G4 = E_inc through the origin) turns MeV
    into GeV for every source. Each shower is scaled by E / E_inc (Geant4 windows are E +- 0.5)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    sys.path.append(ROOT + "/scripts")
    import utils.plotting_features  # noqa: F401  (sets the paper rcParams: usetex, tick sizes)

    def compute_core90(x):
        """The central 90% of the values (5th to 95th percentile by rank)."""
        x = np.sort(x)
        return x[int(len(x) * 0.05) : int(len(x) * 0.95)]

    rng = np.random.default_rng(0)

    def stats(folders, key, n_boot=200):
        """mu90, sigma90 of the core-90 and their bootstrap errors (resampling the showers, core-90 redone each time)."""
        res = []
        for E, d in zip(ENERGIES, folders):
            x = (
                np.load(PROJ + d + f"/{key}.npy")
                * E
                / np.load(PROJ + d + "/inc_en.npy").reshape(-1)
            )
            y = compute_core90(x)
            boot = np.array(
                [
                    (b.mean(), b.std())
                    for b in (
                        compute_core90(rng.choice(x, len(x))) for _ in range(n_boot)
                    )
                ]
            )
            res.append((y.mean(), boot[:, 0].std(), y.std(), boot[:, 1].std()))
        return np.array(res).T  # mu, mu_err, sigma, sigma_err

    E = np.array(ENERGIES, float)
    g = stats([G4_10K[e] for e in ENERGIES], "e_sum_list_r")
    k = np.sum(E * g[0]) / np.sum(g[0] ** 2)  # GeV per MeV visible
    print(f"k = {k:.5g} GeV/MeV (sampling fraction {1e-3 / k:.4f})")
    g = g * k
    fs = [
        stats([f"{d}/{e}GeV_10000" for e in ENERGIES], "e_sum_list") * k
        for _, d, _ in LINEARITY_MODELS
    ]

    plt.rcParams.update({"font.size": 14})
    fig = plt.figure(figsize=(7, 9.6) if vertical else (14, 7))
    outer = (
        gridspec.GridSpec(2, 1, hspace=0.22)
        if vertical
        else gridspec.GridSpec(1, 2, wspace=0.28)
    )
    dx = np.linspace(-1.6, 1.6, len(LINEARITY_MODELS) + 1)
    for j, (m, ylab, rlab, rlim) in enumerate(
        [
            (
                0,
                r"$\mu_{90}$ [GeV]",
                r"$\mu_{90} / \mu_{90}^{\mathrm{G4}}$",
                (0.95, 1.05),
            ),  # ratio ranges of AllShowers fig. 11
            (
                2,
                r"$\sigma_{90} / E_\mathrm{inc}$",
                r"$\sigma_{90} / \sigma_{90}^{\mathrm{G4}}$",
                (0.8, 1.55),
            ),
        ]
    ):
        inner = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[j], height_ratios=[2.2, 1], hspace=0
        )
        ax, axr = fig.add_subplot(inner[0]), fig.add_subplot(inner[1], sharex=None)
        norm = 1 if m == 0 else E
        eb = dict(fmt="o", ms=4.5, capsize=4, elinewidth=1.4, capthick=1.4)
        ax.errorbar(
            E + dx[0],
            g[m] / norm,
            yerr=g[m + 1] / norm,
            color="#3a4149",
            label="Geant4",
            zorder=3,
            **eb,
        )
        for (lab, _, c), f, x in zip(LINEARITY_MODELS, fs, dx[1:]):
            ax.errorbar(
                E + x, f[m] / norm, yerr=f[m + 1] / norm, color=c, label=lab, **eb
            )
            r = f[m] / g[m]
            re = r * np.hypot(f[m + 1] / f[m], g[m + 1] / g[m])
            inside = (r >= rlim[0]) & (r <= rlim[1])
            axr.errorbar(E[inside] + x, r[inside], yerr=re[inside], color=c, **eb)
            for sel, yv, mk in [
                (r < rlim[0], rlim[0], "v"),
                (r > rlim[1], rlim[1], "^"),
            ]:  # off-scale: triangle on the edge
                axr.plot(
                    E[sel] + x, np.full(sel.sum(), yv), mk, color=c, ms=8, clip_on=False
                )
            print(
                f"{'mu90' if m == 0 else 'sigma90'} ratio {lab:>24}: "
                + " ".join(f"{v:.3f}" for v in r)
            )
        axr.axhline(1, color="grey", ls="--", lw=1.2)
        if m == 0:
            ax.plot([5, 95], [5, 95], color="grey", lw=1.2, zorder=0)
            ax.set_ylim(5, 100)
            ax.legend(fontsize=11, loc="upper left", frameon=False)
        else:
            ax.set_ylim(0.06, 0.21)
        for a in (ax, axr):
            a.set_xlim(5, 95)
        ax.tick_params(labelbottom=False)
        ax.yaxis.set_major_locator(
            plt.MaxNLocator(5, prune="lower")
        )  # no label on the join with the ratio panel
        ax.set_ylabel(ylab)
        axr.set_ylim(*rlim)
        axr.set_ylabel(rlab)
        axr.set_xlabel(r"$E_\mathrm{inc}$ [GeV]")
    os.makedirs(out, exist_ok=True)
    fig.savefig(f"{out}/linearity_resolution_models.pdf", bbox_inches="tight")
    fig.savefig(f"{out}/linearity_resolution_models.png", dpi=130, bbox_inches="tight")
    print("wrote", f"{out}/linearity_resolution_models.pdf")


if __name__ == "__main__":
    if sys.argv[1] == "first_hit_100k":
        src, n = fig5_sources(), int(os.environ.get("N_MAX", 100000))
        step = min(5000, n)
        jobs = [(k, f, i, i + step) for k, _, f, _ in src for i in range(0, n, step)]
        acc = {}
        with Pool(os.cpu_count()) as p:
            for k, h in p.imap_unordered(first_hit_100k, jobs):
                for q, v in h.items():
                    acc[f"{k}_{q}"] = acc.get(f"{k}_{q}", 0) + v
        os.makedirs(OUT_NPZ, exist_ok=True)
        np.savez(OUT_NPZ + "first_hit_100k.npz", n=n, **acc)
    elif sys.argv[1] == "plot_paper":
        plot_paper(sys.argv[sys.argv.index("--out") + 1])
    elif sys.argv[1] == "linearity":
        linearity(
            sys.argv[sys.argv.index("--out") + 1],
            vertical="--horizontal" not in sys.argv,
        )
