"""Generation time against accuracy, as the time-vs-quality figure of the accelerated-sampling paper.

Accuracy: for each model the geometric mean, over the observables of table tab:three_energies and the energies
20 / 50 / 80 GeV, of the Jensen-Shannon divergence to Geant4 divided by the Geant4-vs-Geant4 value at that energy
(1 = statistically indistinguishable from Geant4). Error: standard deviation over 1000 Gaussian resamplings of every
divergence within its uncertainty (as the geometric means of the tables). Time: GPU time per shower of table
tab:timing_comparison (NVIDIA A100), at batch size 1 (filled) and at the large batch size of the table
(open: 1400, CaloHadronic 256), joined by a line; Geant4 on one CPU core.

    python scripts/evaluation/plot_time_vs_quality.py --out paper/figures/time_vs_quality.pdf
"""

import argparse

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

MODELS = [  # label, Table II column, colour, marker, GPU s/shower at batch size 1 and large (1400; CaloHadronic 256)
    (
        "AllShowers, NFE 32",
        1,
        "#c41e3a",
        "o",
        0.0399803,
        0.0044996,
    ),  # measured with this work's set-up
    (
        "AllShowers, NFE 4",
        2,
        "#eb6834",
        "o",
        0.0129824,
        0.00071598,
    ),  # measured with this work's set-up
    (
        "CaloHadronic, NFE 59",
        0,
        "#7b3fa0",
        "s",
        0.5194,
        0.2020,
    ),  # large batch = 256 for CaloHadronic; full chain measured with this work's set-up
    (r"This work, $d_s = 128$", 3, "#2a78d6", "D", 0.011, 0.00033693),
    (r"This work, $d_s = 256$, large LMF", 4, "#2e9e5b", "D", 0.011, 0.00035792),
]
# Geant4 at the hardware cost of one GPU: one CPU core costs its share of an EPYC 7402 plus the RAM one single-threaded
# Geant4 process needs (the GPU price already includes its memory). Host server, power and cooling not included.
GPU_PRICE = 12000  # $, A100 80 GB PCIe new, middle of $9,500-14,000 (JarvisLabs price guide, 2026-10-19)
CORE_PRICE = (
    1783 / 24
)  # $, EPYC 7402 list price / 24 cores (AnandTech Rome review, 2019)
RAM_PRICE = 9.17  # $/GB, DDR4 ECC server RAM median (DatacenterDisk index, 2026-10-19)
G4_RAM_GB = (
    8 * 8 / 24
)  # GB per core actually bought: one 8 GB RDIMM in each of the 8 memory channels of a 24-core EPYC
# 7402 (smallest standard population; it covers the 2.1 GB of a full-detector Geant4 job, ECCE computing plan,
# NIMA 1047 (2023) 167859: 2138-2275 MB)
G4_CORES = int(
    GPU_PRICE / (CORE_PRICE + G4_RAM_GB * RAM_PRICE)
)  # whole cores (rounded down)
G4_TIME = (
    2.09 / G4_CORES
)  # s per shower: one EPYC 7402 core (AllShowers paper) / cores at the price of one GPU


def quality(col, table="paper/tables/three_energies_metrics_withLargeDo01_summary.tex"):
    """Table II, JSD block, 'Mean over energies' row: the mean over 20 / 50 / 80 GeV of the geometric mean of the
    divergence over the observables, [x 10^-4], with its error. col = column index (CaloHadronic, AllShowers NFE 32,
    AllShowers NFE 4, d_s = 128, large LMF, Geant4)."""
    rows = [l for l in open(table) if l.startswith("Mean over energies")]
    cells = rows[1].split("&")[1:]  # second block = JSD
    v, e = (
        cells[col]
        .replace(r"\\", "")
        .replace(r"\textbf{", "")
        .replace("}", "")
        .split(r"$\pm$")
    )
    return float(v), float(e)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", required=True)
    args = p.parse_args()
    mpl.rcParams.update({"font.family": "sans-serif", "font.size": 14})
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for lab, src, col, mk, t1, tbig in MODELS:
        q, e = quality(src)
        t = t1  # filled marker: batch size 1
        ax.plot(
            [tbig * 1e3, t * 1e3], [q, q], "-", color=col, lw=1, alpha=0.6, zorder=2
        )
        ax.errorbar(
            t * 1e3, q, yerr=e, fmt=mk, color=col, ms=9, capsize=3, label=lab, zorder=3
        )
        ax.errorbar(
            tbig * 1e3,
            q,
            yerr=e,
            fmt=mk,
            color=col,
            mfc="white",
            mew=1.5,
            ms=9,
            capsize=3,
            zorder=3,
        )
        print(
            f"{lab:36s} JSD [1e-4] {q:6.2f} +- {e:.2f}   bs1 {t1 * 1e3:7.2f} ms, large bs {tbig * 1e3:.3f} ms"
        )
    q, e = quality(5)
    ax.errorbar(
        G4_TIME * 1e3,
        q,
        yerr=e,
        fmt="x",
        color="0.55",
        ms=9,
        mew=2,
        capsize=3,
        label=f"Geant4 vs Geant4 ({G4_CORES:.0f} CPU cores)",
        zorder=3,
    )
    ax.set_xscale("log")  # y linear
    ax.set_xlabel("Time per shower [ms]")
    ax.set_ylabel(r"JSD [$\times 10^{-4}$], mean over energies")
    ax.set_ylim(0, 62)  # headroom for the two-column legend above the points
    ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%g"))
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.grid(True, which="major", color="0.9")
    ax.annotate(
        "better",
        xy=(0.01, 0.01),
        xytext=(0.17, 0.20),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color="0.2"),
        fontsize=13,
        ha="center",
    )
    h, l = ax.get_legend_handles_labels()
    h += [
        mpl.lines.Line2D([], [], ls="", marker="o", color="0.3", ms=8),
        mpl.lines.Line2D(
            [], [], ls="", marker="o", color="0.3", mfc="white", mew=1.5, ms=8
        ),
    ]
    l += ["batch size 1", "batch size 1400 (CaloHadronic 256)"]
    ax.legend(
        h,
        l,
        fontsize=9.5,
        ncol=2,
        frameon=True,
        framealpha=0.9,
        edgecolor="0.8",
        loc="upper center",
    )  # inside, above the points
    fig.savefig(args.out, bbox_inches="tight")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
