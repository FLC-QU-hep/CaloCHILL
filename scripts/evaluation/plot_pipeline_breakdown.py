"""Pipeline breakdown vs shower size, from the CSVs written by
timing_CaloHadronicVAELatent.py --stages (or by the former pipeline_breakdown_sweep.py).

usage: plot_pipeline_breakdown.py <full.csv> <ref0.csv> <out.pdf> [--latent-label L] [--heading H]

<full.csv> is the decoder as trained, <ref0.csv> the 0-refinement-block ablation
(--n-refinement-blocks 0); the point-count line is read from --pc-csv (default
<full.csv>). Every point carries its 1.4826 * MAD / sqrt(repeats) error.

Styling is fixed for the paper: white surface, linear axes, milliseconds, and a
categorical palette checked against the three common colour-vision deficiencies
(all adjacent pairs dE >= 8 simulated, >= 15 unsimulated, every hue >= 3:1 against
white). Do not swap a hue without re-running that check.
"""

import argparse
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# blue / orange / aqua / amber -- validated, see module docstring
BLUE, ORANGE, AQUA, AMBER = "#256abf", "#d95926", "#199e70", "#a86e00"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"

parser = argparse.ArgumentParser()
parser.add_argument("full_csv")
parser.add_argument("ref0_csv")
parser.add_argument("out")
parser.add_argument("--latent-label", default="Latent mean flow, NFE=1")
parser.add_argument(
    "--heading", default=None, help="bold title, as the timing-vs-points panel"
)
parser.add_argument("--subheading", default=None, help="grey line under the heading")
parser.add_argument(
    "--pc-label",
    default="LayerFM (distilled)",
    help="legend label of the point-count stage",
)
parser.add_argument(
    "--pc-csv",
    default=None,
    help="CSV of the point-count line (default: full_csv), e.g. a "
    "pointcount_noretry_timing_v4_*.csv of the former sweep",
)
parser.add_argument(
    "--extra-latent",
    nargs=2,
    metavar=("CSV", "LABEL"),
    default=None,
    help="also draw the latent stage of another run (e.g. the large LMF), same hue, dashed",
)
args = parser.parse_args()

MS = 1e3  # seconds -> milliseconds


# timing_CaloHadronicVAELatent.py --stages columns -> the names of the former sweep's CSVs
ALIASES = {
    "time_per_sample_s": "total_s",
    "time_per_sample_std_s": "total_std_s",
    "pointcount_time_per_sample_s": "pointcount_s",
    "pointcount_time_per_sample_std_s": "pointcount_std_s",
    "latent_time_per_sample_s": "latentdiff_s",
    "latent_time_per_sample_std_s": "latentdiff_std_s",
    "decode_time_per_sample_s": "vaedecode_s",
    "decode_time_per_sample_std_s": "vaedecode_std_s",
}


def read(path):
    with open(path) as f:
        return [
            {
                ALIASES.get(k, k): float(v)
                for k, v in r.items()
                if k not in ("hardware", "model")
            }
            for r in csv.DictReader(f)
        ]


full = read(args.full_csv)
ref0 = read(args.ref0_csv)
pc = read(args.pc_csv or args.full_csv)
pc_key = "pointcount_noretry_s" if "pointcount_noretry_s" in pc[0] else "pointcount_s"
n_ref = int(full[0]["n_refinement_blocks"])
x = [r["mean_n_hits"] for r in full]
x0 = [r["mean_n_hits"] for r in ref0]


def col(rows, key):
    return (
        [r[key] * MS for r in rows],
        [r.get(key.replace("_s", "_std_s"), 0.0) * MS for r in rows],
    )


def blocks(n):
    return f"{n} last layer" + ("" if n == 1 else "s")


plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Helvetica Neue",
            "Helvetica",
            "Nimbus Sans",
            "Arial",
            "DejaVu Sans",
        ],
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.labelsize": 13,
        "legend.fontsize": 10.5,
    }
)

fig, ax = plt.subplots(figsize=(7.2, 4.6))
ax.set_axisbelow(True)
ax.grid(True, color=GRID, linewidth=0.7, alpha=1.0)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)

common = dict(
    linewidth=2.0,
    markersize=5.0,
    capsize=2.5,
    elinewidth=1.0,
    markeredgecolor="white",
    markeredgewidth=0.6,
)
series = (
    (
        x,
        col(full, "total_s"),
        dict(color=BLUE, marker="s", label=f"Total, {blocks(n_ref)}"),
    ),
    (
        x,
        col(full, "vaedecode_s"),
        dict(color=ORANGE, marker="o", label=f"VAE decoder, {blocks(n_ref)}"),
    ),
    (
        x0,
        col(ref0, "vaedecode_s"),
        dict(
            color=ORANGE, marker="v", linestyle=":", label="VAE decoder, 0 last layers"
        ),
    ),
    (
        x,
        col(full, "latentdiff_s"),
        dict(color=AQUA, marker="D", label=args.latent_label),
    ),
    (
        x,
        col(pc, pc_key),
        dict(color=AMBER, marker="^", linestyle="--", label=args.pc_label),
    ),
)
if args.extra_latent:
    extra = read(args.extra_latent[0])
    series = (
        series[:4]
        + (
            (
                [r["mean_n_hits"] for r in extra],
                col(extra, "latentdiff_s"),
                dict(
                    color=AQUA, marker="P", linestyle="--", label=args.extra_latent[1]
                ),
            ),
        )
        + series[4:]
    )
for xs, (y, yerr), kw in series:
    ax.errorbar(xs, y, yerr=yerr, **common, **kw)

ax.set_xlabel("Mean number of points per shower")
ax.set_ylabel("Time per shower [ms]")
ax.set_ylim(bottom=0)
ax.set_xlim(left=0)
ax.legend(loc="upper left", frameon=False, handlelength=2.4, labelspacing=0.5)
if args.heading:  # same look as plot_timing_models.py: bold heading, grey setup line, both left-aligned
    ax.text(
        0,
        1.10 if args.subheading else 1.04,
        args.heading,
        transform=ax.transAxes,
        fontsize=15,
        fontweight="bold",
        color=INK,
        ha="left",
        va="bottom",
    )
    if args.subheading:
        ax.text(
            0,
            1.035,
            args.subheading,
            transform=ax.transAxes,
            fontsize=10.5,
            color=MUTED,
            ha="left",
            va="bottom",
        )
fig.savefig(args.out, dpi=200, bbox_inches="tight", facecolor="white")
print("saved:", args.out)
