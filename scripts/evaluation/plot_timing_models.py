"""Overlay the timing-vs-energy figure for several trained models.

timing.py plots one run at a time (one line per NFE / batch
size). This script takes one timing CSV per model and draws one line per model
instead, so latent sizes can be compared on the same axes, with the AllShowers
timings read from --allshowers-csv as references.

Usage:
    python scripts/evaluation/plot_timing_models.py \
        --csv "ds 256=files/timing/meanflow_nfe1_bs128_2attnL.csv" \
        --csv "ds 512=files/timing/meanflow_nfe1_bs128_2attnL_ds512.csv" \
        --csv "ds 1024=files/timing/meanflow_nfe1_bs128_2attnL_ds1024.csv" \
        --out files/timing/meanflow_nfe1_bs128_vs_energy_2attnL_ds.pdf
"""

import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# categorical slots for the model lines, assigned in fixed order;
# the order is CVD-validated on a white surface (adjacent pairs)
PALETTE = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
]
# AllShowers references: one grey per NFE, light -> dark, so they recede behind the model lines
ALLSHOWERS_STYLE = {4: ("#a8a69f", "o"), 8: ("#6e6c67", "D"), 32: ("#2f2e2b", "v")}
MIN_ENERGY = 10.0  # GeV: start of the training range; the point-count model extrapolates below it
INK, INK_SECONDARY, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
GRID_COLOR, AXIS_COLOR = "#e1e0d9", "#c3c2b7"
PLOT_RC = {
    "font.family": "sans-serif",
    # Helvetica Neue where installed; Nimbus Sans is the Helvetica clone on the cluster
    "font.sans-serif": [
        "Helvetica Neue",
        "Helvetica",
        "Nimbus Sans",
        "Arial",
        "DejaVu Sans",
    ],
    "font.size": 10,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.edgecolor": AXIS_COLOR,
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK_SECONDARY,
    "axes.labelsize": 10,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.titlecolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": AXIS_COLOR,
    "ytick.color": AXIS_COLOR,
    "xtick.labelcolor": INK_MUTED,
    "ytick.labelcolor": INK_MUTED,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.labelcolor": INK_SECONDARY,
}


def load_rows(csv_path, nfe, bs):
    """Timed rows for one (NFE, batch size) point, keyed by (energy, toy size).

    The toy size is 0 unless the CSV comes from a --toy-points sweep, where the
    requested shower size is the only key that means anything."""
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    out = {}
    for r in rows:
        if int(r["nfe"]) != nfe or int(r["batch_size"]) != bs:
            continue
        if r.get("points_scale") and float(r["points_scale"]) != 1.0:
            continue  # scaled-up showers of older CSVs (removed --points-scale option)
        toy = int(float(r["toy_points"])) if r.get("toy_points") else 0
        out[(float(r["energy_gev"]), toy)] = (
            float(r["time_per_sample_s"]) * 1e3,
            float(r["time_per_sample_std_s"]) * 1e3,
            float(r["mean_n_hits"]) if r.get("mean_n_hits") else None,
        )
    if not out:
        raise SystemExit(f"{csv_path}: no rows with nfe={nfe}, batch_size={bs}")
    return out, rows[0]["hardware"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--csv",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="one timing CSV per model, as 'legend label=path'; repeat per model",
    )
    p.add_argument(
        "--color",
        action="append",
        default=None,
        help="line colour per --csv, in the same order (default: the benchmark palette)",
    )
    p.add_argument("--nfe", type=int, default=1)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--model-label", default="CaloCHILL MeanFlow")
    p.add_argument(
        "--ymax",
        type=float,
        default=None,
        help="upper y limit [ms]; use it to zoom in when the model lines sit on top "
        "of each other on the scale set by the AllShowers references",
    )
    p.add_argument(
        "--yscale",
        default="linear",
        choices=["linear", "log"],
        help="log spreads out lines that sit on top of each other on a linear axis "
        "set by the slowest reference curve",
    )
    p.add_argument(
        "--extrapolate-to",
        type=float,
        default=0.0,
        help="extend each model with a dotted straight-line fit out to this "
        "incident energy [GeV], so the curves reach the AllShowers reference "
        "range; 0 disables it. Ignored on a vs-points axis",
    )
    p.add_argument(
        "--allshowers-csv",
        default=None,
        metavar="PATTERN",
        help="AllShowers reference timing CSVs (Euler solver), with {nfe} in place of the NFE, e.g. "
        "files/timing/allshowers_euler_nfe{nfe}_cuda_py311.csv; the rows at --bs are drawn for NFE 4, 8 and 32",
    )
    p.add_argument(
        "--legend-above",
        action="store_true",
        help="paper style: legend strip above the axes (models row, AllShowers row), no title",
    )
    p.add_argument(
        "--black-axes",
        action="store_true",
        help="axes, ticks and labels in black (paper)",
    )
    p.add_argument(
        "--inset-ymax",
        type=float,
        default=None,
        help="linear axis: add an inset with every line from 0 to this time [ms], "
        "to separate the model lines that sit at the bottom of the main axis",
    )
    p.add_argument(
        "--inset-xmax",
        type=float,
        default=None,
        help="inset upper energy [GeV] (default: as the main axis)",
    )
    p.add_argument(
        "--inset-xmin",
        type=float,
        default=None,
        help="inset lower energy [GeV] (default: as the main axis)",
    )
    p.add_argument(
        "--inset-ylog",
        type=float,
        default=None,
        metavar="YMIN",
        help="log y axis in the inset, from YMIN to --inset-ymax [ms]",
    )
    p.add_argument(
        "--hollow",
        type=int,
        nargs="+",
        default=[],
        help="--csv indices (0-based) drawn with open markers underneath the others",
    )  # Fig 7: d_s=128 visible
    p.add_argument(
        "--inset-bounds",
        type=float,
        nargs=4,
        default=[0.06, 0.52, 0.40, 0.43],
        metavar=("X0", "Y0", "W", "H"),
        help="inset position and size, in axes fractions",
    )
    p.add_argument("--out", required=True)
    args = p.parse_args()

    models = []
    hardware_label = None
    for spec in args.csv:
        if "=" not in spec:
            raise SystemExit(f"--csv expects 'label=path', got: {spec}")
        label, path = spec.rsplit("=", 1)  # the label may contain "=", a path does not
        by_energy, hardware = load_rows(path, args.nfe, args.bs)
        hardware_label = hardware_label or hardware
        models.append((label, by_energy))

    # a toy sweep has no shared energy axis: the shower size is the x axis and the key
    is_toy = all(any(k[1] for k in m[1]) for m in models)
    by_points = is_toy
    if is_toy:
        energies = sorted(
            set.intersection(*(set(k[1] for k in m[1] if k[1]) for m in models))
        )
        if not energies:
            raise SystemExit("the given toy CSVs share no shower size")
    else:
        # energies every model was timed at, so the lines are directly comparable
        energies = sorted(
            e
            for e in set.intersection(
                *(set(e for e, t in m[1] if not t) for m in models)
            )
            if e >= MIN_ENERGY
        )
        if not energies:
            raise SystemExit("the given CSVs share no timed energy")

    markers = ["o", "s", "^", "D", "v", "P", "X", "h"]
    series = []
    for idx, (label, by_key) in enumerate(models):
        if is_toy:
            keys = [k for n in energies for k in by_key if k[1] == n]
        else:
            keys = [(e, 0) for e in energies]
        entry = dict(
            x=[by_key[k][2] for k in keys] if by_points else energies,
            y=[by_key[k][0] for k in keys],
            yerr=[by_key[k][1] for k in keys],
            group="model",
            entry=label,
            color=args.color[idx] if args.color else PALETTE[idx % len(PALETTE)],
            marker=markers[idx % len(markers)],
            linestyle="-",
            zorder=3 if idx in args.hollow else 3.5,
            hollow=idx in args.hollow,
        )
        # Out to 90 GeV the pipeline time is linear in incident energy (the point
        # count is, and the decoder is linear in the point count), so a straight
        # line through the timed points is extended -- dotted, no markers -- to
        # say where the model would land on the reference's energy range. It is
        # an extrapolation, not a measurement: nothing above 90 GeV was timed.
        if not by_points and args.extrapolate_to > energies[-1] and len(energies) > 1:
            slope, icpt = np.polyfit(energies, entry["y"], 1)
            entry["x_fit"] = [energies[-1], args.extrapolate_to]
            print(
                f"fit {entry['entry']}: t [ms] = {icpt:.4f} + {slope * 1e3:.3f}e-3 E [GeV]"
            )
            entry["y_fit"] = [entry["y"][-1], icpt + slope * args.extrapolate_to]
        series.append(entry)

    for ref_nfe, (color, marker) in ALLSHOWERS_STYLE.items():
        if by_points or not args.allshowers_csv:
            break
        table = {}  # energy [GeV] -> (ms per shower, error) at --bs
        for r in csv.DictReader(open(args.allshowers_csv.format(nfe=ref_nfe))):
            if int(r["batch_size"]) == args.bs:
                table[float(r["energy_gev"])] = (
                    float(r["time_per_shower_s"]) * 1e3,
                    float(r["err_s"]) * 1e3,
                )
        ref_energies = sorted(e for e in table if e >= energies[0])
        if not ref_energies:
            continue
        series.append(
            dict(
                x=ref_energies,
                y=[table[e][0] for e in ref_energies],
                yerr=[table[e][1] for e in ref_energies],
                group="allshowers",
                entry=f"NFE {ref_nfe}",
                color=color,
                marker=marker,
                linestyle=(0, (4, 2)),
                zorder=2,
            )
        )

    rc = dict(PLOT_RC)
    if (
        args.black_axes
    ):  # axis lines, ticks, tick labels, axis titles and legend text in black
        rc.update(
            {
                k: "black"
                for k in (
                    "axes.edgecolor",
                    "axes.labelcolor",
                    "xtick.color",
                    "ytick.color",
                    "xtick.labelcolor",
                    "ytick.labelcolor",
                    "legend.labelcolor",
                )
            }
        )
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(7.5, 4.6))

        def draw(a):  # every line on axis a: the main axes, then the inset
            for s in series:
                container = a.errorbar(
                    s["x"],
                    s["y"],
                    yerr=s["yerr"],
                    color=s["color"],
                    marker=s["marker"],
                    linestyle=s["linestyle"],
                    linewidth=2,
                    markersize=6.5,
                    markerfacecolor="white" if s.get("hollow") else s["color"],
                    markeredgecolor=s["color"] if s.get("hollow") else "white",
                    markeredgewidth=1.4 if s.get("hollow") else 1.2,
                    elinewidth=1.2,
                    capsize=0,
                    zorder=s["zorder"],
                )
                if a is ax:
                    s["handle"] = container.lines[0]  # line + marker, no error bar
                if s.get("x_fit"):
                    a.plot(
                        s["x_fit"],
                        s["y_fit"],
                        color=s["color"],
                        linestyle=(0, (1, 2)),
                        linewidth=2,
                        zorder=s["zorder"],
                    )

        draw(ax)

        if by_points:
            ax.set_xlabel(
                "Number of points per shower"
                + (" (toy conditioning)" if is_toy else " (generated)")
            )
            ax.set_xlim(left=0)
            ax.xaxis.set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}")
            )
        else:
            ax.set_xlabel("Incident energy [GeV]")
            ax.set_xticks(list(energies) + [100, 110, 120])
        ax.set_ylabel("Time per shower [ms]")
        ax.set_yscale(args.yscale)
        # a log axis cannot start at 0; let matplotlib pick the lower bound there
        if args.yscale == "log":
            if args.ymax is not None:
                ax.set_ylim(top=args.ymax)
            # the sub-ms decade holds every model line, and by default carries no
            # labelled tick at all, so label the minor ticks as plain numbers
            ax.yaxis.set_minor_formatter(
                matplotlib.ticker.FuncFormatter(
                    lambda v, _: f"{v:g}" if v in (0.2, 0.3, 0.5, 2, 3, 5) else ""
                )
            )
            ax.yaxis.set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")
            )
            ax.tick_params(axis="y", which="minor", labelsize=8.5)
        else:
            ax.set_ylim(bottom=0, top=args.ymax)
            if args.inset_ymax:  # zoom on the bottom of the axis, all lines drawn again
                axin = ax.inset_axes(args.inset_bounds)
                draw(axin)
                x0, x1 = ax.get_xlim()
                if args.inset_xmax:
                    x1 = args.inset_xmax + 5  # same margin as the main axis
                if args.inset_xmin:
                    x0 = args.inset_xmin - 5
                axin.set_xlim(x0, x1)
                if (
                    args.inset_ylog
                ):  # log inset: plain-number labels, the range is under a decade
                    axin.set_yscale("log")
                    axin.set_yticks(
                        [0.1, 0.2, 0.3, 0.4, 0.6, 0.8],
                        ["0.1", "0.2", "0.3", "0.4", "0.6", "0.8"],
                    )
                    axin.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
                axin.set_ylim(args.inset_ylog or 0, args.inset_ymax)
                axin.set_xticks(
                    [t for t in ax.get_xticks() if x0 <= t <= x1][::2]
                )  # every second main tick: narrow axis
                axin.grid(True, axis="y", color=GRID_COLOR, linewidth=0.8)
                axin.set_axisbelow(True)
                axin.tick_params(length=2.5, width=0.8, labelsize=8.5)
                ax.indicate_inset_zoom(axin, edgecolor="0.4", linewidth=0.8)
        ax.grid(True, axis="y", which="both", color=GRID_COLOR, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=3, width=0.8)

        legend_kw = dict(
            frameon=False,
            loc="upper left",
            handlelength=3.2,
            handletextpad=0.7,
            labelspacing=0.5,
            borderaxespad=0.0,
            borderpad=0.0,
            alignment="left",
            fontsize=10,
            title_fontproperties={"weight": "bold", "size": 10.5},
        )
        y_anchor = 0.97
        model_title = f"{args.model_label} (NFE {args.nfe})"
        if args.legend_above:  # one row per group above the axes, filled row by row
            rows = [
                [
                    (
                        s["handle"],
                        s["entry"] if g == "model" else f"AllShowers {s['entry']}",
                    )
                    for s in series
                    if s["group"] == g
                ]
                for g in ("model", "allshowers")
            ]
            rows = [r for r in rows if r]
            n = max(len(r) for r in rows)
            blank = (plt.Line2D([], [], alpha=0), "")
            cells = [
                r[i] if i < len(r) else blank for i in range(n) for r in rows
            ]  # column-major
            ax.legend(
                *zip(*cells),
                ncol=n,
                loc="lower left",
                bbox_to_anchor=(0, 1.02, 1, 0.1),
                mode="expand",
                frameon=False,
                fontsize=10,
                handlelength=3.2,
                columnspacing=1.0,
            )
        for group, title in (
            ()
            if args.legend_above
            else (("model", model_title), ("allshowers", "AllShowers"))
        ):
            members = [s for s in series if s["group"] == group]
            if not members:
                continue
            leg = ax.legend(
                [s["handle"] for s in members],
                [s["entry"] for s in members],
                title=title,
                bbox_to_anchor=(0.03, y_anchor),
                **legend_kw,
            )
            leg.get_title().set_color(INK)
            ax.add_artist(leg)
            fig.canvas.draw()
            bbox = leg.get_window_extent().transformed(ax.transAxes.inverted())
            y_anchor = bbox.y0 - 0.05

        subtitle = f"{hardware_label}  ·  batch size {args.bs}"
        if is_toy:
            subtitle += "  ·  toy conditioning, exact point count per shower"
        if any(s.get("x_fit") for s in series):
            subtitle += (
                f"  ·  dotted: linear fit extrapolated to {args.extrapolate_to:g} GeV"
            )
        if not args.legend_above:  # the paper caption carries this instead
            ax.set_title(
                "Generation time vs. "
                + ("number of points" if by_points else "incident energy"),
                loc="left",
                pad=22,
            )
            ax.text(
                0,
                1.02,
                subtitle,
                transform=ax.transAxes,
                fontsize=9.5,
                color=INK_SECONDARY,
                va="bottom",
            )
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        fig.savefig(args.out, dpi=200, bbox_inches="tight")
        plt.close(fig)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
