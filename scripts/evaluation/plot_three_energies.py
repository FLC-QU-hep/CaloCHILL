"""Geant4 vs models at three incident energies, one figure per energy and observable
(energy sum, longitudinal profile, radial profile): Geant4 filled or a dark line, one
colour per model, ratio to Geant4 underneath, one shared legend.pdf.

    python scripts/evaluation/plot_three_energies.py \
        --run 'ds 128=<run>_1' --run 'AllShowers NFE 32=AllShowers_mid32'
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from evaluation.compare_ds_occscale import (
    DEFAULT_RUNS,
    colors_for,
    load_source,
    radial_mm,
)  # noqa: E402
from utils.plotting_features import legend_strip, plt_config  # noqa: E402

PROJ = "./files/projected_array/"

p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
p.add_argument("--run", action="append", help="label=projected_array folder")
p.add_argument("--energies", type=int, nargs=3, default=[20, 50, 80])
p.add_argument(
    "--zoom",
    action="store_true",
    help="longitudinal: inset on the last ECal layers (21-30), linear y",
)
p.add_argument(
    "--zoom-pos",
    choices=["below", "above"],
    default="below",
    help="below: Geant4 as a dark line, inset in the white space under the profile; above: Geant4 filled, inset above it",
)
p.add_argument("--shw", type=int, default=10000)
p.add_argument("--out", default="figs/occ-scale/ds_comparison/three_energies/")
args = p.parse_args()
runs = [
    r.split("=", 1)
    for r in (args.run or [r for r in DEFAULT_RUNS if r.startswith("ds 128=")])
]
os.makedirs(args.out, exist_ok=True)

g4 = {
    E: load_source(
        PROJ + f"Geant4/{E}GeV_g4win{E - 0.5:g}-{E + 0.5:g}_{args.shw}", True
    )
    for E in args.energies
}
gen = {
    (lab, E): load_source(PROJ + f"{f}/{E}GeV_{args.shw}", False)
    for lab, f in runs
    for E in args.energies
}


def radial(d):
    r, e = radial_mm(d)
    return np.histogram(r, plt_config.bins_r_mm, weights=e)[0] / len(d["e_sum_list"])


lo, hi = np.percentile(
    np.concatenate([g4[E]["e_sum_list"] for E in args.energies]), [0.1, 99.9]
)
OBS = [  # name, x label, edges, per-source values, log y
    ("e_sum", "Visible energy [MeV]", np.linspace(0, 1.15 * hi, 60), None, False),
    ("longitudinal", "Layer", np.arange(0.5, 79.5), lambda d: d["e_layers_list"], True),
    ("radial", "Radius [mm]", plt_config.bins_r_mm, radial, True),
]


FIG5_COLORS = {
    "CaloHadronic": "#7b3fa0",  # purple, away from the greens of the large LMF
    "AllShowers NFE 32": "#c41e3a",
    "AllShowers NFE 4": "#eb6834",
    "ds 256 large": "#2e9e5b",  # large LMF ds 256 (566k), its fig. 4 / fig. 6 green
}  # paper fig. 5: purple / red / orange / green
model_c = [FIG5_COLORS.get(lab) or colors_for([lab])[0] for lab, _ in runs]
LIMITS = {
    "e_sum": ("x", (100, 2100)),
    "longitudinal": ("y", (0.5, 60)),
    "radial": ("y", (30, 3e4)),
}  # main panel, MeV
RATIO_LIM = {"e_sum": (0.0, 2.0), "longitudinal": (0.8, 1.2), "radial": (0.5, 1.5)}
FS_LABEL, FS_TICK = 24, 18

for name, xlabel, edges, prof, logy in OBS:
    vals = (
        (lambda d, ed: np.histogram(d["e_sum_list"], ed)[0].astype(float))
        if prof is None
        else (lambda d, ed: prof(d))
    )
    ylabel = "# showers" if prof is None else "Mean energy [MeV]"
    g4_kw = (
        dict(fill=True, alpha=0.35)
        if prof is None
        else (
            dict(fill=True, alpha=1.0) if args.zoom_pos == "above" else dict(lw=2.5)
        )  # profiles: filled (as fig. 5) or a dark line
    )
    for E in args.energies:  # one figure per energy: colour = model, ratio below
        fig, (ax, rx) = plt.subplots(
            2, 1, figsize=(6, 6), height_ratios=[3, 1], sharex=True
        )
        ed = edges
        if prof is None:  # bins on this energy's peak
            lo_e, hi_e = np.percentile(g4[E]["e_sum_list"], [0.5, 99.5])
            ed = np.linspace(lo_e - 0.2 * (hi_e - lo_e), hi_e + 0.2 * (hi_e - lo_e), 40)
        ref = vals(g4[E], ed)
        ref_r = np.where(
            ref > (0.02 * ref.max() if prof is None else 0), ref, np.nan
        )  # no ratio where Geant4 is (nearly) empty
        g4c = (
            "grey"
            if prof is None
            else ("lightgrey" if args.zoom_pos == "above" else "dimgrey")
        )
        ax.stairs(ref, ed, color=g4c, **g4_kw)
        for (lab, _), c in zip(runs, model_c):
            y = vals(gen[(lab, E)], ed)
            ax.stairs(y, ed, lw=2.5, color=c)
            rx.stairs(y / ref_r, ed, lw=2, color=c)
        if args.zoom and name == "longitudinal":  # last ECal layers, linear y
            INSET = [
                0.3,
                0.16,
                0.45,
                0.3,
            ]  # below the shower maximum, where the profile is highest
            ins = ax.inset_axes(
                INSET if args.zoom_pos == "below" else [0.12, 0.6, 0.45, 0.32]
            )  # clear of the frame
            ins.stairs(ref, ed, color=g4c, **g4_kw)
            zv = [ref[20:30]]
            for (lab, _), c in zip(runs, model_c):
                y = vals(gen[(lab, E)], ed)
                ins.stairs(y, ed, lw=2, color=c)
                zv.append(y[20:30])
            zv = np.concatenate(zv)
            ins.set_xlim(20.5, 30.5)
            ins.set_xticks([22, 24, 26, 28, 30])
            ins.set_ylim(zv.min() - 0.08 * np.ptp(zv), zv.max() + 0.08 * np.ptp(zv))
            ins.tick_params(labelsize=FS_TICK - 6)
            ax.indicate_inset_zoom(ins, edgecolor="0.3")
        if (
            args.zoom and name == "radial"
        ):  # first 3 bins (core), linear y, bottom left, under the profile
            ins = ax.inset_axes([0.17, 0.1, 0.38, 0.32])
            ins.stairs(ref, ed, color=g4c, **g4_kw)
            zv = [ref[:3]] + [vals(gen[(lab, E)], ed)[:3] for lab, _ in runs]
            for (lab, _), c in zip(runs, model_c):
                ins.stairs(vals(gen[(lab, E)], ed), ed, lw=2, color=c)
            zv = np.concatenate(zv)
            ins.set_xlim(ed[0], ed[3])
            ins.set_ylim(zv.min() - 0.08 * np.ptp(zv), zv.max() + 0.08 * np.ptp(zv))
            ins.tick_params(labelsize=FS_TICK - 6)
            ins.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            ins.yaxis.get_offset_text().set_fontsize(FS_TICK - 6)
            ax.indicate_inset_zoom(ins, edgecolor="0.3")
        rx.axhline(1, color="grey", lw=1)
        rx.set_ylim(*RATIO_LIM[name])
        rx.set_ylabel(f"{E} GeV\nratio", fontsize=FS_TICK)
        ax.set_yscale("log" if logy else "linear")
        ax.set_ylabel(ylabel, fontsize=FS_LABEL)
        rx.set_xlabel(xlabel, fontsize=FS_LABEL)
        for a in (ax, rx):
            a.tick_params(labelsize=FS_TICK)
        axis, lim = LIMITS[name]
        if args.zoom and name == "longitudinal":
            if (
                args.zoom_pos == "below"
            ):  # floor as high as possible (at most 0.3 MeV) with the profile above the inset
                x0, x1 = ax.get_xlim()
                l0, l1 = (
                    int(x0 + (x1 - x0) * f) for f in (INSET[0], INSET[0] + INSET[2])
                )  # layers under the inset
                m = ref[max(l0, 0) : l1 + 1].min() / 1.15
                t = INSET[1] + INSET[3] + 0.04  # inset top + margin, in axis fraction
                lim = (
                    min(0.3, 10 ** ((np.log10(m) - t * np.log10(lim[1])) / (1 - t))),
                    lim[1],
                )
            else:
                lim = (lim[0], 25 * lim[1])  # room above the profile for the inset
        if args.zoom and name == "radial":
            lim = (3, lim[1])  # room under the profile for the radial inset
        if axis == "y":
            ax.set_ylim(*lim)
        else:  # energy sum: x range around this energy's Geant4 peak
            a, b = np.percentile(g4[E]["e_sum_list"], [1, 99.9])
            ax.set_xlim(a - 0.15 * (b - a), b + 0.15 * (b - a))
        fig.align_ylabels()
        fig.savefig(args.out + f"{name}_{E}GeV.pdf", bbox_inches="tight")
        plt.close(fig)
        print("wrote", args.out + f"{name}_{E}GeV.pdf")

# the panels carry no legend: one strip for the row of three, as legend.pdf
legend_strip(
    [lab for lab, _ in runs],
    model_c,
    args.out + "legend.pdf",
    g4_patch=dict(color="grey", alpha=0.6),
    handlelength=None,
    columnspacing=None,
)
