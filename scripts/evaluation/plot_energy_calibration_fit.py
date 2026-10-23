"""Energy-dependent calibration fit: from samples generated with one global factor f0, the
factor each energy needs is f(E) = f0 / r(E), r = <E_sum gen> / <E_sum G4>, fitted with
f(E) = sum_i c_i E^p_i (powers --powers, default 0 -1 = a + b/E); plotted as the response r(E)
with the curve f0 / f(E), one panel per --run. f0 defaults to the run's calib_0GeV.txt.

    # full-range 100k sample in 2 GeV bins, one panel per ds, its own fit form
    python scripts/evaluation/plot_energy_calibration_fit.py --full-range \
        --run "ds 64=<run>_1" --powers "0 1 2" --run "ds 1024=<run>_1" --powers "0 -1 1" ...
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

ROOT = "./files/"
PROJ = ROOT + "projected_array/"
p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
p.add_argument(
    "--run", action="append", required=True, help="label=projected_array folder"
)
p.add_argument(
    "--powers", action="append", default=None, help="per --run, e.g. '0 1 2'"
)
p.add_argument(
    "--f0", type=float, default=None, help="global factor (default: calib_0GeV.txt)"
)
p.add_argument("--full-range", action="store_true", help="0GeV_100000 in 2 GeV bins")
p.add_argument(
    "--vertical",
    action="store_true",
    help="one row per latent size (standard left, large right)",
)
p.add_argument(
    "--large-run",
    action="append",
    default=[],
    help="second row: 'label=generated_showers folder' of an UNCALIBRATED 10k full-range sample "
    "(direct fit f = 1/r, a + bE + cE^2), drawn under the --run with the same label",
)
p.add_argument(
    "--large-coef",
    action="append",
    default=[],
    help="per --large-run, the coefficients 'a b c' actually used (fit_poly_calib.py); default: refit here",
)
p.add_argument(
    "--out", default="figs/occ-scale/ds_comparison/multi_energy/calibration_fit.pdf"
)
args = p.parse_args()
runs = [r.rsplit("=", 1) for r in args.run]  # labels may contain "="
powers = [[float(x) for x in s.split()] for s in (args.powers or ["0 -1"] * len(runs))]


def ratio(a, b):
    r = a.mean() / b.mean()
    return r, r * np.hypot(
        a.std() / a.mean() / len(a) ** 0.5, b.std() / b.mean() / len(b) ** 0.5
    )


def response(folder):
    """(E, r, dr) of one run per 2 GeV bin of the full range (0GeV_100000)."""
    ld = lambda d, n: np.load(f"{PROJ}{d}/0GeV_100000/{n}.npy").ravel()  # noqa: E731
    ge, gi, me, mi = (
        ld("Geant4", "e_sum_list_r"),
        ld("Geant4", "inc_en"),
        ld(folder, "e_sum_list"),
        ld(folder, "inc_en"),
    )
    edges = np.arange(10, 92, 2.0)
    rr = [
        ratio(me[(mi >= lo) & (mi < hi)], ge[(gi >= lo) & (gi < hi)])
        for lo, hi in zip(edges[:-1], edges[1:])
    ]
    return (0.5 * (edges[1:] + edges[:-1]), *np.array(rr).T)


def response_plain(folder):
    """(E, r, dr) in 2 GeV bins from a plain generated hdf5 (deposited energy, as fit_poly_calib.py)."""
    import glob
    import sys

    sys.path.insert(0, os.path.dirname(__file__))
    import fit_poly_calib as fp

    real, e_real = fp.load_real_reference(sorted(glob.glob(fp.REAL_GLOB))[:5])
    sh, e_gen = fp.load_from_hdf5(
        f"{ROOT}generated_showers/{folder}/showers_0GeV_10000.hdf5"
    )
    gen = fp.total_deposited_energy(sh)
    edges = np.arange(10, 92, 2.0)
    rr = [
        ratio(gen[(e_gen >= lo) & (e_gen < hi)], real[(e_real >= lo) & (e_real < hi)])
        for lo, hi in zip(edges[:-1], edges[1:])
    ]
    return (0.5 * (edges[1:] + edges[:-1]), *np.array(rr).T)


large = dict(r.rsplit("=", 1) for r in args.large_run)
nrow = 2 if large else 1
fig, axs = plt.subplots(
    *((len(runs), nrow) if args.vertical else (nrow, len(runs))),
    figsize=(3.2 * nrow + 1, 3.1 * len(runs))
    if args.vertical
    else (4.2 * len(runs) + 1, 4.5 * nrow),
    sharey=True,
    squeeze=False,
)
if args.vertical:
    axs = axs.T  # axs[row][i] keeps meaning (row = standard / large, i = latent size)
panels = [
    (
        axs[0][i],
        lab,
        folder,
        pw,
        False,
        None,
    )  # (..., plain = uncalibrated large-LMF sample, coefficients used)
    for i, ((lab, folder), pw) in enumerate(zip(runs, powers))
]
for i, (lab, _) in enumerate(runs):
    if nrow == 2 and lab in large:
        k = list(large).index(lab)
        coef = (
            [float(x) for x in args.large_coef[k].split()]
            if k < len(args.large_coef)
            else None
        )
        panels.append((axs[1][i], lab, large[lab], [0.0, 1.0, 2.0], True, coef))
    elif nrow == 2:
        axs[1][i].set_axis_off()
for ax, lab, folder, pw, plain, used in panels:
    f0 = (
        1.0
        if plain
        else (
            args.f0
            or float(open(f"{ROOT}generated_showers/{folder}/calib_0GeV.txt").read())
        )
    )
    E, r, dr = response_plain(folder) if plain else response(folder)
    f, df = f0 / r, f0 / r * dr / r
    X = np.array([E**k for k in pw]).T
    c, *_ = np.linalg.lstsq(
        X / df[:, None], f / df, rcond=None
    )  # weighted least squares
    if used is not None:  # draw the factor actually applied, not a refit
        c = np.array(used)
    fit = lambda e: sum(ci * e**k for ci, k in zip(c, pw))  # noqa: E731
    x = np.linspace(E.min() * 0.95, E.max() * 1.02, 200)
    form = " + ".join(
        {0: "a", -1: "b/E", 1: "bE" if 2 in pw else "cE", 2: "cE^2"}.get(
            int(k), f"E^{k:g}"
        )
        for k in pw
    )
    ax.errorbar(
        E,
        r,
        dr,
        fmt="o",
        color="#2a78d6",
        ms=4 if args.full_range else 7,
        capsize=2,
        label="uncalibrated" if plain else rf"generated at $f_0 = {f0:.4f}$",
    )
    ax.plot(
        x,
        f0 / fit(x),
        color="#eb6834",
        lw=2,
        label=rf"fit: $1\,/\,({form})$" if plain else rf"fit: $f_0\,/\,({form})$",
    )
    ax.axhline(1, color="grey", lw=1)
    ax.set_title(
        lab
        + (
            r", large \textsc{lmf}"
            if plain and plt.rcParams["text.usetex"]
            else ", large LMF"
            if plain
            else ""
        ),
        fontsize=15,
    )
    ax.set_xlabel("Incident energy [GeV]", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.legend(frameon=False, fontsize=10, loc="best")
    print(
        f"{lab}: f0 {f0:.4f}, coef {' '.join(f'{v:.6g}' for v in c)} (powers {pw}), "
        f"max |residual| {abs(f - X @ c).max() * 100:.2f}%"
    )
for row in (
    axs.T if args.vertical else axs
):  # y label on the first panel of every figure row
    for a in row:
        if a.axison:
            a.set_ylabel(
                "E ratio",
                fontsize=15,
            )
            a.tick_params(labelleft=True)
            break
fig.tight_layout(h_pad=2.0)
os.makedirs(os.path.dirname(args.out), exist_ok=True)
fig.savefig(args.out, bbox_inches="tight")
print("wrote", args.out)
