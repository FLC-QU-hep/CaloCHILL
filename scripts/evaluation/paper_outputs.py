"""Paper figures and tables, one subcommand per former paper_*.py script:

    python scripts/evaluation/paper_outputs.py {table-ds-metrics,table-three-energies,table-timing,fig-ds-profiles} [options]

Each section below keeps the former script's docstring as _DOC_<COMMAND>."""

import argparse
import csv
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.use("Agg")
sys.path.insert(0, os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.compare_ds_occscale import (
    DEFAULT_RUNS,
    DS_COLORS,
    GEN,
    PROJ,
    load_source,
    radial_mm,
)  # noqa: E402
from evaluation.velocity_energy_profile import (
    CELL_LINES,
    YLABEL,
    load,
    plot_profile,
    series,
)  # noqa: E402
from utils.plotting_features import (
    legend_strip,
    plt_config,
    plt_hit_e,
    plt_radial_mm,
    plt_spinal,
)  # noqa: E402

# ---------------------------------------------------------------- table-ds-metrics
_DOC_TABLE_DS_METRICS = """Paper table of the latent-size comparison: normalised Wasserstein distance and
JSD(Geant4, model) per observable, from compare_ds_occscale.py's metrics.csv.

One table, two blocks (WD / sigma_G4 x 10^2, JSD x 10^4), one column per generated source
(ds N -> $d_s = N$, AllShowers / CaloHadronic -> the paper's macros) and the
Geant4-vs-Geant4 floor. The lowest model value per row is bold (all of them if they
tie at the printed precision), and each block ends with the geometric mean of the entries of every column.

    python scripts/evaluation/paper_outputs.py table-ds-metrics \
        --csv figs/occ-scale/ds_comparison/0GeV_100000/metrics.csv \
        --out paper/tables/ds_metrics.tex
"""


NAMES = {
    "ECAL energy sum": "ECal energy sum",
    "HCAL energy sum": "HCal energy sum",
    "N hits": "Number of hits",
    "Start layer": "Shower start layer",
    "COG x": r"Centre of gravity $x$",
    "COG y": r"Centre of gravity $y$",
    "COG z": r"Centre of gravity $z$",
    "Cell energy (log10)": "Cell energy",
    "Radial profile ECAL": "Radial profile ECal",
    "Radial profile HCAL": "Radial profile HCal",
    "Hit time (log10)": "Cell time",
    "Velocity (log10)": "Velocity",
    "Longitudinal velocity (log10)": "Longitudinal velocity",
    "Radial velocity (log10)": "Radial velocity",
}
# in metrics.csv but left out of the paper tables (mean rank is over the rest)
DROP = {"ECAL energy sum", "HCAL energy sum", "Start layer", "Hit time (log10)"}
FLOOR = "Geant4 floor"


def decimals(err):
    """Decimals so that the error does not print as zero (1 normally, up to 5)."""
    return next(d for d in range(1, 6) if err >= 0.5 * 10**-d or d == 5)


def geo_mean(vals, errs, n=1000):
    """vals, errs: (observable, ..., column). Geometric mean over the observables of
    the entries of every column, and its error: the standard deviation of that mean
    over n Gaussian resamplings of every entry within its statistical uncertainty."""
    vals, errs = np.clip(np.asarray(vals), 1e-12, None), np.asarray(errs)
    draws = np.clip(
        np.random.default_rng(0).normal(vals, errs, (n,) + vals.shape), 1e-12, None
    )
    # nanmean: a column without an observable (no-time model, no velocity) averages over the others
    return np.exp(np.nanmean(np.log(vals), axis=0)), np.exp(
        np.nanmean(np.log(draws), axis=1)
    ).std(axis=0)


MODEL_HEADERS = {
    "AllShowers NFE 32": r"\as NFE~32",
    "AllShowers NFE 4": r"\as NFE~4",
    "CaloHadronic": r"\cc",
    "ds 128 big do0.1 566k": r"$d_s = 128$, large \LMF",
    "ds 256 big do0.1 1199k": r"$d_s = 256$, large \LMF",
    "ds 512 big do0.1 850k": r"$d_s = 512$, large \LMF",
    "ds 256 no time": r"$d_s = 256$, no time",
}  # the paper's macros


def header(src):
    if src in MODEL_HEADERS:
        return MODEL_HEADERS[src]
    if src.startswith("ds "):
        return r"$d_s = %s$" % src.split()[-1]
    return src


BLOCKS = [  # (csv value, csv error, scale, block title)
    (
        "wd_norm",
        "wd_norm_err",
        1e2,
        r"$\mathrm{WD}/\sigma_{\textsc{Geant4}}$ [$\times 10^{-2}$]",
    ),
    ("jsd", "jsd_err", 1e4, r"JSD [$\times 10^{-4}$]"),
]


def main_table_ds_metrics():
    p = argparse.ArgumentParser(description=_DOC_TABLE_DS_METRICS.split("\n")[0])
    p.add_argument("--csv", required=True)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--keep",
        nargs="*",
        default=[],
        help="observables normally left out (DROP) to show anyway, e.g. 'ECAL energy sum'",
    )
    p.add_argument(
        "--side-labels",
        action="store_true",
        help="WD / JSD block titles as a rotated first column (table VII)",
    )
    args = p.parse_args()
    drop = DROP - set(args.keep)

    rows = list(csv.DictReader(open(args.csv)))
    srcs = list(dict.fromkeys(r["source"] for r in rows))
    ds = [s for s in srcs if s != FLOOR]
    obs = [o for o in dict.fromkeys(r["observable"] for r in rows) if o not in drop]
    get = {(r["observable"], r["source"]): r for r in rows}
    ncol = len(ds) + 1

    big = [
        s for s in ds if " big " in s
    ]  # large-LMF columns (contiguous, last): one spanning header instead of a wide label each
    lines = [r"\begin{tabular}{l" + "c" * ncol + "}", r"\toprule"]
    if big:
        i = ds.index(big[0]) + 2
        lines += [
            "& " * (i - 1) + r"\multicolumn{%d}{c}{large \LMF} & \\" % len(big),
            r"\cmidrule(lr){%d-%d}" % (i, i + len(big) - 1),
        ]
    lines += [
        " & "
        + " & ".join(header(s.split(" big")[0]) for s in ds)
        + r" & \textsc{Geant4} \\",
    ]
    summary = list(
        lines
    )  # main-text table: header + one geometric-mean row per block, as table-three-energies
    for key, ekey, scale, title in BLOCKS:
        lines += [
            r"\midrule",
            r"\multicolumn{%d}{l}{%s} \\" % (ncol + 1, title),
            r"\midrule",
        ]
        summary += lines[-3:]
        cols = ds + [FLOOR]
        val = (
            lambda o, s, k: float(get[(o, s)][k]) * scale if (o, s) in get else np.nan
        )  # missing: no time
        vals = np.array([[val(o, s, key) for s in cols] for o in obs])
        errs = np.array([[val(o, s, ekey) for s in cols] for o in obs])
        for o, vo in zip(obs, vals):
            best = min(round(v, 1) for v in vo[: len(ds)] if not np.isnan(v))
            cells = []
            for s in ds + [FLOOR]:
                v, e = val(o, s, key), val(o, s, ekey)
                if np.isnan(v):
                    cells.append("--")
                    continue
                d = decimals(e)
                c = r"$%.*f \pm %.*f$" % (d, v, d, e)
                if s != FLOOR and round(v, 1) == best:
                    c = r"$\mathbf{%.*f \pm %.*f}$" % (d, v, d, e)
                cells.append(c)
            lines.append(NAMES.get(o, o) + " & " + " & ".join(cells) + r" \\")
        gm, gme = geo_mean(vals, errs)  # over the observables, the floor column too
        partial = np.isnan(vals).any(
            axis=0
        )  # a column without some observables: its mean is not comparable
        gbest = gm[: len(ds)][~partial[: len(ds)]].min()
        gcells = [
            (
                r"$\mathbf{%.2f \pm %.2f}$"
                if i < len(ds) and g == gbest and not partial[i]
                else r"$%.2f \pm %.2f^\dagger$"
                if partial[i]
                else r"$%.2f \pm %.2f$"
            )
            % (g, e)
            for i, (g, e) in enumerate(zip(gm, gme))
        ]
        lines += [
            r"\cmidrule(l){2-%d}" % (len(cols) + 1),
            r"Geometric mean & " + " & ".join(gcells) + r" \\",
        ]
        summary.append(lines[-1])
    lines += [r"\bottomrule", r"\end{tabular}"]
    summary += [r"\bottomrule", r"\end{tabular}"]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    lines = side_labels(lines) if args.side_labels else lines
    for out, rows in [
        (args.out, lines),
        (args.out.replace(".tex", "_summary.tex"), summary),
    ]:
        with open(out, "w") as f:
            f.write(
                "%% generated by CaloCHILL/scripts/evaluation/paper_outputs.py table-ds-metrics\n"
                "%% from %s\n" % os.path.abspath(args.csv)
            )
            f.write("\n".join(rows) + "\n")
        print("wrote", out)
    vertical_summary(args.out.replace(".tex", "_summary.tex"))


# ---------------------------------------------------------------- table-three-energies
_DOC_TABLE_THREE_ENERGIES = """Paper table: WD / sigma_G4 (x 10^2) and JSD (x 10^4) per observable and model, each cell
'20 / 50 / 80 GeV' from three compare_ds_occscale.py metrics.csv; per energy the best model
is bold; one error per cell, the largest of the three. Columns in --sources order, then the Geant4 floor.

    python scripts/evaluation/paper_outputs.py table-three-energies --out table.tex \
        --csv figs/.../20GeV_10000_datasetG4/metrics.csv figs/.../50GeV_10000_datasetG4/metrics.csv \
              figs/.../80GeV_10000_datasetG4/metrics.csv
"""


def side_labels(lines):
    """Table VII layout (2026-10-13, --side-labels): each block title (the \multicolumn row between two
    \midrule) becomes a first column, rotated by 90 degrees and spanning the block's rows."""
    out, i = [], 0
    while i < len(lines):
        l = lines[i]
        if l.startswith(r"\begin{tabular}{l"):
            out.append(l.replace(r"\begin{tabular}{l", r"\begin{tabular}{ll", 1))
        elif l.startswith(r"\multicolumn"):
            title = l[l.index("}{l}{") + 5 : l.rindex("}")].split(", at 20")[0]
            i += 2  # skip the title and the \midrule under it
            rows = []
            while not lines[i].startswith((r"\midrule", r"\bottomrule")):
                rows.append(lines[i])
                i += 1
            n = sum(not r.startswith(r"\cmidrule") for r in rows)
            first = True
            for r in rows:
                if r.startswith(r"\cmidrule"):
                    a, b = r[r.index("{") + 1 : r.rindex("}")].split("-")
                    out.append(r"\cmidrule(l){%d-%d}" % (int(a) + 1, int(b) + 1))
                elif first:
                    out.append(
                        r"\multirow{%d}{*}{\rotatebox[origin=c]{90}{%s}} & "
                        % (n, title)
                        + r
                    )
                    first = False
                else:
                    out.append("& " + r)
            continue
        elif l.startswith(
            r"\cmidrule"
        ):  # header rule, e.g. over the large-LMF columns of table 1
            a, b = l[l.index("{") + 1 : l.rindex("}")].split("-")
            out.append(l[: l.index("{")] + "{%d-%d}" % (int(a) + 1, int(b) + 1))
        elif "&" in l and not l.startswith("\\"):  # header rows
            out.append("& " + l.lstrip())
        else:
            out.append(l)
        i += 1
    # the \midrule that opened each block title is kept; drop a doubled one
    return [
        l
        for k, l in enumerate(out)
        if not (l == r"\midrule" and k and out[k - 1] == r"\midrule")
    ]


def vertical_summary(path):
    """One-column version of a horizontal summary table (paper, 2026-10-15): models as rows, the WD and JSD blocks
    stacked. A cell 'a / b / c $\pm$ e' (one value per energy, one error: the largest) becomes columns 20 / 50 / 80 GeV
    and the error; the 'Mean over energies' row becomes the last column. Writes <path stem>_vertical.tex."""
    import re

    rows = [l.strip() for l in open(path) if l.strip() and not l.startswith("%")]
    cells = lambda l: [
        c.strip() for c in l.rstrip("\\").rstrip().rstrip("\\").split("&")
    ]
    names, group = None, None
    blocks = []  # (title, {row label: cells})
    for l in rows:
        m = re.match(r"\\multicolumn\{\d+\}\{l\}\{(.*)\} \\\\$", l)
        if m:
            blocks.append([m.group(1), {}])
        elif "multicolumn" in l and "{c}" in l:  # column group, e.g. large LMF
            c = cells(l)
            k = next(i for i, x in enumerate(c) if "multicolumn" in x)
            n, title = re.match(r"\\multicolumn\{(\d+)\}\{c\}\{(.*)\}", c[k]).groups()
            group = (k, int(n), title)
        elif names is None and l.startswith("&"):
            names = cells(l)[1:]
        elif blocks and "&" in l:
            c = cells(l)
            blocks[-1][1][c[0]] = c[1:]
    per_energy = any(" / " in v for _, b in blocks for v in b.get("Geometric mean", []))
    out = []
    if per_energy:
        out += [
            r"\begin{tabular}{lccccc}",
            r"\toprule",
            r" & 20\,GeV & 50\,GeV & 80\,GeV & $\pm$ & Mean \\",
        ]
    else:
        out += [
            r"\begin{tabular}{l" + "c" * len(blocks) + "}",
            r"\toprule",
            " & " + " & ".join(t for t, _ in blocks) + r" \\",
        ]

    def label(i):
        if group and group[0] - 1 <= i < group[0] - 1 + group[1]:
            return names[i] + ", " + group[2]
        return names[i] + (" floor" if "Geant4" in names[i] else "")

    if per_energy:
        for title, b in blocks:
            out += [
                r"\midrule",
                r"\multicolumn{6}{l}{%s} \\"
                % re.sub(r", at 20 / 50 / 80\\,GeV", "", title),
                r"\midrule",
            ]
            for i in range(len(names)):
                g, e = b["Geometric mean"][i].rsplit(r"$\pm$", 1)
                out.append(
                    label(i)
                    + " & "
                    + " & ".join(x.strip() for x in g.split(" / "))
                    + " & "
                    + e.strip()
                    + " & "
                    + b["Mean over energies"][i]
                    + r" \\"
                )
    else:
        out.append(r"\midrule")
        for i in range(len(names)):
            out.append(
                label(i)
                + " & "
                + " & ".join(b["Geometric mean"][i] for _, b in blocks)
                + r" \\"
            )
    out += [r"\bottomrule", r"\end{tabular}"]
    dst = path.replace(".tex", "_vertical.tex")
    with open(dst, "w") as f:
        f.write(
            "% generated by CaloCHILL/scripts/evaluation/paper_outputs.py: vertical_summary of "
            + os.path.basename(path)
            + "\n"
        )
        f.write("\n".join(out) + "\n")
    print("wrote", dst)


def main_table_three_energies():
    p = argparse.ArgumentParser(description=_DOC_TABLE_THREE_ENERGIES.split("\n")[0])
    p.add_argument(
        "--csv", nargs=3, required=True, help="metrics.csv at 20, 50, 80 GeV"
    )
    p.add_argument(
        "--sources",
        nargs="+",
        default=["CaloHadronic", "AllShowers NFE 32", "AllShowers NFE 4", "ds 128"],
    )
    p.add_argument("--out", required=True)
    args = p.parse_args()

    gets = [
        {(r["observable"], r["source"]): r for r in csv.DictReader(open(f))}
        for f in args.csv
    ]
    obs = [
        o
        for o in dict.fromkeys(o for o, _ in gets[0])
        if all((o, s) in g for g in gets for s in args.sources)
        and o != "Start layer"  # left out of the paper tables
    ]
    cols = args.sources + [FLOOR]

    lines = [
        r"\begin{tabular}{l" + "c" * len(cols) + "}",
        r"\toprule",
        " & " + " & ".join(header(s) for s in args.sources) + r" & \textsc{Geant4} \\",
    ]
    summary = list(lines)  # small results-section table: only the geometric mean rows
    for key, _, scale, title in BLOCKS:
        block = [
            r"\midrule",
            r"\multicolumn{%d}{l}{%s, at 20 / 50 / 80\,GeV} \\"
            % (len(cols) + 1, title),
            r"\midrule",
        ]
        lines += block
        summary += block
        for o in obs:
            v = [[float(g[(o, s)][key]) * scale for g in gets] for s in cols]
            best = [
                min(round(v[i][e], 1) for i in range(len(args.sources)))
                for e in range(3)
            ]
            err = [
                max(float(g[(o, s)][key + "_err"]) * scale for g in gets) for s in cols
            ]  # one error: the largest
            dec = [decimals(x) for x in err]
            cells = [
                " / ".join(
                    (
                        r"\textbf{%.*f}"
                        if i < len(args.sources) and round(x, 1) == best[e]
                        else "%.*f"
                    )
                    % (dec[i], x)
                    for e, x in enumerate(v[i])
                )
                + r" $\pm$ %.*f" % (dec[i], err[i])
                for i in range(len(cols))
            ]
            lines.append(NAMES.get(o, o) + " & " + " & ".join(cells) + r" \\")
        gm, gme = (
            geo_mean(  # (energy, column): geometric mean of the entries and its error, floor column too
                *(
                    [
                        [[float(g[(o, s)][k]) * scale for s in cols] for g in gets]
                        for o in obs
                    ]
                    for k in (key, key + "_err")
                )
            )
        )
        gbest = gm[:, : len(args.sources)].min(axis=1)  # best model, floor aside
        lines += [
            r"\cmidrule(l){2-%d}" % (len(cols) + 1),
            "Geometric mean & "
            + " & ".join(
                " / ".join(
                    (
                        r"\textbf{%.2f}"
                        if i < len(args.sources) and gm[e, i] == gbest[e]
                        else "%.2f"
                    )
                    % gm[e, i]
                    for e in range(3)
                )
                + r" $\pm$ %.2f" % gme[:, i].max()  # one error: the largest, as above
                for i in range(len(cols))
            )
            + r" \\",
        ]
        mean, merr = (
            gm.mean(axis=0),
            np.sqrt((gme**2).sum(axis=0)) / 3,
        )  # over the three energies
        mbest = mean[: len(args.sources)].min()
        lines.append(
            "Mean over energies & "
            + " & ".join(
                (
                    r"\textbf{%.2f} $\pm$ %.2f"
                    if i < len(args.sources) and mean[i] == mbest
                    else r"%.2f $\pm$ %.2f"
                )
                % (mean[i], merr[i])
                for i in range(len(cols))
            )
            + r" \\"
        )
        summary += lines[-2:]
    lines += [r"\bottomrule", r"\end{tabular}"]
    summary += [r"\bottomrule", r"\end{tabular}"]

    for out, rows in [
        (args.out, lines),
        (args.out.replace(".tex", "_summary.tex"), summary),
    ]:
        with open(out, "w") as f:
            f.write(
                "%% generated by CaloCHILL/scripts/evaluation/paper_outputs.py table-three-energies\n"
            )
            f.write("\n".join(rows) + "\n")
        print("wrote", out)
    vertical_summary(args.out.replace(".tex", "_summary.tex"))


# ---------------------------------------------------------------- table-timing
_DOC_TABLE_TIMING = """Paper timing table in the layout of the AllShowers paper's Table 2: Geant4,
CaloHadronic and AllShowers as published there, plus this work at d_s = 128 and 256.

Reference rows are copied from AllShowers Table 2 (Geant4 and CaloHadronic taken there
from the CaloHadronic paper). Our rows are the _summary.csv files written by
slurm_scripts/timing_table.sh: MeanFlow NFE 1, time per shower averaged over 10..90 GeV.
The speed factor is Geant4-on-CPU (2.09 s) over the time per shower, as in the reference.

    python scripts/evaluation/paper_outputs.py table-timing \
        --out paper/tables/timing_comparison.tex
"""


GEANT4_S = 2.09
TIMING = "files/timing/table_meanflow_nfe1_{dev}_{tag}_torch213_summary.csv"  # torch 2.13, as the AllShowers NFE 4 runs
OURS = [
    ("ds128", r"This work, $d_s = 128$"),  # short enough for one PRD column
    (
        "ds256big566k",
        r"\shortstack[l]{This work, $d_s = 256$,\\ large \LMF}",
    ),  # two lines: one PRD column
]
# large-batch row only, GPU: no-time VAE + large LMF with the points-only LayerFM
OURS_LARGE_ONLY = [
    (
        "ds256bigNOtime566k",
        r"\shortstack[l]{This work, $d_s = 256$,\\ large \LMF, no time}",
    ),
]
# batch sizes shown for this work; on CPU only batch size 1, as for AllShowers
OURS_BS = {"cpu": (1,), "cuda": (1, 16)}
# largest batch that fits on the 80 GB A100 at every energy; timed per energy
LARGE_BS_SIZE = 1400
LARGE_BS = (
    "files/timing/largebs/table_meanflow_nfe1_cuda_{tag}_bs1400_{e}GeV_torch213.csv"
)

# hardware -> [(model, [(nfe, [(batch size, s/shower, speed-up)])])], all as printed in
# the reference: its speed-ups were computed from unrounded times, so they are copied,
# not recomputed from the rounded times
REFERENCE = {
    "cpu": [
        (
            r"\cc",
            [
                (1, [(1, "0.59", "3.5"), (16, "0.73", "2.8")]),
                (59, [(1, "34.8", "<0.1"), (16, "43.3", "<0.1")]),
            ],
        ),
        # measured with this work's set-up (files/timing/allshowers_timing.py: 1 thread EPYC 7402,
        # bs 1, 20 warm-up + 30 timed per energy, mean 10..90 GeV, timer from conditioning to physical-unit points)
        (r"\as", [(4, [(1, "0.50", "4.2")]), (32, [(1, "3.8", "0.6")])]),
    ],
    "cuda": [
        (
            r"\cc",
            [
                (1, [(1, "0.0086", "243.0"), (16, "0.0033", "633.3")]),
                # bs 1/16: full chain measured with this work's set-up; the reference's 0.40/0.15 time the networks only
                (
                    59,
                    [(1, "0.52", "4.0"), (16, "0.27", "7.7"), (256, "0.20", "10.3")],
                ),  # measured with this work's set-up: fastest of the 16/64/256/1400 batch-size scan
            ],
        ),
        (
            r"\as",
            [
                # measured with this work's set-up (files/timing/allshowers_timing.py, A100 PCIe 80GB, Euler,
                # 20 warm-up + 30 timed): mean 10..90 GeV, bs 1 without the one glitch energy (err > 10%)
                (
                    4,
                    [
                        (1, "0.013", "161.0"),
                        (16, "0.0013", "1589.9"),
                        (1400, "0.00072", "2919.1"),
                    ],
                ),
                (
                    32,
                    [
                        (1, "0.040", "52.3"),
                        (16, "0.0052", "404.4"),
                        (1400, "0.0045", "464.5"),
                    ],
                ),  # bs 1400 as this work
            ],
        ),
    ],
}


def fmt_time(t):
    """Two significant figures, trailing zeros kept (0.0010, not 0.001)."""
    return f"{float(f'{t:.2g}'):#.2g}".rstrip(".")


def fmt_speed(t):
    f = GEANT4_S / t
    return r"$<0.1\times$" if f < 0.1 else rf"${f:.1f}\times$"


def large_bs(tag, root):
    """Large-batch time per shower, one CSV per energy, mean over 10..90 GeV as the other rows; None if one is missing."""
    big = [
        os.path.join(root, LARGE_BS.format(tag=tag, e=e)) for e in range(10, 100, 10)
    ]
    if all(os.path.exists(f) for f in big):
        ts = [float(next(csv.DictReader(open(f)))["time_per_sample_s"]) for f in big]
        return sum(ts) / len(ts)


def ours(dev, tag, root):
    path = os.path.join(root, TIMING.format(dev=dev, tag=tag))
    rows = list(csv.DictReader(open(path)))
    out = [
        (int(r["batch_size"]), float(r["avg_time_per_sample_s"]))
        for r in sorted(rows, key=lambda r: int(r["batch_size"]))
        if int(r["nfe"]) == 1 and int(r["batch_size"]) in OURS_BS[dev]
    ]
    if dev == "cuda" and (t := large_bs(tag, root)) is not None:
        out.append((LARGE_BS_SIZE, t))
    return out


def main_table_timing():
    p = argparse.ArgumentParser(description=_DOC_TABLE_TIMING.split("\n")[0])
    p.add_argument("--out", required=True)
    p.add_argument("--root", default=".")
    args = p.parse_args()

    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Hardware & Model & NFE & Batch & Time [s] & Speed-up \\",  # narrow enough for one PRD column
        r"\midrule",
    ]
    for dev, hw in (("cpu", "CPU"), ("cuda", "GPU")):
        models = list(REFERENCE[dev]) + [
            (
                lab,
                [
                    (
                        1,
                        [
                            (bs, fmt_time(t), None, t)
                            for bs, t in ours(dev, tag, args.root)
                        ],
                    )
                ],
            )
            for tag, lab in OURS
        ]
        if dev == "cuda":
            for tag, lab in OURS_LARGE_ONLY:
                if (t := large_bs(tag, args.root)) is not None:
                    models.append((lab, [(1, [(LARGE_BS_SIZE, fmt_time(t), None, t)])]))
        if dev == "cpu":
            models.insert(0, (r"\textsc{Geant4}", [("--", [(1, "2.09", "1.0")])]))
        for i, (model, by_nfe) in enumerate(models):
            if i:
                lines.append(r"\cmidrule(l){2-6}")
            for j, (nfe, rows) in enumerate(by_nfe):
                if j:
                    lines.append(r"\cmidrule(l){3-6}")
                for k, (bs, t_str, speed, *t) in enumerate(rows):
                    cells = [
                        hw if i == j == k == 0 else "",
                        model if j == k == 0 else "",
                        str(nfe) if k == 0 else "",
                        str(bs),
                        t_str,
                        rf"${speed}\times$" if speed else fmt_speed(t[0]),
                    ]
                    lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\midrule" if dev == "cpu" else r"\bottomrule")
    lines.append(r"\end{tabular}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(
            "% generated by CaloCHILL/scripts/evaluation/paper_outputs.py table-timing\n"
        )
        f.write("\n".join(lines) + "\n")
    print("wrote", args.out)


# ---------------------------------------------------------------- fig-ds-profiles
_DOC_FIG_DS_PROFILES = """Paper panels: spinal, radial and cell-energy profiles of Geant4 and a set of
generated sources, drawn by the same helpers as compare_ds_occscale.py but with no
legend inside any panel, plus one horizontal legend strip for the row. The projected
velocity panel is velocity_energy_profile.py's energy-per-shower profile (clamped
generated times) from its histogram caches, with the velocity axis cut to the
populated range; sources without hit times (AllShowers, CaloHadronic) are left out of
it. Sets (SETS below):
    ds_0GeV       the five ds on the full-range 100k samples
    ds_reco       20k of its Geant4 showers, encoded and decoded, against themselves (fig. 11)
    ds_0GeV_calibEfull_large_do01  calibrated ds plus the three large LMF (fig. 5)

    python scripts/evaluation/paper_outputs.py fig-ds-profiles --set ds_0GeV \
        --out paper/figures/ds_0GeV/
"""


VELOCITY_XLIM = (2e-2, 2e3)  # mm/ns: data span 0.03-900, HCal c_eff line at 1260
# paper fonts: x1.35 of the plotting_features defaults, so the 0.245-linewidth panels print like figure 5
FS_LABEL, FS_TICK = 33, 24
RATIO_LIMS = {
    "ds_reco": {"spinal": (0.95, 1.05), "velocity": (0.4, 1.6)},
    "ds_0GeV_calibEfull_large_do01": {
        "spinal": (0.9, 1.1),
        "velocity": (0.4, 1.6),
    },  # fig. 5
}  # default 0.6-1.4


# per set: Geant4 cache, (legend label, cache folder, colour, has hit times) per
# source, and whether to draw the velocity panel
SETS = {
    "ds_0GeV": (
        "Geant4/0GeV_100000",
        [
            (
                r"$d_s = %s$" % s.split("=")[0].split()[-1],
                s.split("=", 1)[1] + "/0GeV_100000",
                c,
                True,
            )
            for s, c in zip(DEFAULT_RUNS, DS_COLORS)
        ],
        True,
    ),
}
# ds_0GeV regenerated with the per-ds energy-dependent calibration a + b/E (run aliases *_calibEfull_)
SETS["ds_0GeV_calibEfull"] = (
    SETS["ds_0GeV"][0],
    [
        (lab, f.replace("__1/0GeV_100000", "_calibEfull__1/0GeV_100000"), c, t)
        for lab, f, c, t in SETS["ds_0GeV"][1]
    ],
    True,
)
# the same five ds on 20k Geant4 showers of the ds_0GeV reference, each encoded and
# decoded once (eval --real_file dataset/g4_reference_subset_20000.hdf5), against
# those same 20k showers
SETS["ds_reco"] = (
    "Geant4/0GeV_ddsim_20000",
    [
        (lab, folder.replace("/0GeV_100000", "/reco_0GeV_20000"), c, t)
        for lab, folder, c, t in SETS["ds_0GeV"][1]
    ],
    True,
)
# no-time VAE, orange as in the timing figure, right after ds 256; no velocity line
SETS["ds_reco"][1].insert(
    [lab for lab, _, _, _ in SETS["ds_reco"][1]].index(r"$d_s = 256$") + 1,
    (
        r"$d_s = 256$, no time",
        "pions_pions_HGx9_LatentDiff_SWD_zdim128_ds256_2LastattnLayer_NOtime2026_09_25__12_21_27__mean_flow_2026_10_05__12_58_38_eval1878560__1/reco_0GeV_20000",
        "#e07b00",
        False,
    ),
)
# figure 5: ds_0GeV_calibEfull plus the dropout-0.1 large LMF at ds 128/256/512, shades of green, best checkpoint per ds by geo mean of the 20/50/80 GeV scans
BIG_DO01 = {
    128: "ds128_2LastattnLayer_2026_09_16__16_14_55__mean_flow_2026_10_06__22_08_04_it566k_",
    256: "ds256_KL1e-2_2LastattnLayer_2026_08_24__13_18_40__mean_flow_2026_10_06__22_25_31_it1199k_",
    512: "ds512_KL1e-2_2LastattnLayer_2026_08_25__11_02_15__mean_flow_2026_10_05__11_42_41_it850k_",
}
SETS["ds_0GeV_calibEfull_large_do01"] = (
    SETS["ds_0GeV"][0],
    SETS["ds_0GeV_calibEfull"][1]
    + [
        (r"$d_s = %d$, large" % ds, GEN + f + "_1/0GeV_100000", c, True)
        for (ds, f), c in zip(BIG_DO01.items(), ["#7ccf8e", "#2e9e5b", "#135c33"])
    ],
    True,
)


def velocity_panel(g4, sources, out):
    ref = load(PROJ + g4)
    timed = [(folder, c) for _, folder, c, has_time in sources if has_time]
    fakes = [load(PROJ + folder) for folder, _ in timed]
    plot_profile(  # energy per shower
        ref["bins"],
        series(ref, "energy"),
        [series(f, "energy") for f in fakes],
        [c for _, c in timed],
        YLABEL["energy"],
        out + "velocity_energy.pdf",
        CELL_LINES,
        xlim=VELOCITY_XLIM,
    )


def main_fig_ds_profiles():
    p = argparse.ArgumentParser(description=_DOC_FIG_DS_PROFILES.split("\n")[0])
    p.add_argument("--set", default="ds_0GeV", choices=list(SETS))
    p.add_argument("--out", required=True)
    p.add_argument(
        "--zoom",
        action="store_true",
        help="insets: last ECal layers of the longitudinal profile, the two low-energy peaks of the cell energy",
    )
    p.add_argument(
        "--colors",
        nargs="+",
        default=None,
        help="override the colours of the generated sources, in order",
    )
    args = p.parse_args()
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
    out = args.out.rstrip("/") + "/"
    os.makedirs(out, exist_ok=True)

    g4, sources, with_velocity = SETS[args.set]
    real = load_source(PROJ + g4, real=True)
    labels = [lab for lab, _, _, _ in sources]
    colors = args.colors or [c for _, _, c, _ in sources]
    sources = [
        (lab, f, c, v) for (lab, f, _, v), c in zip(sources, colors)
    ]  # velocity panel uses them too
    fakes = [load_source(PROJ + folder, real=False) for _, folder, _, _ in sources]
    plt_config.color_lines = colors
    plot_labels = ["Geant4"] + labels
    shw = len(real["e_sum_list"])
    counts = [len(f["e_sum_list"]) for f in fakes]

    legend_strip(labels, colors, out + "legend.pdf")
    plt_config.font_labelx = plt_config.font_labely = FS_LABEL
    mpl.rcParams["xtick.labelsize"] = mpl.rcParams["ytick.labelsize"] = FS_TICK

    def ratio_lims(panel):
        plt_config.ratio_lims = RATIO_LIMS.get(args.set, {}).get(panel, (0.6, 1.4))

    axes_legend = mpl.axes.Axes.legend
    mpl.axes.Axes.legend = lambda self, *a, **k: None  # the strip replaces them
    try:
        ratio_lims("spinal")
        plt_spinal(
            real["e_layers_list"],
            [f["e_layers_list"] for f in fakes],
            shw,
            labels=plot_labels,
            gen_counts=counts,
            my_dir=out,
            zoom=dict(x=(20.5, 30.5), bounds=[0.42, 0.1, 0.5, 0.3])
            if args.zoom
            else None,
        )
        plt.close("all")
        ratio_lims("radial")
        plt_radial_mm(
            radial_mm(real),
            [radial_mm(f) for f in fakes],
            shw,
            labels=plot_labels,
            gen_counts=counts,
            my_dir=out,
        )
        plt.close("all")
        ratio_lims("hits")
        plt_hit_e(
            real["hits_list"],
            [f["hits_list"] for f in fakes],
            shw,
            labels=plot_labels,
            gen_counts=counts,
            my_dir=out,
            plt_config=plt_config,
            title=plt_config.title,
            zoom=dict(x=(0.08, 1.2), bounds=[0.16, 0.1, 0.48, 0.45])
            if args.zoom
            else None,
        )
        plt.close("all")
    finally:
        mpl.axes.Axes.legend = axes_legend
    if with_velocity:
        ratio_lims("velocity")
        velocity_panel(g4, sources, out)
    print("wrote", ", ".join(sorted(os.listdir(out))), "in", out)


COMMANDS = {
    "table-ds-metrics": main_table_ds_metrics,
    "table-three-energies": main_table_three_energies,
    "table-timing": main_table_timing,
    "fig-ds-profiles": main_fig_ds_profiles,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        sys.exit(
            f"usage: {sys.argv[0]} {{{','.join(COMMANDS)}}} [options]  (--help per command)"
        )
    COMMANDS[sys.argv.pop(1)]()
