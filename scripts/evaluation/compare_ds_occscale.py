"""Compare the latent sizes (ds) of the mean_flow LatentDiff runs on the occ-scale
features, all against the same Geant4 reference.

Reads only what occ-scale.py has already cached in files/projected_array/<run>/<tag>/
(no projection, no hdf5), so it runs in a minute or two on the login node:

    python scripts/evaluation/compare_ds_occscale.py            # default five ds, 50GeV_8000
    python scripts/evaluation/compare_ds_occscale.py --run "ds 64=<gen folder>" ...

Writes figs/occ-scale/ds_comparison/<tag>/metrics.csv: Wasserstein distance (normalised
by the Geant4 std) and Jensen-Shannon divergence JSD(Geant4, model) per observable and
ds, mean +- std over --splits disjoint subsets, plus the Geant4-vs-Geant4 value at the
same subset size as the statistical floor (paper tables: paper_outputs.py table-*).

COG is compared raw; N hits uses the hit count; the longitudinal and radial profiles are compared as energy-weighted hit
distributions, never as quantiles of the 78-bin mean profile.
"""

import argparse
import csv
import os
import sys

import matplotlib as mpl

mpl.use("Agg")
import numpy as np
from scipy.stats import wasserstein_distance

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# imported first: its rcParams are overridden below (velocity_energy_profile.py and
# paper_outputs.py rely on this module setting them)
from utils.plotting_features import plt_config  # noqa: E402, F401

mpl.rcParams["xtick.labelsize"] = 25
mpl.rcParams["ytick.labelsize"] = 25
mpl.rcParams["font.size"] = 35
mpl.rcParams["legend.frameon"] = False
mpl.rcParams["text.usetex"] = False
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Helvetica", "Nimbus Sans", "Arial", "DejaVu Sans"]
mpl.rcParams["figure.facecolor"] = "white"
mpl.rcParams["savefig.facecolor"] = "white"

PROJ = "files/projected_array/"
GEN = "pions_pions_HGx9_LatentDiff_SWD_zdim128_"
DEFAULT_RUNS = [
    "ds 64="
    + GEN
    + "ds64_2LastattnLayer_2026_09_16__15_08_50__mean_flow_2026_09_21__09_55_57__1",
    "ds 128="
    + GEN
    + "ds128_2LastattnLayer_2026_09_16__16_14_55__mean_flow_2026_09_22__10_40_48__1",
    "ds 256="
    + GEN
    + "ds256_KL1e-2_2LastattnLayer_2026_08_24__13_18_40__mean_flow_2026_09_11__18_19_41__1",
    "ds 512="
    + GEN
    + "ds512_KL1e-2_2LastattnLayer_2026_08_25__11_02_15__mean_flow_2026_09_14__14_20_26__1",
    "ds 1024="
    + GEN
    + "ds1024_KL1e-2_2LastattnLayer_2026_08_25__11_54_01__mean_flow_2026_09_14__14_29_07__1",
]
# ordinal blue ramp: light -> dark = small -> large ds
DS_COLORS = ["#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CALOH_COLOR = "#1baf7a"  # CaloHadronic, as in the paper's 50 GeV comparison figure
# the other models, same colours as the paper figures (paper_outputs.py fig-ds-profiles)
REF_COLORS = {
    "CaloHadronic": CALOH_COLOR,
    "AllShowers": "#eb6834",
    "AllShowers NFE 32": "#eb6834",
    "AllShowers NFE 8": "#c41e3a",
    "AllShowers NFE 4": "#7b3fa0",
    # large LMF (d512/L2), 566k checkpoints and the dropout-0.1 checkpoints of the paper
    "ds 128 big 566k": "#f2a0a8",
    "ds 256 big 566k": "#f5c27a",
    "ds 512 big 566k": "#b9a0d6",
    "ds 128 big do0.1 566k": "#8c564b",
    "ds 256 big do0.1 1199k": "#17becf",
    "ds 512 big do0.1 850k": "#bcbd22",
}


def colors_for(labels):
    """Each ds keeps its place on the blue ramp whichever ds are present; the other
    models (--run "CaloHadronic=...", "AllShowers=...") take REF_COLORS."""
    ds = [s.split("=", 1)[0] for s in DEFAULT_RUNS]
    return [
        DS_COLORS[ds.index(lab)] if lab in ds else REF_COLORS[lab] for lab in labels
    ]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--tag", default="50GeV_8000", help="cache sub-folder, e.g. 50GeV_8000"
    )
    p.add_argument(
        "--g4",
        default="Geant4/50GeV_g4win49.5-50.5_8000",
        help="Geant4 cache folder under files/projected_array/",
    )
    p.add_argument(
        "--run",
        action="append",
        help='"label=<generated folder>[@<tag>]", repeatable (the tag overrides --tag for '
        "that source); default: the five mean_flow NFE1 runs",
    )
    p.add_argument(
        "--splits", type=int, default=5, help="disjoint subsets for the error"
    )
    p.add_argument(
        "--kl-bins", type=int, default=25, help="Geant4-quantile bins for the JSD"
    )
    p.add_argument(
        "--out", default=None, help="default figs/occ-scale/ds_comparison/<tag>/"
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--keep-partial",
        action="store_true",
        help="keep an observable that only some sources have (e.g. the velocity with a no-time model); the others get no entry",
    )
    p.add_argument(
        "--radial-split",
        action="store_true",
        help="radial profile as two rows, ECAL and HCAL (each in its own cell units) instead of one mixed row",
    )
    return p.parse_args()


# ── loading ──────────────────────────────────────────────────────────────────────
def load_source(folder, real):
    """Features one source cached by occ-scale.py. Geant4 files carry an '_r'
    suffix (a few time/cog ones do not); generated files never do.
    A comma-separated list of folders is pooled: arrays concatenated along their last
    (shower / hit) axis, the 78-bin mean profile averaged (equal-size samples)."""
    if "," in folder:
        parts = [load_source(f, real) for f in folder.split(",")]
        pool = lambda k, a: (  # noqa: E731
            np.mean(a, axis=0) if k == "e_layers_list" else np.concatenate(a, axis=-1)
        )
        return {
            k: None if v is None else pool(k, [p[k] for p in parts])
            for k, v in parts[0].items()
        }

    def ld(name):
        for cand in [name + "_r", name] if real else [name]:
            f = os.path.join(folder, cand + ".npy")
            if os.path.exists(f):
                return np.load(f, allow_pickle=True)
        return None

    keys = [
        "e_sum_list",
        "e_sum_ecal_list",
        "e_sum_hcal_list",
        "occ_list",
        "start_layer_list",
        "cog_x",
        "cog_y",
        "cog_z",
        "hits_list",
        "e_radial",
        "e_radial_ecal",
        "e_radial_hcal",
        "e_layers_list",
        "X",
        "Y",
        "Z",
        "time_hits_list",
        "velocity",
    ]
    # e_layers / hits naming differs for Geant4 ("e_layers_list_r", "hits_list_r"),
    # which ld() already resolves through the suffix.
    return {k: ld(k) for k in keys}


# ── metrics ──────────────────────────────────────────────────────────────────────
def weighted_std(x, w):
    m = np.average(x, weights=w)
    return np.sqrt(np.average((x - m) ** 2, weights=w))


def weighted_quantiles(x, w, qs):
    o = np.argsort(x)
    cw = np.cumsum(w[o])
    cw /= cw[-1]
    return np.interp(qs, cw, x[o])


def js_div(a, b, wa, wb, edges):
    """Jensen-Shannon divergence (natural log, 0 .. ln 2): symmetric and finite
    without any epsilon, unlike KL, where one empty model bin dominated."""
    pa = np.histogram(a, edges, weights=wa)[0].astype(float)
    pb = np.histogram(b, edges, weights=wb)[0].astype(float)
    pa /= pa.sum()
    pb /= pb.sum()
    pm = 0.5 * (pa + pb)

    def kl(p):
        m = p > 0
        return np.sum(p[m] * np.log(p[m] / pm[m]))

    return float(0.5 * kl(pa) + 0.5 * kl(pb))


def split_idx(n, k, rng):
    return np.array_split(rng.permutation(n), k)


def compare(real, fake, wr, wf, k, nbins, rng):
    """WD / std(Geant4) and JSD(Geant4, model), mean +- std over k disjoint subsets
    of each sample. Bins are the Geant4 (weighted) quantiles of the full sample."""
    norm = weighted_std(real, wr) if wr is not None else np.std(real)
    qs = np.linspace(0, 1, nbins + 1)
    edges = (
        weighted_quantiles(real, wr, qs) if wr is not None else np.quantile(real, qs)
    )
    edges = np.unique(edges)
    edges[0], edges[-1] = -np.inf, np.inf
    ir, jf = split_idx(len(real), k, rng), split_idx(len(fake), k, rng)
    wd, jsd = [], []
    for a, b in zip(ir, jf):
        ua = None if wr is None else wr[a]
        ub = None if wf is None else wf[b]
        wd.append(wasserstein_distance(real[a], fake[b], ua, ub) / norm)
        jsd.append(js_div(real[a], fake[b], ua, ub, edges))
    return np.mean(wd), np.std(wd), np.mean(jsd), np.std(jsd)


def g4_floor(real, wr, k, nbins, rng):
    """Geant4 against itself at the same subset size as compare() (len/k per side):
    k pairs of disjoint subsets, each pair drawn afresh, so pairs may overlap across
    draws. Anything a model reaches at this level is statistically perfect."""
    m = len(real) // k
    perms = [rng.permutation(len(real)) for _ in range(k)]
    idx = [p[:m] for p in perms] + [p[m : 2 * m] for p in perms]
    norm = weighted_std(real, wr) if wr is not None else np.std(real)
    qs = np.linspace(0, 1, nbins + 1)
    edges = (
        weighted_quantiles(real, wr, qs) if wr is not None else np.quantile(real, qs)
    )
    edges = np.unique(edges)
    edges[0], edges[-1] = -np.inf, np.inf
    wd, jsd = [], []
    for a, b in zip(idx[:k], idx[k:]):
        ua = None if wr is None else wr[a]
        ub = None if wr is None else wr[b]
        wd.append(wasserstein_distance(real[a], real[b], ua, ub) / norm)
        jsd.append(js_div(real[a], real[b], ua, ub, edges))
    return np.mean(wd), np.std(wd), np.mean(jsd), np.std(jsd)


def observables(d, radial_split=False):
    """name -> (values, weights or None). Per-shower observables are unweighted;
    the profiles are energy-weighted hit distributions."""
    obs = {
        "Energy sum": (d["e_sum_list"], None),
        "ECAL energy sum": (d["e_sum_ecal_list"], None),
        "HCAL energy sum": (d["e_sum_hcal_list"], None),
        "N hits": (d["occ_list"], None),
        "Start layer": (d["start_layer_list"], None),
        "COG x": (d["cog_x"], None),
        "COG y": (d["cog_y"], None),
        "COG z": (d["cog_z"], None),
        "Cell energy (log10)": (np.log10(np.clip(d["hits_list"], 1e-6, None)), None),
        "Longitudinal profile": (d["Y"], d["hits_list"]),
        "Radial profile": (d["e_radial"][0], d["e_radial"][1]),
    }
    if radial_split:  # e_radial mixes ECAL (5 mm) and HCAL (30 mm) cell units
        del obs["Radial profile"]
        obs["Radial profile ECAL"] = (d["e_radial_ecal"][0], d["e_radial_ecal"][1])
        obs["Radial profile HCAL"] = (d["e_radial_hcal"][0], d["e_radial_hcal"][1])
    if d["time_hits_list"] is not None:
        t = d["time_hits_list"]
        obs["Hit time (log10)"] = (np.log10(t[t > 0]), None)
    if d["velocity"] is not None:  # gun-to-cell distance / cell time, per cell
        v = d["velocity"]
        obs["Velocity (log10)"] = (np.log10(v[v > 0]), None)
    return {
        k: (np.asarray(v, float), None if w is None else np.asarray(w, float))
        for k, (v, w) in obs.items()
    }


# ── outputs ──────────────────────────────────────────────────────────────────────
def write_tables(rows, labels, out):
    with open(out + "metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["observable", "source", "wd_norm", "wd_norm_err", "jsd", "jsd_err"])
        for r in rows:
            w.writerow([r["obs"], r["src"], *(f"{v:.5g}" for v in r["vals"])])

    obs_names = list(dict.fromkeys(r["obs"] for r in rows))
    get = {(r["obs"], r["src"]): r["vals"] for r in rows}
    cols = labels + ["Geant4 floor"]

    for i_mean, name in [(0, "WD / std(G4)"), (2, "JSD(G4, model)")]:
        print(f"\n{name}")
        print(f"{'observable':24s}" + "".join(f"{c:>14s}" for c in cols))
        for o in obs_names:
            vals = [get[(o, c)][i_mean] for c in labels]
            best = int(np.argmin(vals))
            cells = [
                f"{get[(o, c)][i_mean]:>13.3f}" + ("*" if j == best else " ")
                for j, c in enumerate(cols)
            ]
            print(f"{o:24s}" + "".join(cells))


def radial_mm(d):
    """Radial profile in mm: ECAL cells are 5 mm and HCAL cells 30 mm, so the
    cell-index radius is scaled per part."""
    return np.concatenate((d["e_radial_ecal"] * 5, d["e_radial_hcal"] * 30), axis=1)


def main():
    args = parse_args()
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
    out = args.out or f"figs/occ-scale/ds_comparison/{args.tag}/"
    out = out.rstrip("/") + "/"
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    real = load_source(",".join(PROJ + g for g in args.g4.split(",")), real=True)
    labels, fakes = [], []
    for spec in args.run or DEFAULT_RUNS:
        lab, folder = spec.split("=", 1)
        folder, _, tag = folder.partition("@")
        path = ",".join(PROJ + folder + "/" + t for t in (tag or args.tag).split(","))
        if not all(
            os.path.exists(os.path.join(p, "e_sum_list.npy")) for p in path.split(",")
        ):
            print(f"SKIP {lab}: no occ-scale cache at {path}")
            continue
        labels.append(lab)
        fakes.append(load_source(path, real=False))
        print(f"loaded {lab}: {len(fakes[-1]['e_sum_list'])} showers")
    if not fakes:
        sys.exit("no generated source found")
    print(f"Geant4: {len(real['e_sum_list'])} showers from {args.g4}")

    obs_r = observables(real, args.radial_split)
    rows = []
    for o, (vr, wr) in obs_r.items():
        for lab, f in zip(labels, fakes):
            obs_f = observables(f, args.radial_split)
            if o not in obs_f:
                continue
            vf, wf = obs_f[o]
            rows.append(
                {
                    "obs": o,
                    "src": lab,
                    "vals": compare(vr, vf, wr, wf, args.splits, args.kl_bins, rng),
                }
            )
        rows.append(
            {
                "obs": o,
                "src": "Geant4 floor",
                "vals": g4_floor(vr, wr, args.splits, args.kl_bins, rng),
            }
        )
    # an observable only some sources have (time) would leave holes in the table
    have = {
        o
        for o in obs_r
        if (any if args.keep_partial else all)(
            any(r["obs"] == o and r["src"] == lab for r in rows) for lab in labels
        )
    }
    rows = [r for r in rows if r["obs"] in have]

    write_tables(rows, labels, out)
    print("\noutputs in", out)


if __name__ == "__main__":
    main()
