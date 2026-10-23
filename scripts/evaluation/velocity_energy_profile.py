"""Velocity and velocity-energy profile of the projected occ-scale showers.

The velocity profile is to velocity what the spinal/radial profiles are to depth and
radius: the energy deposited per shower in each velocity bin. It needs every cell's
energy and time together, which the cached occ-scale feature arrays do not keep, so
the projected events hdf5 are streamed (both grids, gzip) in parallel.

Cells and velocities are exactly occ-scale.py's: velocity = gun-to-cell distance /
cell time for 0 < t <= 200 ns; the energy profile additionally keeps cells with
E >= 1e-2 MeV (the occ-scale energy threshold). Generated times are clamped on read to
the combined floor (Geant4 per-cell minimum or distance/c_eff, whichever is later),
as occ-scale.py --clip-time-to-geant4 does; Geant4 is never clamped.

    # one histogram cache per source; paper_outputs.py fig-ds-profiles draws the velocity panel from them
    python scripts/evaluation/velocity_energy_profile.py compute \
        --src files/projected_array/Geant4/0GeV_100000 --workers 16
    python scripts/evaluation/velocity_energy_profile.py compute --clamp \
        --src files/projected_array/<run>/0GeV_100000 --workers 16
"""

import argparse
import glob
import os
import sys
from multiprocessing import Pool

import h5py
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.plotting_features import (  # noqa: E402
    c_eff,
    cell_distance_grid,
    combined_time_floor,
    plt_config,
    ratio_plots,
)

C = 299.792458  # mm/ns
THR_E = 1e-2  # MeV, occ-scale.py threshold
THR_T = 200  # ns, occ-scale.py threshold_time
G4_FLOOR = "files/projected_array/Geant4/time_floor.npy"
CACHE = "velocity_energy_hist.npz"
BINS = plt_config.velocity_bins


# ── stage 1: histograms ──────────────────────────────────────────────────────────
_w = {}


def _init(path, clamp):
    _w["f"] = h5py.File(path, "r")
    dist = cell_distance_grid()
    _w["dist"] = dist.ravel()
    _w["ecal"] = np.broadcast_to(
        (np.arange(dist.shape[1]) < 30)[None, :, None], dist.shape
    ).ravel()
    floor = None
    if clamp:
        g4 = np.load(G4_FLOOR) if os.path.exists(G4_FLOOR) else None
        floor = combined_time_floor(g4).ravel()
    _w["floor"] = floor


def _chunk(rng):
    """counts / energy / energy^2 per velocity bin, [ECAL, HCAL] rows."""
    nb = len(BINS) - 1
    cnt, esum, esq = np.zeros((2, nb)), np.zeros((2, nb)), np.zeros((2, nb))
    ds_e, ds_t = _w["f"]["projected_showers"], _w["f"]["projected_time_showers"]
    for j in range(*rng):
        T = ds_t[j].ravel()
        hit = np.flatnonzero(T > 0)
        t = T[hit].astype(np.float64)
        if _w["floor"] is not None:
            t = np.fmax(t, _w["floor"][hit])  # fmax: a NaN floor leaves t as is
        keep = t <= THR_T
        idx, t = hit[keep], t[keep]
        v = _w["dist"][idx] / t
        part = np.where(_w["ecal"][idx], 0, 1)
        e = ds_e[j].ravel()[idx].astype(np.float64)
        e = np.where(e >= THR_E, e, 0.0)
        for p in (0, 1):
            sel = part == p
            cnt[p] += np.histogram(v[sel], BINS)[0]
            esum[p] += np.histogram(v[sel], BINS, weights=e[sel])[0]
            esq[p] += np.histogram(v[sel], BINS, weights=e[sel] ** 2)[0]
    return cnt, esum, esq


def compute(src, clamp, workers, n=None):
    path = sorted(glob.glob(os.path.join(src, "events_*.hdf5")))[0]
    with h5py.File(path, "r") as f:
        n_tot = f["projected_showers"].shape[0]
    n = min(n or n_tot, n_tot)
    step = 100
    chunks = [(s, min(s + step, n)) for s in range(0, n, step)]
    cnt = esum = esq = 0
    with Pool(workers, initializer=_init, initargs=(path, clamp)) as pool:
        for k, (c, s, q) in enumerate(pool.imap_unordered(_chunk, chunks)):
            cnt, esum, esq = cnt + c, esum + s, esq + q
            if k % 100 == 0:
                print(f"{k}/{len(chunks)} chunks", flush=True)
    out = os.path.join(src, CACHE)
    np.savez(out, bins=BINS, counts=cnt, e_sum=esum, e_sq=esq, n_showers=n, clamp=clamp)
    n_unphys = (
        cnt[0][BINS[:-1] >= c_eff("ecal")].sum()
        + cnt[1][BINS[:-1] >= c_eff("hcal")].sum()
    )
    print(
        f"wrote {out}: {n} showers, {int(cnt.sum())} cells, clamp={clamp}, "
        f"cells in bins above c_eff: {int(n_unphys)}"
    )


# ── longitudinal / radial velocity, energy weighted (Table 1 rows) ──────────────────
# v_L = (y_layer - y_gun) / t, v_R = rho / t with rho the distance of the cell centre from the beam axis
# (x = z = 0). Per source: energy-weighted histograms of log10 v in K random cell subsets (as compare_ds_occscale
# splits per-cell observables), plus for Geant4 K pairs of disjoint subsets for its floor; the WD and the JSD are
# computed from these fine histograms (VC_BINS), so no per-cell arrays are stored.
VC_BINS = np.linspace(-4, 4, 3001)
VC_CACHE = "velocity_components_hist.npz"
K = 5


def _vc_grids(s=180, n_layers=78):
    from utils.plotting_features import GUN_POSITION, LAYER_BOTTOM_POS, LAYER_TOP_POS

    depth = (np.load(LAYER_BOTTOM_POS) + np.load(LAYER_TOP_POS)) / 2 - GUN_POSITION[1]
    ce = np.arange(-450, 450 + 5, 5)[:s] + 2.5  # ECAL cell centres
    ch = np.full(s, np.nan)
    ch[75:105] = np.arange(-450, 450 + 30, 30)[:30] + 15  # HCAL cell centres
    rho = np.empty((s, n_layers, s))
    for ll in range(n_layers):
        c = ce if ll < 30 else ch
        rho[:, ll, :] = np.sqrt(c[:, None] ** 2 + c[None, :] ** 2)
    dep = np.broadcast_to(depth[None, :, None], rho.shape)
    return dep.ravel(), rho.ravel()


def _vc_chunk(rng_seed_range):
    seed, (a, b) = rng_seed_range
    rng = np.random.default_rng(seed)
    nb = len(VC_BINS) - 1
    sub = np.zeros((K, 2, nb))
    fa, fb = np.zeros((K, 2, nb)), np.zeros((K, 2, nb))
    ds_e, ds_t = _w["f"]["projected_showers"], _w["f"]["projected_time_showers"]
    for j in range(a, b):
        T = ds_t[j].ravel()
        Ev = ds_e[j].ravel()
        hit = np.flatnonzero((T > 0) & (Ev >= THR_E))
        t = T[hit].astype(np.float64)
        if _w["floor"] is not None:
            t = np.fmax(t, _w["floor"][hit])
        keep = t <= THR_T
        idx, t = hit[keep], t[keep]
        e = Ev[idx].astype(np.float64)
        lv = [np.log10(_w["dep"][idx] / t), np.log10(_w["rho"][idx] / t)]
        u = rng.integers(0, K, len(idx))
        for c in (0, 1):
            for k in range(K):
                m = u == k
                sub[k, c] += np.histogram(lv[c][m], VC_BINS, weights=e[m])[0]
        if _w["g4"]:  # floor: K pairs of disjoint subsets, each pair drawn afresh
            for k in range(K):
                v = rng.integers(0, K, len(idx))
                for c in (0, 1):
                    fa[k, c] += np.histogram(lv[c][v == 0], VC_BINS, weights=e[v == 0])[
                        0
                    ]
                    fb[k, c] += np.histogram(lv[c][v == 1], VC_BINS, weights=e[v == 1])[
                        0
                    ]
    return sub, fa, fb


def compute_components(src, clamp, workers, n=None):
    path = sorted(glob.glob(os.path.join(src, "events_*.hdf5")))[0]
    with h5py.File(path, "r") as f:
        n_tot = f["projected_showers"].shape[0]
    n = min(n or n_tot, n_tot)
    chunks = [(i, (s0, min(s0 + 100, n))) for i, s0 in enumerate(range(0, n, 100))]
    g4 = not clamp

    def init(path, clamp):
        _init(path, clamp)
        _w["dep"], _w["rho"] = _vc_grids()
        _w["g4"] = g4

    sub = fa = fb = 0
    with Pool(workers, initializer=init, initargs=(path, clamp)) as pool:
        for k, (s_, a_, b_) in enumerate(pool.imap_unordered(_vc_chunk, chunks)):
            sub, fa, fb = sub + s_, fa + a_, fb + b_
            if k % 100 == 0:
                print(f"{k}/{len(chunks)} chunks", flush=True)
    out = os.path.join(src, VC_CACHE)
    np.savez(
        out, bins=VC_BINS, sub=sub, floor_a=fa, floor_b=fb, n_showers=n, clamp=clamp
    )
    print(f"wrote {out}: {n} showers")


def _wd_js(ha, hb, centres, norm, jedges):
    pa, pb = ha / ha.sum(), hb / hb.sum()
    wd = (
        np.sum(np.abs(np.cumsum(pa) - np.cumsum(pb))) * (centres[1] - centres[0]) / norm
    )
    ja = np.array([pa[i:j].sum() for i, j in zip(jedges[:-1], jedges[1:])])
    jb = np.array([pb[i:j].sum() for i, j in zip(jedges[:-1], jedges[1:])])
    pm = 0.5 * (ja + jb)
    kl = lambda p: np.sum(p[p > 0] * np.log(p[p > 0] / pm[p > 0]))
    return wd, 0.5 * kl(ja) + 0.5 * kl(jb)


def component_rows(g4_src, runs, nbins=25):
    """CSV rows (observable, source, wd_norm, wd_norm_err, jsd, jsd_err) as compare_ds_occscale writes them."""
    g = dict(np.load(os.path.join(g4_src, VC_CACHE)))
    centres = 0.5 * (VC_BINS[1:] + VC_BINS[:-1])
    rows = []
    for c, name in (
        (0, "Longitudinal velocity (log10)"),
        (1, "Radial velocity (log10)"),
    ):
        full = g["sub"][:, c].sum(0)
        p = full / full.sum()
        mu = np.sum(p * centres)
        norm = np.sqrt(np.sum(p * (centres - mu) ** 2))
        cdf = np.cumsum(p)
        if c == 0:
            # v_L: Geant4 is a needle at the speed of light, so equal-energy bins would all fall inside it and the
            # JSD would count tiny time shifts as large ones; use nbins equal-width bins in log10 v over the range
            # holding 99.8% of the Geant4 energy instead (outermost bins open)
            lo, hi = np.searchsorted(cdf, [0.001, 0.999])
            q = np.linspace(lo, hi, nbins + 1).round().astype(int)[1:-1]
        else:
            q = np.searchsorted(
                cdf, np.linspace(0, 1, nbins + 1)[1:-1]
            )  # Geant4 quantile bin edges, in fine bins
        jedges = np.unique(np.concatenate([[0], q, [len(centres)]]))
        for lab, src in runs:
            f = dict(np.load(os.path.join(src, VC_CACHE)))
            r = np.array(
                [
                    _wd_js(g["sub"][k, c], f["sub"][k, c], centres, norm, jedges)
                    for k in range(K)
                ]
            )
            rows.append(
                [
                    name,
                    lab,
                    r[:, 0].mean(),
                    r[:, 0].std(),
                    r[:, 1].mean(),
                    r[:, 1].std(),
                ]
            )
        r = np.array(
            [
                _wd_js(g["floor_a"][k, c], g["floor_b"][k, c], centres, norm, jedges)
                for k in range(K)
            ]
        )
        rows.append(
            [
                name,
                "Geant4 floor",
                r[:, 0].mean(),
                r[:, 0].std(),
                r[:, 1].mean(),
                r[:, 1].std(),
            ]
        )
    return rows


# ── stage 2: figures ─────────────────────────────────────────────────────────────
def load(src):
    return dict(np.load(os.path.join(src, CACHE)))


def series(d, kind):
    """per-shower values and errors, ECAL + HCAL summed."""
    n = float(d["n_showers"])
    if kind == "count":
        v = d["counts"].sum(0)
        return v / n, np.sqrt(v) / n
    return d["e_sum"].sum(0) / n, np.sqrt(d["e_sq"].sum(0)) / n


def plot_profile(
    bins, ref, fakes, colors, ylabel, out, c_lines, xlabel="Velocity [mm/ns]", xlim=None
):
    """plt_velocity's layout from per-shower histograms: Geant4 filled grey with its
    hatched error band, each model a line with its band, causal lines, ratio panel.
    ref / fakes: (values, errors). c_lines: [(c, label, colour), ...]. xlim defaults to
    the full bin range."""
    fig, axs = plt.subplots(2, 1, figsize=(7, 9), height_ratios=[3, 1], sharex=True)
    rv, re = ref
    axs[0].stairs(rv, bins, fill=True, color="lightgrey", rasterized=True)
    axs[0].stairs(
        rv + re,
        bins,
        baseline=np.clip(rv - re, 0, None),
        color="dimgrey",
        lw=3,
        hatch="///",
    )
    for (fv, fe), col in zip(fakes, colors):
        axs[0].stairs(
            fv + fe,
            bins,
            baseline=np.clip(fv - fe, 0, None),
            color=col,
            alpha=plt_config.alpha,
            fill=True,
        )
        axs[0].stairs(fv, bins, color=col, lw=3, baseline=None)
    positive = np.concatenate([rv[rv > 0]] + [f[0][f[0] > 0] for f in fakes])
    top = 10 * positive.max()
    bottom = max(positive.min(), top * 1e-9) / 2
    for c, lab, col in c_lines:
        axs[0].plot([c, c], [bottom, top], "--", color=col)
        axs[0].text(
            c,
            top * 0.9,
            lab,
            fontsize=20,
            color=col,
            va="top",
            ha="right" if lab == "ECal" else "left",
        )
    axs[0].plot([C, C], [bottom, top], "-k")
    axs[0].axvspan(C, max(c for c, _, _ in c_lines), facecolor="red", alpha=0.1)
    old = plt_config.color_lines
    plt_config.color_lines = list(colors)
    ratio_plots(
        axs[1],
        rv,
        [f[0] for f in fakes],
        re,
        [f[1] for f in fakes],
        bins,
        pos=(bins[:-1] + bins[1:]) / 2,
        plt_config=plt_config,
    )
    plt_config.color_lines = old
    axs[0].set_xlim(*(xlim or (bins.min(), bins.max())))
    axs[0].set_ylim(bottom, top)
    axs[0].set_yscale("log")
    axs[0].set_xscale("log")
    axs[0].set_ylabel(ylabel, fontsize=plt_config.font_labelx)
    axs[1].set_ylabel(
        "ratio to G4", fontsize=plt_config.font_ratio
    )  # as the other ratio panels
    plt.xlabel(xlabel, fontsize=plt_config.font_labelx)
    fig.tight_layout()
    fig.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


CELL_LINES = [(c_eff("ecal"), "ECal", "r"), (c_eff("hcal"), "HCal", "k")]
YLABEL = {"count": "cells / shower", "energy": "energy / shower [MeV]"}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compute")
    c.add_argument("--src", required=True, help="projected_array/<source>/<tag> folder")
    c.add_argument("--clamp", action="store_true", help="generated source: clamp times")
    c.add_argument("--workers", type=int, default=8)
    c.add_argument("--n", type=int, default=None, help="first n showers (quick test)")
    v = sub.add_parser(
        "components", help="energy-weighted log10 v_L / v_R histograms of one source"
    )
    v.add_argument("--src", required=True)
    v.add_argument("--clamp", action="store_true")
    v.add_argument("--workers", type=int, default=8)
    v.add_argument("--n", type=int, default=None)
    r = sub.add_parser(
        "component-rows",
        help="append the v_L / v_R rows to a compare_ds_occscale metrics csv",
    )
    r.add_argument("--g4", required=True)
    r.add_argument(
        "--run", action="append", required=True, help="label=projected_array folder"
    )
    r.add_argument("--csv", required=True)
    args = p.parse_args()
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
    if args.cmd == "compute":
        compute(args.src, args.clamp, args.workers, args.n)
    elif args.cmd == "components":
        compute_components(args.src, args.clamp, args.workers, args.n)
    elif args.cmd == "component-rows":
        import csv

        rows = component_rows(args.g4, [r.split("=", 1) for r in args.run])
        with open(args.csv, "a", newline="") as f:
            csv.writer(f).writerows([r[:2] + [f"{x:.5g}" for x in r[2:]] for r in rows])
        print(f"appended {len(rows)} rows to {args.csv}")


if __name__ == "__main__":
    main()
