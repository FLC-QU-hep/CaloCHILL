"""Event display in the style of the AllShowers paper: one Geant4 shower (top) and the shower generated from its
own conditioning (bottom), points at their position (depth y vs x, in dm), size = energy, colour = hit time.

The generated shower takes the per-layer point counts, the per-layer earliest times and the incident energy
straight from the Geant4 one (no LayerFM), so the two are as close as the model allows.

    # 1. Geant4 showers at 50 +- 0.5 GeV from the held-out file 99 -> small hdf5, sorted by number of points
    python scripts/evaluation/event_display.py select --energy 50 --out files/event_display/g4_50GeV.hdf5
    # 2. generate their counterparts (GPU job), e.g.
    python scripts/evaluation/eval_checkpoints_CaloHadronicVAELatent.py --inc_en 50 --fixed_energy --num <N> --bs 32 \
        --num_steps 1 --latentdiff_dir <run>/ --real_file files/event_display/g4_50GeV.hdf5 --g4_conditioning \
        --energy_calibration_factor a b c --energy_calibration_powers 0 1 2
    # 3. candidates overview (best-looking first) and the final two-panel figure for the chosen index
    python scripts/evaluation/event_display.py plot --g4 files/event_display/g4_50GeV.hdf5 --gen <gen hdf5> \
        --out figs/event_display/ --candidates 24 [--idx 17]
"""

import argparse
import os
import sys

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.constants import dataset_path_pions  # noqa: E402

REPO = "./"
TEST_FILE = dataset_path_pions.format(99)  # held-out test file
LAYER_POS = (
    np.load(REPO + "files/geometry/layer_bottom_pos_ecal+hcal.npy")
    + np.load(REPO + "files/geometry/layer_top_pos_ecal+hcal.npy")
) / 2  # layer centres along the beam, mm
N_ECAL = 30  # layers 0-29 ECal, 30-77 HCal


def select(args):
    f = h5py.File(TEST_FILE, "r")
    e = f["energy"][:, 0]
    lo, hi = (
        (args.e_min, args.e_max)
        if args.e_min is not None
        else (args.energy - args.window, args.energy + args.window)
    )
    sel = np.where((e > lo) & (e < hi))[0]
    if (
        args.max_showers and len(sel) > args.max_showers
    ):  # a random subset, reproducible
        sel = np.sort(
            np.random.default_rng(args.seed).choice(
                sel, args.max_showers, replace=False
            )
        )
    n = f["n_points"][:][sel]
    # the eval script re-sorts its real showers by number of points; load() pairs them back by incident energy
    sel = sel[np.argsort(n, kind="stable")]
    assert len(np.unique(e[sel])) == len(sel), (
        "repeated incident energies: pairing by energy would be ambiguous"
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with h5py.File(args.out, "w") as g:
        g["events"] = f["events"][:][
            sel
        ]  # sel is sorted by n, not by index: read all, then pick
        g["energy"] = f["energy"][:][sel]
        g["shower_start"] = f["shower_start"][:][sel]
        g["orig_index"] = sel
    print(f"wrote {args.out}: {len(sel)} showers in [{lo}, {hi}] GeV")


def load(args):
    """Both sets as lists of (x, depth, z, E [MeV], t [ns]) point arrays, same order."""
    with h5py.File(args.g4, "r") as f:
        ev, e, start = f["events"][:], f["energy"][:, 0], f["shower_start"][:]
    g4 = []
    for s in ev:  # (5, P): x, layer, z, E [GeV], t
        m = s[3] > 0
        g4.append(
            np.stack(
                [
                    s[0][m],
                    LAYER_POS[s[1][m].astype(int)],
                    s[2][m],
                    s[3][m] * 1000,
                    s[4][m],
                ],
                1,
            )
        )
    out = {}
    with h5py.File(args.gen, "r") as f:
        for k in (
            "gen",
            "reco",
        ):  # reco: the same Geant4 shower through the VAE, encoded and decoded once
            v, ve = f[k + "_events"][:], f[k + "_energy"][:, 0]
            # pair by incident energy (unique per shower): the eval script reorders by number of points
            order = np.argsort(ve)[np.argsort(np.argsort(e))]
            assert len(v) == len(g4) and np.allclose(ve[order], e, atol=1e-3), (
                f"{k} does not pair with the Geant4 file"
            )
            out[k] = []
            for s in v[order]:  # (P, 5): x, z, layer, E [MeV], t
                m = s[:, 3] > 0
                out[k].append(
                    np.stack(
                        [
                            s[m, 0],
                            LAYER_POS[s[m, 2].astype(int)],
                            s[m, 1],
                            s[m, 3],
                            s[m, 4],
                        ],
                        1,
                    )
                )
    return g4, out["gen"], out["reco"], e, start


def project(p):
    """Points -> readout cells, as the analysis projection (utils/projection.py): 5 mm ECal and 30 mm HCal cells,
    energy summed, time = earliest hit, drawn at the cell centre (the per-shower random grid shift is left out)."""
    layer = np.searchsorted(LAYER_POS, p[:, 1] - 1e-3)
    size = np.where(layer < N_ECAL, 5.0, 30.0)
    ix, iz = np.floor((p[:, 0] + 450) / size), np.floor((p[:, 2] + 450) / size)
    key, inv = np.unique(np.stack([layer, ix, iz], 1), axis=0, return_inverse=True)
    inv = inv.ravel()
    e = np.bincount(inv, p[:, 3])
    t = np.full(len(key), np.inf)
    np.minimum.at(t, inv, p[:, 4])
    sz = np.where(key[:, 0] < N_ECAL, 5.0, 30.0)
    return np.stack(
        [
            -450 + (key[:, 1] + 0.5) * sz,
            LAYER_POS[key[:, 0].astype(int)],
            -450 + (key[:, 2] + 0.5) * sz,
            e,
            t,
        ],
        1,
    )


def track_hits(p, r_core=60.0, link=45.0, min_layers=6, max_rms=20.0):
    """Hits on secondary MIP-like tracks: off the shower core (> r_core mm from the energy-weighted axis), linked to
    neighbours within `link` mm in other layers, in chains spanning >= min_layers layers with ~one hit per layer and
    <= max_rms mm transverse scatter about a straight line. Returns a boolean mask over the points of p."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    x0, z0 = np.average(p[:, 0], weights=p[:, 3]), np.average(p[:, 2], weights=p[:, 3])
    off = np.hypot(p[:, 0] - x0, p[:, 2] - z0) > r_core
    idx = np.where(off)[0]
    mask = np.zeros(len(p), bool)
    if len(idx) < min_layers:
        return mask
    q = p[idx][:, :3]
    layer = np.searchsorted(LAYER_POS, p[idx, 1] - 1e-3)
    pairs = cKDTree(q).query_pairs(link, output_type="ndarray")
    pairs = pairs[layer[pairs[:, 0]] != layer[pairs[:, 1]]] if len(pairs) else pairs
    if not len(pairs):
        return mask
    g = coo_matrix(
        (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(q), len(q))
    )
    n, lab = connected_components(g, directed=False)
    for k in range(n):
        m = lab == k
        nl = len(np.unique(layer[m]))
        if (
            nl < min_layers or m.sum() > 2.5 * nl
        ):  # too short, or a blob rather than a chain
            continue
        y = q[m, 1]  # straight line x(y), z(y)
        rms = np.sqrt(
            np.mean(
                [
                    np.var(q[m, j] - np.polyval(np.polyfit(y, q[m, j], 1), y))
                    for j in (0, 2)
                ]
            )
        )
        if rms <= max_rms:
            mask[idx[m]] = True
    return mask


def track_score(g4, gen, match=40.0, view=0):
    """(G4 track hits, generated track hits, G4 track hits with a generated track hit within `match` mm in the
    displayed projection: transverse coordinate `view` (0 = x, 2 = z) against depth). The generated shower only
    gets the per-layer counts and times of the Geant4 one, so alignment in 3D would be chance; what the reader
    sees is the projection."""
    from scipy.spatial import cKDTree

    tg, tm = track_hits(g4), track_hits(gen)
    if not tg.any() or not tm.any():
        return int(tg.sum()), int(tm.sum()), 0
    d, _ = cKDTree(gen[tm][:, [view, 1]]).query(g4[tg][:, [view, 1]])
    return int(tg.sum()), int(tm.sum()), int((d < match).sum())


def score(p):
    """Higher = nicer display: a long primary track before the interaction plus many hits away from the core."""
    core = np.average(p[:, 0], weights=p[:, 3])
    far = (np.abs(p[:, 0] - core) > 100).sum()
    return far


def draw(ax, p, norm, cmap, smax, emax, xlim, title, view=0):
    o = np.argsort(-p[:, 3])  # big points first, small ones on top
    p = p[o]
    ax.scatter(
        p[:, view] / 100,
        p[:, 1] / 100,
        s=np.clip(smax * p[:, 3] / emax, 0.3, smax),
        c=p[:, 4],
        cmap=cmap,
        norm=norm,
        linewidths=0,
        rasterized=True,
    )
    ax.axhline(
        (LAYER_POS[N_ECAL - 1] + LAYER_POS[N_ECAL]) / 200, color="grey", lw=1
    )  # ECal / HCal
    ax.set_xlim(*xlim)
    ax.xaxis.set_major_locator(
        mpl.ticker.MultipleLocator(2)
    )  # -4 ... 4 dm, whatever the centre
    ax.set_ylim(LAYER_POS[0] / 100 - 0.3, LAYER_POS[-1] / 100 + 0.3)
    ax.set_title(title, fontsize=15)
    ax.set_xlabel(rf"${'xz'[view // 2]}$ [dm]", fontsize=14)
    ax.set_ylabel(r"$y$ [dm]", fontsize=14)
    ax.tick_params(labelsize=12)


def figure(panels, e, path, args, view=0):
    """panels: [(title, points)], drawn side by side with one shared size and time scale."""
    cmap = plt.get_cmap(args.cmap)
    if args.cmap_max < 1:  # drop the pale end of the map, invisible on white
        cmap = mpl.colors.ListedColormap(cmap(np.linspace(0, args.cmap_max, 256)))
    if (
        not args.early_cool
    ):  # earliest hits (the start of the shower) in the warm end of the map
        cmap = cmap.reversed()
    cmap = cmap.with_extremes(
        over=args.over_color
    )  # hits later than the scale: one colour of their own
    t = np.concatenate([p[:, 4] for _, p in panels])
    # most hits arrive within a few ns: the scale stops at a percentile so that they spread over the colours,
    # later hits saturate (arrow on the colour bar)
    w = np.concatenate(
        [p[:, 3] for _, p in panels]
    )  # energy-weighted: the visible points, not the tiny late halo
    o = np.argsort(t)
    c = np.cumsum(w[o]) / w.sum()
    tmax = args.t_max or t[o][np.searchsorted(c, args.t_pct / 100)]
    norm = mpl.colors.LogNorm(max(t[o][np.searchsorted(c, 0.01)], 0.05), tmax)
    emax = np.percentile(
        np.concatenate([p[:, 3] for _, p in panels]), 99.5
    )  # common size scale
    g4 = panels[0][1]
    xc = np.average(g4[:, view], weights=g4[:, 3]) / 100
    xlim = (xc - args.half_width, xc + args.half_width)
    fig, axs = plt.subplots(
        1,
        len(panels),
        figsize=(args.panel_width * len(panels) + 0.9, args.panel_width * 1.62),
        sharey=True,
        constrained_layout=True,
    )
    fig.suptitle(rf"$\pi^+$, $E$ = {e:.1f} GeV", fontsize=16)
    for ax, (title, p) in zip(axs, panels):
        draw(ax, p, norm, cmap, args.smax, emax, xlim, title, view)
    for ax in axs[1:]:
        ax.set_ylabel("")
    cb = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=axs,
        shrink=0.85,
        pad=0.01,
        extend="max" if args.t_pct < 100 else "neither",
    )
    cb.set_label("time [ns]", fontsize=14)
    ticks = [
        x for x in (0.5, 1, 2, 3, 5, 7, 10, 20, 50, 100) if norm.vmin <= x <= norm.vmax
    ]
    cb.set_ticks(
        ticks, labels=[f"{x:g}" for x in ticks]
    )  # a few plain values, not 2 x 10^0
    cb.ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    cb.ax.tick_params(labelsize=12)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def figure3d(panels, e, path, args):
    """Trial: the same panels in 3D (x, z transverse, depth y vertical), shared size and time scale."""
    t = np.concatenate([p[:, 4] for _, p in panels])
    w = np.concatenate([p[:, 3] for _, p in panels])
    o = np.argsort(t)
    c = np.cumsum(w[o]) / w.sum()
    lo, hi = (
        max(t[o][np.searchsorted(c, 0.01)], 0.05),
        args.t_max or t[o][np.searchsorted(c, 0.95)],
    )
    norm = mpl.colors.Normalize(lo, hi) if args.t_linear else mpl.colors.LogNorm(lo, hi)
    # --t-shift: every time takes the colour that (time + shift) had, i.e. the palette slides towards its late end
    d = (
        (args.cmap_max - args.cmap_min) * args.t_shift / (hi - lo)
        if args.t_linear
        else 0.0
    )
    base = plt.get_cmap(args.cmap)
    a, b = args.cmap_min, args.cmap_max
    if args.early_cool:
        a, b = a + d, b + d
    else:  # reversed map: late = low end, so a later colour means a lower position
        a, b = a - d, b - d
    cmap = mpl.colors.ListedColormap(base(np.linspace(max(a, 0), min(b, 1), 256)))
    if not args.early_cool:
        cmap = cmap.reversed()
    cmap = cmap.with_extremes(over=args.over_color)
    emax = np.percentile(w, 99.5)
    g4 = panels[0][1]
    xc, zc = (np.average(g4[:, j], weights=g4[:, 3]) / 100 for j in (0, 2))
    n = len(panels)
    base = 2.7 * n + (
        0 if args.cbar_top else 0.7
    )  # colour bar on the right takes 0.7 in
    W = base + (n - 1) * args.panel_gap
    H = 3.6 + (0.4 if args.cbar_top else 0)  # colour bar on top takes 0.4 in
    fig = plt.figure(figsize=(W, H))
    pw = (
        0.86 * base / W / n
    )  # 3D axes carry wide built-in margins: place them by hand, overlapping, so the boxes sit close
    for k, (title, p) in enumerate(panels):
        ax = fig.add_axes(
            [
                0.01 + k * (pw * 0.97 + args.panel_gap / W),
                0.06 * 3.6 / H,
                pw * 1.12,
                0.88 * 3.6 / H,
            ],
            projection="3d",
        )
        q = p[np.argsort(-p[:, 3])]
        ax.scatter(
            q[:, 0] / 100,
            q[:, 2] / 100,
            q[:, 1] / 100,
            s=np.clip(
                args.smax * (q[:, 3] / emax) ** args.size_power, args.smin, args.smax
            ),
            c=q[:, 4],
            cmap=cmap,
            norm=norm,
            linewidths=0,
            depthshade=False,
            rasterized=True,
            axlim_clip=True,
        )  # nothing drawn outside the shown x / z range
        hw = args.half_width
        xx, zz = np.meshgrid([xc - hw, xc + hw], [zc - hw, zc + hw])
        ax.plot_surface(
            xx,
            zz,
            np.full_like(xx, (LAYER_POS[N_ECAL - 1] + LAYER_POS[N_ECAL]) / 200),
            color="0.93",
            alpha=0.6,
            shade=False,
        )  # ECal/HCal plane: light, darker than the walls
        ax.set_xlim(xc - hw, xc + hw)
        ax.set_ylim(zc - hw, zc + hw)
        ax.set_zlim(LAYER_POS[0] / 100 - 0.3, LAYER_POS[-1] / 100 + 0.3)
        ax.view_init(elev=args.elev, azim=args.azim)
        ax.grid(False)
        ax.set_facecolor(
            "none"
        )  # overlapping panels: no panel hides its neighbour's labels
        for a in (ax.xaxis, ax.yaxis, ax.zaxis):  # walls nearly white
            a.set_pane_color((0.99, 0.99, 0.99, 1.0))
        ax.set_box_aspect((1, 1, 1.6), zoom=args.zoom)
        tk = int(args.half_width)  # ticks at the edges of the shown range
        ax.set_xticks([round(xc) - tk, round(xc), round(xc) + tk])
        ax.set_yticks([round(zc) - tk, round(zc), round(zc) + tk])
        ax.set_zticks([20, 25, 30])
        ax.tick_params(labelsize=10, pad=0)
        ax.set_xlabel(r"$x$ [dm]", labelpad=-2)
        ax.set_ylabel(r"$z$ [dm]", labelpad=-2)
        if (
            k == n - 1 or not args.cbar_top
        ):  # side by side, the next panel would hide it: depth label on the last panel only
            ax.set_zlabel(r"$y$ [dm]", labelpad=-2)
        ax.set_title(title, fontsize=14)
    # no title: the caption carries the particle and energy
    if args.cbar_top:  # horizontal, centred above the panels
        cax = fig.add_axes([0.3, 1 - 0.22 / H, 0.4, 0.11 / H])
        cb = fig.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=cax,
            extend="max",
            orientation="horizontal",
        )
        cax.xaxis.set_ticks_position("top")
        cax.xaxis.set_label_position("top")
    else:
        cax = fig.add_axes([1 - 0.1 * base / W, 0.22, 0.014, 0.56])
        cb = fig.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, extend="max"
        )
    cb.set_label("time [ns]", fontsize=14)
    ticks = [
        x
        for x in (
            (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
            if args.t_linear
            else (0.5, 1, 2, 3, 5, 7, 10, 20, 50, 100)
        )
        if norm.vmin <= x <= norm.vmax
    ]
    cb.set_ticks(ticks, labels=[f"{x:g}" for x in ticks])
    (cb.ax.xaxis if args.cbar_top else cb.ax.yaxis).set_minor_formatter(
        mpl.ticker.NullFormatter()
    )
    fig.savefig(
        path, dpi=300, bbox_inches="tight", pad_inches=0.3 if args.cbar_top else 0.15
    )  # padding keeps the bottom x labels
    plt.close(fig)


def plot(args):
    # same text settings as the other paper figures (scripts/utils/plotting_features.py)
    mpl.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Nimbus Sans", "Arial", "DejaVu Sans"],
        }
    )
    g4, gen, reco, e, start = load(args)
    if args.projected:
        g4, gen, reco = ([project(p) for p in s] for s in (g4, gen, reco))
        args.tag += "_projected"
    os.makedirs(args.out, exist_ok=True)

    def panels(i):
        p = [(r"\textsc{Geant4}", g4[i])]
        if args.reco:
            p.append(("VAE reconstruction", reco[i]))
        return p + [(args.label, gen[i])]

    if args.idx is not None and args.three_d:
        for i in args.idx:
            path = os.path.join(args.out, f"event_display3d_{args.tag}_{i}.pdf")
            figure3d(panels(i), e[i], path, args)
            print("wrote", path)
        return
    if args.idx is not None:
        for i in args.idx:
            path = os.path.join(
                args.out,
                f"event_display_{args.tag}{'_reco' if args.reco else ''}_{i}.pdf",
            )
            figure(panels(i), e[i], path, args, args.view)
            print("wrote", path)
        return
    # candidates: interaction in the HCal (visible primary track through the ECal), ranked by off-core hits
    cand = [i for i in range(len(g4)) if args.start_min <= start[i] <= args.start_max]
    if (
        args.rank == "tracks"
    ):  # secondary tracks in Geant4 that the generated shower also has, in the same place
        ts = {
            i: max(
                (track_score(g4[i], gen[i], view=v) + (v,) for v in (0, 2)),
                key=lambda t: t[2],
            )
            for i in cand
        }
        cand = sorted(cand, key=lambda i: -ts[i][2])[: args.candidates]
        print(
            "candidates (index, start, G4 track hits, gen track hits, aligned, view):",
            [(i, int(start[i]), *ts[i]) for i in cand],
        )
    else:
        cand = sorted(cand, key=lambda i: -score(g4[i]))[: args.candidates]
        print(
            "candidates (index, shower start, off-core hits):",
            [(i, int(start[i]), score(g4[i])) for i in cand],
        )
    for i in cand:
        v = ts[i][3] if args.rank == "tracks" else args.view
        figure(
            panels(i),
            e[i],
            os.path.join(args.out, f"candidate_{args.tag}_{i:04d}{'xz'[v // 2]}.png"),
            args,
            v,
        )
    print(f"wrote {len(cand)} candidates to {args.out}; pick one with --idx")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--energy", type=float, default=50)
    s.add_argument("--window", type=float, default=0.5)
    s.add_argument(
        "--e-min",
        type=float,
        default=None,
        help="energy range instead of --energy +- --window",
    )
    s.add_argument("--e-max", type=float, default=None)
    s.add_argument(
        "--max-showers",
        type=int,
        default=0,
        help="random subset of at most this many (0 = all)",
    )
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--out", required=True)
    q = sub.add_parser("plot")
    q.add_argument("--g4", required=True)
    q.add_argument("--gen", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--tag", default="50GeV")
    q.add_argument("--label", default="This work")
    q.add_argument("--idx", type=int, nargs="+", default=None)
    q.add_argument("--candidates", type=int, default=24)
    q.add_argument(
        "--view",
        type=int,
        choices=[0, 2],
        default=0,
        help="horizontal axis: 0 = x, 2 = z",
    )
    q.add_argument(
        "--three-d", action="store_true", help="trial: 3D panels (x, z, depth)"
    )
    q.add_argument("--elev", type=float, default=18)
    q.add_argument("--azim", type=float, default=-55)
    q.add_argument(
        "--rank",
        choices=["offcore", "tracks"],
        default="offcore",
        help="tracks: rank by Geant4 secondary-track hits matched by generated track hits",
    )
    q.add_argument(
        "--start-min",
        type=int,
        default=30,
        help="shower start layer range of the candidates",
    )
    q.add_argument("--start-max", type=int, default=50)
    q.add_argument(
        "--half-width",
        type=float,
        default=4.0,
        help="x range around the shower core, dm",
    )
    q.add_argument(
        "--smax",
        type=float,
        default=40,
        help="marker size of the most energetic points",
    )
    q.add_argument(
        "--cmap",
        default="plasma",
        help="colour map of the hit time (plasma, as the LDA figures)",
    )
    q.add_argument(
        "--cmap-max",
        type=float,
        default=0.85,
        help="upper end of the map used (1 = full)",
    )
    q.add_argument(
        "--cmap-min",
        type=float,
        default=0.0,
        help="lower end of the map used (higher = gentler colour steps)",
    )
    q.add_argument(
        "--t-linear",
        action="store_true",
        help="3D: linear time colour scale instead of log",
    )
    q.add_argument(
        "--panel-gap",
        type=float,
        default=0.0,
        help="3D: extra space between panels, inches",
    )
    q.add_argument(
        "--t-shift",
        type=float,
        default=0.0,
        help="3D, linear scale: give each time the colour of time + t-shift [ns] (darker)",
    )
    q.add_argument(
        "--t-max",
        type=float,
        default=7.0,
        help="end of the time colour scale [ns]; 0 = from --t-pct",
    )
    q.add_argument(
        "--over-color", default="0.72", help="colour of the hits later than the scale"
    )
    q.add_argument(
        "--panel-width",
        type=float,
        default=2.6,
        help="inches per panel (smaller = bigger text in the paper)",
    )
    q.add_argument(
        "--early-cool",
        action="store_true",
        help="map not reversed: earliest hits in the dark/cool end",
    )
    q.add_argument(
        "--t-pct",
        type=float,
        default=95,
        help="energy-weighted percentile of the hit times where the colour scale ends",
    )
    q.add_argument(
        "--cbar-top",
        action="store_true",
        help="3D: horizontal colour bar above the panels",
    )
    q.add_argument(
        "--smin",
        type=float,
        default=0.3,
        help="3D: smallest marker size (raise it to keep MIP-like track cells visible)",
    )
    q.add_argument(
        "--size-power",
        type=float,
        default=1.0,
        help="3D: marker size ~ E^power (0.5 lifts MIP-like track cells above the halo)",
    )
    q.add_argument(
        "--zoom",
        type=float,
        default=1.0,
        help="3D: enlarge the box inside each panel (less empty margin)",
    )
    q.add_argument(
        "--projected",
        action="store_true",
        help="draw the readout cells (after the projection) instead of the points",
    )
    q.add_argument(
        "--reco",
        action="store_true",
        help="add the VAE reconstruction of the Geant4 shower as a middle panel",
    )
    a = p.parse_args()
    select(a) if a.cmd == "select" else plot(a)
