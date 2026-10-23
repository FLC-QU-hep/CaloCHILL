"""Energy-dependent calibration f(E) = sum_k a_k E^p_k from an UNCALIBRATED (factor 1) full-range sample:
per 2 GeV bin f = mean(real deposited) / mean(gen deposited), least-squares fit on the bin centres.
Prints "a0 a1 a2 | powers" like fits_full_range_aic.txt, and writes the global factor (incident-energy
range common to both samples) to calib_0GeV.txt next to the hdf5, the f0 of plot_energy_calibration_fit.py.
    python scripts/evaluation/fit_poly_calib.py <showers_0GeV_N.hdf5> [powers, default 0 1 2]  (from the repository root)"""

import glob
import os
import sys

import h5py
import numpy as np

REAL_GLOB = "dataset/hdf5_HGx9new/all_interactions_pions_regular_ECAL+HCAL_10-90GeV_*.slcio.root_with_time.hdf5"


def load_from_hdf5(path, events_key="gen_events", energy_key="gen_energy"):
    """(showers [N, points, features], incident energy [GeV]) of a generated hdf5."""
    with h5py.File(path, "r") as f:
        return f[events_key][:], np.asarray(f[energy_key][:]).reshape(-1)


def total_deposited_energy(showers):
    """Sum of hit energies (feature 3) per shower; padded hits are zero."""
    return showers[:, :, 3].sum(axis=1)


def load_real_reference(file_paths):
    """(deposited energy [MeV], incident energy [GeV]) of Geant4 dataset files (hit energy in GeV, x1000 as dataset.py)."""
    dep, inc = [], []
    for path in file_paths:
        with h5py.File(path, "r") as f:
            dep.append(f["events"][:][:, 3, :].sum(axis=1) * 1000)
            inc.append(f["energy"][:].reshape(-1))
    return np.concatenate(dep), np.concatenate(inc)


if __name__ == "__main__":
    powers = [int(p) for p in sys.argv[2:]] or [0, 1, 2]
    showers, e_gen = load_from_hdf5(sys.argv[1])
    gen = total_deposited_energy(showers)
    real, e_real = load_real_reference(sorted(glob.glob(REAL_GLOB))[:5])

    edges = np.arange(10, 90.001, 2.0)
    centres, f = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        g, r = (e_gen >= lo) & (e_gen < hi), (e_real >= lo) & (e_real < hi)
        if g.sum() > 20 and r.sum() > 20:
            centres.append(0.5 * (lo + hi))
            f.append(real[r].mean() / gen[g].mean())
    centres, f = np.array(centres), np.array(f)
    A = np.stack([centres**p for p in powers], 1)
    coef = np.linalg.lstsq(A, f, rcond=None)[0]
    rms = np.sqrt(np.mean((A @ coef - f) ** 2))
    print(f"bins {len(f)}  global {real.mean() / gen.mean():.5f}  fit rms {rms:.4f}")
    print(" ".join(f"{c:.6g}" for c in coef), "|", " ".join(map(str, powers)))

    # global factor on the common incident-energy range (as the former compute_energy_calibration.py)
    lo, hi = max(e_gen.min(), e_real.min()), min(e_gen.max(), e_real.max())
    f0 = (
        real[(e_real >= lo) & (e_real <= hi)].mean()
        / gen[(e_gen >= lo) & (e_gen <= hi)].mean()
    )
    calib_file = os.path.join(
        os.path.dirname(os.path.abspath(sys.argv[1])), "calib_0GeV.txt"
    )
    with open(calib_file, "w") as fh:
        fh.write(str(f0))
    print(f"global factor {f0:.5f} -> {calib_file}")
