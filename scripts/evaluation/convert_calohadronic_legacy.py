"""Convert a legacy CaloHadronic (HGx9_CNF_ECAL_HCAL_30) sample into the format
occ-scale.py reads, so it goes through the current projection (dither included).

Legacy: gen_showers_<E>GeV_<N>.hdf5 with 'events' (N, 4, P) = [x, layer, z, e] and
'energy' (N, 1) normalised as (E - 50) / 50, no time.
Output: showers_<E>GeV_<N>.hdf5 with 'gen_events' (N, P, 5) = [x, z, layer, e, t=0]
and 'gen_energy' (N, 1) in GeV.
Same conversion as the 2026-07-23 rebuild of showers_50GeV_5000.hdf5.

    python scripts/evaluation/convert_calohadronic_legacy.py --tag 10GeV_10000
"""

import argparse
import os

import h5py
import numpy as np

DIR = "./files/generated_showers/HGx9_CNF_ECAL_HCAL_30/"


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--tag", required=True, help="e.g. 10GeV_10000")
    args = p.parse_args()
    src, dst = DIR + f"gen_showers_{args.tag}.hdf5", DIR + f"showers_{args.tag}.hdf5"
    if os.path.exists(dst):
        raise SystemExit(f"{dst} exists, not overwriting")
    with h5py.File(src, "r") as f:
        ev, en = f["events"][:], f["energy"][:]
    out = np.zeros((ev.shape[0], ev.shape[2], 5), dtype=ev.dtype)
    out[..., :4] = np.moveaxis(ev[:, [0, 2, 1, 3]], 1, 2)
    with h5py.File(dst, "w") as f:
        f.create_dataset("gen_events", data=out)
        f.create_dataset("gen_energy", data=en * 50 + 50)
    print("wrote", dst, out.shape)


if __name__ == "__main__":
    main()
