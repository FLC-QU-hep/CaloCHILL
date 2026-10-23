import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import h5py
import numpy as np
from tqdm import tqdm
from utils.constants import (
    dataset_path_pions as dataset_path,
)
from utils.constants import (
    f_names,
    f_names_r,
    time_names,
    x_bins,
    x_bins_hcal,
    y_bins,
    y_bins_ecal,
    y_bins_hcal,
    z_bins,
    z_bins_hcal,
)
from utils.plotting_features import c_eff, combined_time_floor, get_cog2
from utils.projection import projection_ecal_hcal

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, required=True)
parser.add_argument(
    "--load-dataset-and-project", action=argparse.BooleanOptionalAction, default=True
)
parser.add_argument(
    "--compute-features", action=argparse.BooleanOptionalAction, default=True
)
parser.add_argument(
    "--reco-showers-to-save", action=argparse.BooleanOptionalAction, default=True
)
parser.add_argument(
    "--fake-showers-to-save", action=argparse.BooleanOptionalAction, default=True
)
parser.add_argument(
    "--real-showers-to-save", action=argparse.BooleanOptionalAction, default=True
)
parser.add_argument("--with-time", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument(
    "--compare-reco", action=argparse.BooleanOptionalAction, default=True
)
parser.add_argument(
    "--compare-allshowers", action=argparse.BooleanOptionalAction, default=False
)
parser.add_argument(
    "--allshowers-path",
    type=str,
    default=None,
    help="Path to a bussthor-format AllShowers hdf5 (point-cloud showers with "
    "'showers'/'num_points'/'energies' datasets, per-hit columns [x, z, layer, "
    "energy]), used as an extra comparison source when --compare-allshowers is set.",
)
parser.add_argument(
    "--allshowers-cache",
    default="AllShowers_bussthor",
    help="Folder under files/projected_array/ for the --allshowers-path features, so "
    "different AllShowers samples (e.g. solvers) keep separate caches.",
)
parser.add_argument("--shw", type=int, default=10000)
parser.add_argument(
    "--g4-file",
    default=None,
    help="Take the Geant4 reference from this one HGx9 hdf5 (e.g. a converted "
    "fixed-energy ddsim sample, scripts/create_data/convert_ddsim_h5_to_hgx9.py), its "
    "first --shw showers, instead of the dataset files in [--g4-e-min, --g4-e-max]; "
    "cached as Geant4/<E>GeV_ddsim_<shw>",
)
parser.add_argument("--e-min", type=float, default=10)
parser.add_argument("--e-max", type=float, default=90)
parser.add_argument(
    "--g4-e-min",
    type=float,
    default=None,
    help="Lower edge of the incident-energy window for the Geant4 reference "
    "showers (defaults to --e-min). A narrow window has only ~45 showers per "
    "dataset file, so widen it (e.g. 49.5-50.5) to get --shw distinct showers.",
)
parser.add_argument(
    "--g4-e-max",
    type=float,
    default=None,
    help="Upper edge of the Geant4 reference window (defaults to --e-max).",
)
parser.add_argument("--calib", type=str, default="")
parser.add_argument(
    "--clip-time-to-geant4",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Clip generated (non-Geant4) per-cell hit times to the per-cell "
    "minimum time observed in the Geant4 reference sample, removing "
    "unphysical (faster-than-light) hits. The reference floor is cached to "
    "geant4_time_floor_path and reused across runs; a run that (re)projects "
    "real Geant4 showers with timing will build it if missing.",
)
args = parser.parse_args()

name = args.name
load_dataset_and_project = args.load_dataset_and_project
compute_features = args.compute_features
reco_SHOWERS_TO_SAVE = args.reco_showers_to_save
fake_SHOWERS_TO_SAVE = args.fake_showers_to_save
real_SHOWERS_TO_SAVE = args.real_showers_to_save
both_to_save = reco_SHOWERS_TO_SAVE and fake_SHOWERS_TO_SAVE and real_SHOWERS_TO_SAVE
with_time_flag = args.with_time
compare_with_reco_showers = args.compare_reco
compare_with_AllShowers = args.compare_allshowers
allshowers_path = args.allshowers_path
if compare_with_AllShowers and allshowers_path is None:
    parser.error("--compare-allshowers needs --allshowers-path")
shw = args.shw
e_min, e_max = args.e_min, args.e_max
g4_e_min = args.g4_e_min if args.g4_e_min is not None else e_min
g4_e_max = args.g4_e_max if args.g4_e_max is not None else e_max
calib_suffix = f"_calib{args.calib}" if args.calib else ""
clip_time_to_geant4 = args.clip_time_to_geant4
print("e_min:", e_min, "e_max: ", e_max)

if e_min < 12 and e_max > 88:
    inc_en = "0GeV_"  # all range, matches eval script convention
else:
    inc_en = str(int(e_max + e_min) // 2) + "GeV_"

inc_en_g4 = inc_en
if args.g4_file:
    inc_en_g4 = inc_en + "ddsim_"
elif (g4_e_min, g4_e_max) != (e_min, e_max):
    # keep a Geant4 cache built with a different window in its own folder
    inc_en_g4 = inc_en + f"g4win{g4_e_min:g}-{g4_e_max:g}_"
print("Geant4 window: g4_e_min:", g4_e_min, "g4_e_max:", g4_e_max)

print(name)

# AllShowers (bussthor) is a static reference dataset, independent of which model
# checkpoint/name is being evaluated: skip re-loading/re-projecting/re-computing its
# features once they're already cached on disk for this inc_en/shw combination.
allshowers_cache_dir = (
    "./files/projected_array/" + args.allshowers_cache + "/" + inc_en + str(shw)
)
allshowers_cached = compare_with_AllShowers and os.path.exists(
    os.path.join(allshowers_cache_dir, "e_sum_list.npy")
)
if allshowers_cached:
    print("AllShowers features already cached — skipping reload/reprojection.")


def denormalize_inc_energy(raw):
    """gen_energy/reco_energy is stored differently across model checkpoints:
    older runs save it normalized to roughly [-1, 1] (via (E-50)/50, so the
    correct inverse is (x+1)/2*100), while newer runs save the physical
    incident energy in GeV directly. Detect which one we got: normalized
    values never exceed ~1.5 in magnitude, physical GeV values for this
    dataset (10-90 GeV) always do.
    """
    if np.abs(raw).max() <= 1.5:
        return (raw + 1) / 2 * 100
    return raw


geant4_time_floor_path = "./files/projected_array/Geant4/time_floor.npy"


def compute_geant4_time_floor(real_events_time):
    """Per-cell (x, layer, z) minimum hit time observed in the Geant4 reference
    sample -- used as a causality floor when clipping generated shower timing.
    Cells Geant4 never registered a hit in stay NaN (no floor, left unclipped).
    """
    hit_times = np.where(real_events_time > 0, real_events_time, np.nan)
    return np.nanmin(hit_times, axis=0)


def build_geant4_time_floor_from_file(real_events_hdf5_path, batch_size=500):
    """Stream an already-projected Geant4 events hdf5 (containing
    'projected_time_showers') to build the per-cell time floor, without
    needing to reproject real showers this run and without loading the
    (potentially very large) file into memory at once."""
    floor = None
    with h5py.File(real_events_hdf5_path, "r") as f:
        real_time = f["projected_time_showers"]
        for start in range(0, real_time.shape[0], batch_size):
            batch_floor = compute_geant4_time_floor(
                real_time[start : start + batch_size]
            )
            floor = batch_floor if floor is None else np.fmin(floor, batch_floor)
    return floor


def clip_time_to_geant4_floor(events_time, time_floor):
    """Clip per-cell generated hit times to be no earlier than the later of the
    Geant4 floor and the causal floor, removing unphysical (faster-than-light)
    hits. Cells Geant4 never hit (NaN Geant4 floor) get the causal floor alone;
    time_floor=None uses the causal floor everywhere."""
    floor = combined_time_floor(time_floor)
    floor_b = np.broadcast_to(floor, events_time.shape)
    hit_mask = (events_time > 0) & ~np.isnan(floor_b)
    events_time[hit_mask] = np.maximum(events_time[hit_mask], floor_b[hit_mask])
    return events_time


if load_dataset_and_project:
    file = h5py.File(
        "./files/generated_showers/"
        + name
        + "/showers_"
        + inc_en
        + str(shw)
        + calib_suffix
        + ".hdf5",
        "r",
    )
    if fake_SHOWERS_TO_SAVE:
        fake_showers = file["gen_events"][:]
        fake_showers = np.moveaxis(fake_showers, 1, 2)  # put layers in the middle
        if (
            fake_showers.shape[1] == 4
        ):  # no-time model: zero time column, run with --no-with-time
            fake_showers = np.concatenate(
                [fake_showers, np.zeros_like(fake_showers[:, :1])], axis=1
            )
        # gen format is [x, y, layer, e, t]; projection expects [x, layer, z, e, t]
        fake_showers = fake_showers[:, [0, 2, 1, 3, 4], :]
        fake_inc_energies = denormalize_inc_energy(file["gen_energy"][:])
        shw = fake_showers.shape[0]

        fake_showers[:, 1] = fake_showers[:, 1] + 0.5
        fake_showers[:, 1][fake_showers[:, 3] == 0] = 0

    if compare_with_reco_showers:
        if reco_SHOWERS_TO_SAVE:
            reco_showers = file["reco_events"][:]
            reco_showers = np.moveaxis(reco_showers, 1, 2)
            if (
                reco_showers.shape[1] == 4
            ):  # no-time VAE: zero time column, as for the generated showers
                reco_showers = np.concatenate(
                    [reco_showers, np.zeros_like(reco_showers[:, :1])], axis=1
                )
            # reco format is [x, y, layer, e, t]; projection expects [x, layer, z, e, t]
            reco_showers = reco_showers[:, [0, 2, 1, 3, 4], :]
            reco_inc_energies = denormalize_inc_energy(file["reco_energy"][:])
            reco_showers[:, 1] = reco_showers[:, 1] + 0.5
            reco_showers[:, 1][reco_showers[:, 3] == 0] = 0

    if real_SHOWERS_TO_SAVE:
        real_showers = np.zeros((shw, 5, 6000))
        point_per_layer_file = np.zeros((shw, 6000))
        inc_energies = np.zeros((shw, 1))
        # Take `shw` *distinct* showers in [g4_e_min, g4_e_max], walking the dataset
        # files in random order without replacement.
        if name.split("_")[0] == "HGx9":
            file_order = np.random.permutation(np.arange(1, 101))
        else:
            # the fixed files first, then the rest in order if more showers are needed
            fixed_files = [2, 4, 7, 9, 11, 17, 20]
            file_order = fixed_files + [
                i for i in range(1, 101) if i not in fixed_files
            ]
        n_taken = 0
        if args.g4_file:
            with h5py.File(args.g4_file, "r") as fh:
                n_taken = min(shw, fh["events"].shape[0])
                real_showers[:n_taken] = fh["events"][:n_taken]
                inc_energies[:n_taken, 0] = fh["energy"][:n_taken].reshape(-1)
            file_order = []  # nothing to walk
        for file_i in tqdm(file_order):
            if n_taken == shw:
                break
            with h5py.File(dataset_path.format(file_i), "r") as fh:
                energy = fh["energy"][:].reshape(-1)
                idx = np.flatnonzero((energy > g4_e_min) & (energy < g4_e_max))
                if len(idx) > shw - n_taken:
                    idx = np.sort(np.random.choice(idx, shw - n_taken, replace=False))
                if len(idx) == 0:
                    continue
                real_showers[n_taken : n_taken + len(idx)] = fh["events"][idx]
                inc_energies[n_taken : n_taken + len(idx), 0] = energy[idx]
            n_taken += len(idx)
        if n_taken < shw:
            raise ValueError(
                f"Only {n_taken} Geant4 showers in [{g4_e_min}, {g4_e_max}] GeV "
                f"across the dataset files, but --shw {shw} requested; widen "
                "--g4-e-min/--g4-e-max or lower --shw."
            )
        print(
            f"Geant4: {n_taken} distinct showers"
            + (
                f" from {args.g4_file}"
                if args.g4_file
                else f" in [{g4_e_min}, {g4_e_max}] GeV"
            )
        )
        # sort real showers by number of points (fake ones are already sorted)
        idx_sorted_real = np.argsort(np.sum(real_showers[:, 3] > 0, axis=1))
        real_showers = real_showers[idx_sorted_real]
        inc_energies = inc_energies[idx_sorted_real]
        real_showers[:, 3] = real_showers[:, 3] * 1000
        real_showers[:, 1] = real_showers[:, 1] + 0.5
        real_showers[:, 1][real_showers[:, 3] == 0] = 0

    if compare_with_AllShowers and not allshowers_cached:
        print("loading AllShowers (bussthor) showers...")
        with h5py.File(allshowers_path, "r") as af:
            af_energies = af["energies"][:].reshape(-1)
            af_num_points = af["num_points"][:]
            af_mask = (af_energies > e_min) & (af_energies < e_max)
            af_matched_idx = np.where(af_mask)[0]
            n_allshowers = min(shw, len(af_matched_idx))
            af_idx = af_matched_idx[
                np.linspace(0, len(af_matched_idx) - 1, n_allshowers).astype(int)
            ]
            allshowers_inc_energies = af_energies[af_idx].reshape(-1, 1)
            max_points = int(af_num_points[af_idx].max())
            allshowers_showers = np.zeros((n_allshowers, 4, max_points))
            raw_showers = af["showers"]
            for k, idx in enumerate(tqdm(af_idx, desc="Loading AllShowers...")):
                n_pts = af_num_points[idx]
                hits = raw_showers[idx].reshape(-1, 4)[:n_pts]
                # bussthor's own (col0, col1) axis labels ("x", "z") do not match this
                # project's Geant4/ILD x/z convention: cross-checked against Geant4 cog
                # truth and bussthor's own observables/center_of_energy, this project's
                # x = -bussthor col1 and z = bussthor col0. projection expects [x, layer, z, e].
                allshowers_showers[k, 0, :n_pts] = -hits[:, 1]
                allshowers_showers[k, 1, :n_pts] = hits[:, 2]
                allshowers_showers[k, 2, :n_pts] = hits[:, 0]
                allshowers_showers[k, 3, :n_pts] = hits[:, 3]
        allshowers_showers[:, 3] = allshowers_showers[:, 3] * 1000  # GeV -> MeV
        allshowers_showers[:, 1] = allshowers_showers[:, 1] + 0.5
        allshowers_showers[:, 1][allshowers_showers[:, 3] == 0] = 0


def make_directories(name):
    folder = "./"
    os.makedirs(
        folder + "files/projected_array/" + name,
        exist_ok=True,
    )

    os.makedirs(
        folder + "files/projected_array/" + name + "/" + inc_en + str(shw),
        exist_ok=True,
    )
    save_dir_fake = folder + "files/projected_array/" + name + "/" + inc_en + str(shw)

    os.makedirs(
        folder + "files/projected_array/" + name + "/reco_" + inc_en + str(shw),
        exist_ok=True,
    )
    save_dir_reco = (
        folder + "files/projected_array/" + name + "/reco_" + inc_en + str(shw)
    )

    os.makedirs(
        folder + "files/projected_array/Geant4/",
        exist_ok=True,
    )
    os.makedirs(
        folder + "files/projected_array/Geant4/" + inc_en_g4 + str(shw),
        exist_ok=True,
    )
    save_dir_real = folder + "files/projected_array/Geant4/" + inc_en_g4 + str(shw)

    os.makedirs(
        folder
        + "files/projected_array/"
        + args.allshowers_cache
        + "/"
        + inc_en
        + str(shw),
        exist_ok=True,
    )
    save_dir_allshowers = (
        folder
        + "files/projected_array/"
        + args.allshowers_cache
        + "/"
        + inc_en
        + str(shw)
    )

    return (
        save_dir_real,
        save_dir_fake,
        save_dir_reco,
        save_dir_allshowers,
    )


(
    save_dir_real,
    save_dir_fake,
    save_dir_reco,
    save_dir_allshowers,
) = make_directories(name)

if load_dataset_and_project:
    if fake_SHOWERS_TO_SAVE:
        np.save(save_dir_fake + "/inc_en.npy", fake_inc_energies)
    if real_SHOWERS_TO_SAVE:
        np.save(save_dir_real + "/inc_en.npy", inc_energies)
    if reco_SHOWERS_TO_SAVE and compare_with_reco_showers:
        np.save(save_dir_reco + "/inc_en.npy", reco_inc_energies)
    if compare_with_AllShowers and not allshowers_cached:
        np.save(save_dir_allshowers + "/inc_en.npy", allshowers_inc_energies)

    showers, save_dir, inc_energy_ = [], [], []
    if both_to_save:
        print("saving both real and fake showers...")
        showers = (
            [fake_showers, real_showers, reco_showers]
            if compare_with_reco_showers
            else [fake_showers, real_showers]
        )
        inc_energy_ = (
            [fake_inc_energies, inc_energies, reco_inc_energies]
            if compare_with_reco_showers
            else [fake_inc_energies, inc_energies]
        )
        save_dir = (
            [save_dir_fake, save_dir_real, save_dir_reco]
            if compare_with_reco_showers
            else [save_dir_fake, save_dir_real]
        )
    else:
        if real_SHOWERS_TO_SAVE:
            showers.append(real_showers)
            save_dir.append(save_dir_real)
            inc_energy_.append(inc_energies)
            del real_showers
        if fake_SHOWERS_TO_SAVE:
            print("saving fake showers...")
            showers.append(fake_showers)
            save_dir.append(save_dir_fake)
            inc_energy_.append(fake_inc_energies)
            del fake_showers
    if reco_SHOWERS_TO_SAVE and compare_with_reco_showers:
        print("saving reco showers...")
        showers.append(reco_showers)
        save_dir.append(save_dir_reco)
        inc_energy_.append(reco_inc_energies)
        del reco_showers

    with_time_ = [with_time_flag] * len(showers)

    if compare_with_AllShowers and not allshowers_cached:
        print("saving AllShowers showers...")
        showers.append(allshowers_showers)
        save_dir.append(save_dir_allshowers)
        inc_energy_.append(allshowers_inc_energies)
        with_time_.append(False)  # AllShowers has no per-hit time information
        del allshowers_showers

    geant4_time_floor = None
    real_time_floor_accum = None
    if clip_time_to_geant4:
        if os.path.exists(geant4_time_floor_path):
            geant4_time_floor = np.load(geant4_time_floor_path)
            print(f"Loaded Geant4 time floor from {geant4_time_floor_path}")
        else:
            real_events_path = os.path.join(save_dir_real, f"events_{shw}.hdf5")
            real_events_has_time = False
            if os.path.exists(real_events_path):
                with h5py.File(real_events_path, "r") as f:
                    real_events_has_time = "projected_time_showers" in f
            if real_events_has_time:
                print(
                    f"No cached Geant4 time floor -- building it from existing "
                    f"{real_events_path} ..."
                )
                geant4_time_floor = build_geant4_time_floor_from_file(real_events_path)
                os.makedirs(os.path.dirname(geant4_time_floor_path), exist_ok=True)
                np.save(geant4_time_floor_path, geant4_time_floor)
                print(f"Saved Geant4 time floor to {geant4_time_floor_path}")
            elif not (real_SHOWERS_TO_SAVE and with_time_flag):
                print(
                    "WARNING: --clip-time-to-geant4 is set but no Geant4 time "
                    f"floor is available (no cached floor at "
                    f"{geant4_time_floor_path}, no existing real events hdf5 "
                    f"with time at {real_events_path}, and real showers are "
                    "not being (re)projected with timing this run) -- "
                    "generated showers are clipped to the causal floor only."
                )

    bs = 1000
    for gg in range(len(showers)):
        print(
            "projecting "
            + save_dir[gg].split("/")[-2]
            + " "
            + save_dir[gg].split("/")[-1]
            + " showers..."
        )
        is_real_group = clip_time_to_geant4 and save_dir[gg] == save_dir_real
        if showers[gg].shape[0] <= bs:
            bs = showers[gg].shape[0]

        n_gg = int(np.array(showers[gg]).shape[0])
        l0, l1, l2, l3 = (
            n_gg,
            int(x_bins.shape[0] - 1),
            int(y_bins.shape[0] - 1),
            int(z_bins.shape[0] - 1),
        )
        source_with_time = with_time_[gg]

        # A projected shower is a dense 180 x 78 x 180 grid (20 MB uncompressed); gzip-4 on
        # one-shower chunks is lossless and ~440x smaller, and h5py decompresses on read.
        h5_compression = dict(
            chunks=(1, l1, l2, l3), compression="gzip", compression_opts=4, shuffle=True
        )
        file = h5py.File(save_dir[gg] + "/events_" + str(l0) + ".hdf5", "w")
        events_group = file.create_dataset(
            "projected_showers", (l0, l1, l2, l3), dtype="f4", **h5_compression
        )
        energy_group = file.create_dataset("inc_energy", (l0, 1), dtype="f4")

        if source_with_time:
            events_time_group = file.create_dataset(
                "projected_time_showers", (l0, l1, l2, l3), dtype="f4", **h5_compression
            )

        for chunck in range(0, n_gg, bs):
            if (n_gg - chunck) < bs:
                bs = n_gg - chunck
            print(
                f"Writing to: events_goup[{chunck}:{chunck + bs}], shape: {events_group[chunck : chunck + bs].shape}"
            )

            events, events_time = projection_ecal_hcal(
                showers[gg][chunck : chunck + bs],
                x_bins,
                x_bins_hcal,
                y_bins,
                y_bins_ecal,
                y_bins_hcal,
                z_bins,
                z_bins_hcal,
                with_time=source_with_time,
            )
            events_group[chunck : chunck + bs] = np.ascontiguousarray(events).astype(
                "float32"
            )
            if source_with_time:
                if is_real_group and geant4_time_floor is None:
                    chunk_floor = compute_geant4_time_floor(events_time)
                    real_time_floor_accum = (
                        chunk_floor
                        if real_time_floor_accum is None
                        else np.fmin(real_time_floor_accum, chunk_floor)
                    )
                elif clip_time_to_geant4 and not is_real_group:
                    events_time = clip_time_to_geant4_floor(
                        events_time, geant4_time_floor
                    )
                events_time_group[chunck : chunck + bs] = np.ascontiguousarray(
                    events_time
                ).astype("float32")

            energy_group[chunck : chunck + bs] = np.ascontiguousarray(
                inc_energy_[gg][chunck : chunck + bs]
            ).astype("float32")
            del events, events_time
        file.close()

        if (
            is_real_group
            and geant4_time_floor is None
            and real_time_floor_accum is not None
        ):
            os.makedirs(os.path.dirname(geant4_time_floor_path), exist_ok=True)
            np.save(geant4_time_floor_path, real_time_floor_accum)
            geant4_time_floor = real_time_floor_accum
            print(f"Saved Geant4 time floor to {geant4_time_floor_path}")


def get_features(events, all=True, num_showers=20000, thr=0.25):
    occ = 0
    occ_list, occ_list_025 = [], []  # occupancy
    hits_list = []  # energy per cell
    e_layers_list, e_radial, e_sum_list = [], [], []
    e_sum_hcal_list, e_sum_ecal_list, e_radial_ecal, e_radial_hcal = [], [], [], []
    e_layers_ecal_list, e_layers_hcal_list = [], []
    hits_ecal_list, hits_hcal_list = [], []
    start_layer_list = []
    X, Z, Y, numb_active_hits_tot, numb_active_hits_ecal, numb_active_hits_hcal = (
        [],
        [],
        [],
        [],
        [],
        [],
    )

    if all:
        tot = np.array(events).shape[0]
    else:
        tot = int(num_showers)
    for j in tqdm(range(tot), mininterval=60, desc="Features..."):
        occ, occ_025, e_sum, e_sum_hcal, e_sum_ecal = 0, 0, 0, 0, 0
        my_shw = np.moveaxis(np.array(events[j]), 0, 1)
        numb_active_hits_tot.append(len(np.where(my_shw > 0)[0]))
        numb_active_hits_ecal.append(len(np.where(my_shw[:30] > 0)[0]))
        numb_active_hits_hcal.append(len(np.where(my_shw[30:] > 0)[0]))

        e_layers, e_layers_ecal, e_layers_hcal = [], [], []
        for ll, layer in enumerate(my_shw):
            # already in MeV
            layer[layer < thr] = 0
            hit_mask = layer > 0

            occ += hit_mask.sum()
            occ_025 += (layer > 0.25).sum()
            Y.append(np.repeat(ll, hit_mask.sum()))
            layer_hits = layer[hit_mask]
            e_sum += layer.sum()
            if ll < 30:
                e_sum_ecal += layer.sum()
            else:
                e_sum_hcal += layer.sum()

            if ll == 0:
                start_l, e_sum_old = ll, layer.sum()
            elif layer.sum() > e_sum_old:
                start_l, e_sum_old = ll, layer.sum()

            e_layers.append(layer.sum())
            hits_list.append(layer_hits)

            if ll > 29:
                e_layers_hcal.append(layer.sum())
                hits_hcal_list.append(layer_hits)
            else:
                e_layers_ecal.append(layer.sum())
                hits_ecal_list.append(layer_hits)

            # radial profile
            x_hit_idx, z_hit_idx = np.where(hit_mask)
            e_cell = layer[x_hit_idx, z_hit_idx]
            center_of_cell = 89.5  # grid centre, in cells
            dist_to_origin = np.sqrt(
                (x_hit_idx - center_of_cell) ** 2 + (z_hit_idx - center_of_cell) ** 2
            )
            X.append(x_hit_idx)
            Z.append(z_hit_idx)
            e_radial.append([dist_to_origin, e_cell])
            if ll < 30:
                e_radial_ecal.append([dist_to_origin, e_cell])
            else:
                e_radial_hcal.append([dist_to_origin, e_cell])
        start_layer_list.append(start_l)
        e_layers_list.append(e_layers)
        e_layers_ecal_list.append(e_layers_ecal)
        e_layers_hcal_list.append(e_layers_hcal)
        occ_list.append(occ)
        occ_list_025.append(occ_025)
        e_sum_list.append(e_sum)
        e_sum_ecal_list.append(e_sum_ecal)
        e_sum_hcal_list.append(e_sum_hcal)

    e_radial = np.concatenate(e_radial, axis=1)
    e_radial_ecal = np.concatenate(e_radial_ecal, axis=1)
    e_radial_hcal = np.concatenate(e_radial_hcal, axis=1)
    occ_list = np.array(occ_list)
    e_sum_list = np.array(e_sum_list)
    start_layer_list = np.array(start_layer_list)
    e_sum_ecal_list = np.array(e_sum_ecal_list)
    e_sum_hcal_list = np.array(e_sum_hcal_list)
    hits_list = np.concatenate(hits_list)
    hits_ecal_list = np.concatenate(hits_ecal_list)
    hits_hcal_list = np.concatenate(hits_hcal_list)
    e_layers_list = np.array(e_layers_list).sum(axis=0) / tot
    e_layers_ecal_list = np.array(e_layers_ecal_list).sum(axis=0) / tot
    e_layers_hcal_list = np.array(e_layers_hcal_list).sum(axis=0) / tot
    X = np.concatenate(X)
    Y = np.concatenate(Y)
    Z = np.concatenate(Z)
    return (
        e_radial,
        e_radial_ecal,
        e_radial_hcal,
        occ_list,
        occ_list_025,
        e_sum_list,
        e_sum_ecal_list,
        e_sum_hcal_list,
        hits_list,
        hits_ecal_list,
        hits_hcal_list,
        e_layers_list,
        e_layers_ecal_list,
        e_layers_hcal_list,
        X,
        Y,
        Z,
        numb_active_hits_tot,
        numb_active_hits_ecal,
        numb_active_hits_hcal,
        start_layer_list,
    )


def get_time_features(events, all=True, num_showers=20000, thr=200):
    time_hits_list = []
    time_layers_list, time_radial, time_sum_list = [], [], []
    time_sum_hcal_list, time_sum_ecal_list = [], []
    time_radial_ecal, time_radial_hcal = [], []
    time_hits_ecal_list, time_hits_hcal_list = [], []
    start_layer_list = []
    numb_active_time_hits_tot = []
    numb_active_time_hits_ecal, numb_active_time_hits_hcal = [], []

    gun_position = [0, 1804.7, 0]
    velocity, vel_per_layer_list = [], []
    n_unphysical_hits_per_shw = []
    n_unphysical_hits_per_layer_list = []

    if all:
        tot = np.array(events).shape[0]
    else:
        tot = int(num_showers)

    layer_bottom_pos = np.load("./files/geometry/layer_bottom_pos_ecal+hcal.npy")
    layer_top_pos = np.load("./files/geometry/layer_top_pos_ecal+hcal.npy")
    layer_mean_pos = (layer_top_pos + layer_bottom_pos) / 2

    for j in tqdm(range(tot), mininterval=60, desc="Time Features..."):
        time_sum, time_sum_hcal, time_sum_ecal = 0, 0, 0
        my_shw = np.moveaxis(np.array(events[j]), 0, 1)
        numb_active_time_hits_tot.append(len(np.where(my_shw > 0)[0]))
        numb_active_time_hits_ecal.append(len(np.where(my_shw[:30] > 0)[0]))
        numb_active_time_hits_hcal.append(len(np.where(my_shw[30:] > 0)[0]))

        time_layers = []
        vel_per_layer = []
        n_unphysical_hits_per_layer = []
        for ll, layer in enumerate(my_shw):
            layer[layer > thr] = 0
            hit_mask = layer > 0

            layer_hits = layer[hit_mask]
            time_sum += layer.sum()
            if ll < 30:
                c = c_eff("ecal")
                time_sum_ecal += layer.sum()
            else:
                c = c_eff("hcal")
                time_sum_hcal += layer.sum()

            # get start layer: the layer with the earliest hit (empty cells are 0)
            layer_min = layer_hits.min() if layer_hits.size else np.inf
            if ll == 0:
                start_l, min_time_old = ll, layer_min
            elif layer_min < min_time_old:
                start_l, min_time_old = ll, layer_min

            time_layers.append(layer.sum())
            time_hits_list.append(layer_hits)

            if ll > 29:
                time_hits_hcal_list.append(layer_hits)
            else:
                time_hits_ecal_list.append(layer_hits)

            x_hit_idx, z_hit_idx = np.where(hit_mask)

            # ecal has cell sizes of 5 mm while hcal of 30 mm
            if ll > 29:
                millimiters_per_cell = 30
            else:
                millimiters_per_cell = 5

            xz_mm = np.arange(-450, 450 + millimiters_per_cell, millimiters_per_cell)
            if ll > 29:  # hcal: first 75 cells are zeros from the projection
                # 75 = (180 ecal cells - 30 hcal cells) / 2
                xz_mm = np.concatenate([np.zeros((75)), xz_mm])

            if len(x_hit_idx) > 0:
                v = (
                    np.sqrt(
                        (layer_mean_pos[ll] - gun_position[1]) ** 2
                        + (xz_mm[x_hit_idx] - gun_position[0]) ** 2
                        + (xz_mm[z_hit_idx] - gun_position[2]) ** 2
                    )
                    / (layer[x_hit_idx, z_hit_idx])
                )
                velocity.append(v)
                vel_per_layer.append({str(ll): v})
                unphysical_hits = np.sum(v > c)
                n_unphysical_hits_per_layer.append(unphysical_hits)
            else:
                n_unphysical_hits_per_layer.append(0)

            # radial profile
            time_cell = layer[x_hit_idx, z_hit_idx]
            center_of_cell = 89.5  # grid centre, in cells
            dist_to_origin = np.sqrt(
                (x_hit_idx - center_of_cell) ** 2 + (z_hit_idx - center_of_cell) ** 2
            )
            time_radial.append([dist_to_origin, time_cell])
            if ll < 30:
                time_radial_ecal.append([dist_to_origin, time_cell])
            else:
                time_radial_hcal.append([dist_to_origin, time_cell])
        start_layer_list.append(start_l)
        time_layers_list.append(time_layers)
        time_sum_list.append(time_sum)
        time_sum_ecal_list.append(time_sum_ecal)
        time_sum_hcal_list.append(time_sum_hcal)
        vel_per_layer_list.append(vel_per_layer)

        n_unphysical_hits_per_layer_list.append(n_unphysical_hits_per_layer)
        n_unphysical_hits_per_shw.append(np.sum(np.array(n_unphysical_hits_per_layer)))

    time_radial = np.concatenate(time_radial, axis=1)
    time_radial_ecal = np.concatenate(time_radial_ecal, axis=1)
    time_radial_hcal = np.concatenate(time_radial_hcal, axis=1)
    time_sum_list = np.array(time_sum_list)
    start_layer_list = np.array(start_layer_list)
    time_sum_ecal_list = np.array(time_sum_ecal_list)
    time_sum_hcal_list = np.array(time_sum_hcal_list)
    time_hits_list = np.concatenate(time_hits_list)
    time_hits_ecal_list = np.concatenate(time_hits_ecal_list)
    time_hits_hcal_list = np.concatenate(time_hits_hcal_list)
    time_layers_list = np.array(time_layers_list).sum(axis=0)
    velocity = np.concatenate(velocity)
    n_unphysical_hits_per_shw = np.array(n_unphysical_hits_per_shw)
    n_unphysical_hits_per_layer = np.array(n_unphysical_hits_per_layer_list).sum(axis=0)

    # every hit's velocity, per layer (one array per layer; empty layers get [0.0])
    result_vel_per_layer = {}
    for vel_per_layer in vel_per_layer_list:
        for d in vel_per_layer:
            for k, v in d.items():
                result_vel_per_layer.setdefault(k, []).append(
                    np.asarray(v, dtype=float)
                )
    result_vel_per_layer = {
        k: np.concatenate(v) for k, v in result_vel_per_layer.items()
    }
    for ll in range(78):
        if str(ll) not in result_vel_per_layer:
            result_vel_per_layer[str(ll)] = [0.0]
    return (
        time_radial,
        time_radial_ecal,
        time_radial_hcal,
        time_sum_list,
        time_sum_ecal_list,
        time_sum_hcal_list,
        time_hits_list,
        time_hits_ecal_list,
        time_hits_hcal_list,
        time_layers_list,
        numb_active_time_hits_tot,
        numb_active_time_hits_ecal,
        numb_active_time_hits_hcal,
        start_layer_list,
        velocity,
        result_vel_per_layer,
        n_unphysical_hits_per_shw,
        n_unphysical_hits_per_layer,
    )


threshold = 1e-2
threshold_time = 200  # in ns


# h5py files opened by process_showers; kept referenced so the lazy datasets it
# returns stay readable for the rest of the run
_open_h5 = []


def compute_and_save_features(
    events,
    save_dir,
    prefix,
    names,
    num_showers,
    threshold,
    with_time=False,
    time_events=None,
    time_names=None,
    threshold_time=None,
):
    """Compute main and optional time features, then save them."""
    print(f"Computing features for {prefix} showers...")

    if with_time and time_events is not None:
        time_features = get_time_features(
            time_events, all=False, num_showers=num_showers, thr=threshold_time
        )
        for name, arr in zip(time_names, time_features):
            np.save(os.path.join(save_dir, f"{name}.npy"), arr)
        del time_features

    features = get_features(events, all=False, num_showers=num_showers, thr=threshold)
    for name, arr in zip(names, features):
        np.save(os.path.join(save_dir, f"{name}.npy"), arr)
    del features

    print(f"Features saved for {prefix}.")


def compute_and_save_cog(
    events, save_dir, prefix, num_showers, with_time=False, time_events=None
):
    """Compute and save center-of-gravity (COG) features."""
    print(f"Computing COG for {prefix} showers...")
    if num_showers > 3000:
        bs = 1000
        cogs = []
        with h5py.File(os.path.join(save_dir, f"events_{num_showers}.hdf5"), "r") as f:
            ev = f["projected_showers"]
            for j in tqdm(range(0, ev.shape[0], bs)):
                batch = ev[j : j + bs]
                cogs.append(get_cog2(batch.astype(np.float32)))
        my_list_cog = np.concatenate(cogs, axis=1)
    else:
        my_list_cog = get_cog2(np.asarray(events[:], dtype=np.float32))

    cog_names = [f"cog_{axis}" for axis in ("x", "y", "z")]
    for name, arr in zip(cog_names, my_list_cog):
        np.save(os.path.join(save_dir, f"{name}.npy"), arr)
    print(f"COG saved for {prefix}.")


def process_showers(
    prefix,
    save_dir,
    num_showers,
    threshold,
    with_time,
    threshold_time,
    f_names,
    time_names,
    load=True,
):
    """Load, compute features, and COG for a given shower type (fake/real/reco)."""
    if not load:
        return

    path = os.path.join(save_dir, f"events_{num_showers}.hdf5")
    # The feature loops read one shower at a time (events[j]), so hand them the h5py
    # datasets instead of loading the grids: loaded, 8k showers took ~160 GB of RAM
    # and 100k would need ~2 TB. h5py reads (and decompresses) each shower on demand.
    fh = h5py.File(path, "r")
    _open_h5.append(fh)
    events = fh["projected_showers"]
    events_time = fh["projected_time_showers"] if with_time else None

    compute_and_save_features(
        events,
        save_dir,
        prefix,
        f_names,
        num_showers,
        threshold,
        with_time,
        events_time,
        time_names,
        threshold_time,
    )
    compute_and_save_cog(events, save_dir, prefix, num_showers, with_time, events_time)
    return events


if compute_features:
    print("loading...")

    if real_SHOWERS_TO_SAVE:
        process_showers(
            "real",
            save_dir_real,
            shw,
            threshold,
            with_time_flag,
            threshold_time,
            f_names_r,
            time_names,
        )

    if fake_SHOWERS_TO_SAVE:
        process_showers(
            "fake",
            save_dir_fake,
            shw,
            threshold,
            with_time_flag,
            threshold_time,
            f_names,
            time_names,
        )

    if compare_with_reco_showers and reco_SHOWERS_TO_SAVE:
        _ = process_showers(
            "reco",
            save_dir_reco,
            shw,
            threshold,
            with_time_flag,
            threshold_time,
            f_names,
            time_names,
        )

    if compare_with_AllShowers and not allshowers_cached:
        _ = process_showers(
            "allshowers",
            save_dir_allshowers,
            n_allshowers,
            threshold,
            False,  # no time data for AllShowers
            threshold_time,
            f_names,
            time_names,
        )


print("SAVE DONE")
