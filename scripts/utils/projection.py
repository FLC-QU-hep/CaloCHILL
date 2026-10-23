import matplotlib as mpl
import numpy as np
from scipy.stats import binned_statistic_dd
from tqdm import tqdm

mpl.rcParams["xtick.labelsize"] = 25
mpl.rcParams["ytick.labelsize"] = 25
mpl.rcParams["font.size"] = 35
mpl.rcParams["legend.frameon"] = False
mpl.rcParams["text.usetex"] = True
mpl.rcParams["font.family"] = "serif"

layer_bottom_pos = np.load("./files/geometry/layer_bottom_pos_ecal+hcal.npy")
layer_top_pos = np.load("./files/geometry/layer_top_pos_ecal+hcal.npy")

Xmin, Xmax = -450, 450
Zmin, Zmax = -450, 450
half_cell_size_ecal = 5.0883331298828125 / 2
half_cell_size_hcal = 5.0883331298828125 / 2 * 12
cell_thickness_ecal = layer_top_pos[0] - layer_bottom_pos[0]  # 0.52502441406250
cell_thickness_hcal = layer_top_pos[-1] - layer_bottom_pos[-1]  # 3


def projection(showers, with_time=False):
    events = []
    energy = []
    for i in tqdm(range(showers.shape[0]), mininterval=5, desc="Projecting..."):
        one_shw = np.moveaxis(showers[i], 0, 1)
        H, my_edges = np.histogramdd(
            one_shw[:, :3], bins=[30, 30, 78], weights=one_shw[:, 3]
        )
        events.append(H)
        del H, my_edges, one_shw
    return events, energy


def merge_ecal_and_hcal(H_ecal, H_hcal):
    zeros_pad = int((H_ecal.shape[0] - H_hcal.shape[0]) / 2)
    H_hcal = np.append(
        H_hcal, np.zeros((zeros_pad, H_hcal.shape[1], H_hcal.shape[2])), axis=0
    )
    H_hcal = np.append(
        np.zeros((zeros_pad, H_hcal.shape[1], H_hcal.shape[2])), H_hcal, axis=0
    )
    H_hcal = np.append(
        H_hcal, np.zeros((H_hcal.shape[0], H_hcal.shape[1], zeros_pad)), axis=2
    )
    H_hcal = np.append(
        np.zeros((H_hcal.shape[0], H_hcal.shape[1], zeros_pad)), H_hcal, axis=2
    )
    H = np.concatenate([H_ecal, H_hcal], axis=1)
    return H


def projection_ecal_hcal(
    showers,
    x_bins,
    x_bins_hcal,
    y_bins,
    y_bins_ecal,
    y_bins_hcal,
    z_bins,
    z_bins_hcal,
    with_time=False,
):
    s = int(x_bins.shape[0] - 1)
    ys, _, _ = (
        int(y_bins.shape[0] - 1),
        int(y_bins_ecal.shape[0] - 1),
        int(y_bins_hcal.shape[0] - 1),
    )
    events = np.zeros(((showers.shape[0], s, ys, s)))
    events_time = np.zeros(((showers.shape[0], s, ys, s)))

    for i in range(showers.shape[0]):
        one_shw = np.moveaxis(showers[i], 0, 1)
        ecal_shw = one_shw[one_shw[:, 1] < 30, :]
        hcal_shw = one_shw[one_shw[:, 1] >= 30, :]

        # ECAL projection
        random_shift_ecal = np.random.uniform(0, 5)
        H_ecal, _ = np.histogramdd(
            ecal_shw[:, :3],
            bins=[x_bins + random_shift_ecal, y_bins_ecal, z_bins + random_shift_ecal],
            weights=ecal_shw[:, 3],
        )

        # HCAL projection
        random_shift_hcal = np.random.uniform(0, 30)
        H_hcal, _ = np.histogramdd(
            hcal_shw[:, :3],
            bins=[
                x_bins_hcal + random_shift_hcal,
                y_bins_hcal,
                z_bins_hcal + random_shift_hcal,
            ],
            weights=hcal_shw[:, 3],
        )

        # merge ECAL and HCAL
        H = merge_ecal_and_hcal(H_ecal, H_hcal)
        events[i] = H
        del H, H_ecal, H_hcal

        if with_time:
            if ecal_shw[:, :3].shape[0] == 0:
                H_time_ecal = np.full(
                    (x_bins, y_bins, z_bins), np.nan
                )  # or +inf or 0 depending on convention
            else:
                H_time_ecal, _, _ = binned_statistic_dd(
                    ecal_shw[:, :3],
                    ecal_shw[:, 4],
                    statistic="min",
                    bins=[
                        x_bins + random_shift_ecal,
                        y_bins_ecal,
                        z_bins + random_shift_ecal,
                    ],
                )
            if hcal_shw[:, :3].shape[0] == 0:
                H_time_hcal = np.full(
                    (len(x_bins_hcal) + 1, len(y_bins_hcal) - 1, len(z_bins_hcal) + 1),
                    np.nan,
                )  # or +inf or 0 depending on convention
            else:
                H_time_hcal, _, _ = binned_statistic_dd(
                    hcal_shw[:, :3],
                    hcal_shw[:, 4],
                    statistic="min",
                    bins=[
                        x_bins_hcal + random_shift_hcal,
                        y_bins_hcal,
                        z_bins_hcal + random_shift_hcal,
                    ],
                )
            H_time = merge_ecal_and_hcal(H_time_ecal, H_time_hcal)
            events_time[i] = H_time
            del H_time, H_time_ecal, H_time_hcal

        del one_shw, ecal_shw, hcal_shw
    del showers
    return events, events_time
