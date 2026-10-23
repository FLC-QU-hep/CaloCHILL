import os
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.distributed as dist
from scipy.stats import binned_statistic
from torch.utils.data import Dataset
from utils.constants import (
    Xmax_photons,
    Xmax_pions,
    Xmin_photons,
    Xmin_pions,
    Ymax,
    Ymin,
    Zmax_photons,
    Zmax_pions,
    Zmin_photons,
    Zmin_pions,
    log_mean_photons,
    log_mean_pions,
    log_mean_time_photons,
    log_mean_time_pions,
    log_var_photons,
    log_var_pions,
    log_var_time_photons,
    log_var_time_pions,
    max_points_per_layer_photons,
    max_points_per_layer_pions,
    max_points_photons,
    max_points_pions,
    maximum_time,
)


def get_layers(PperL, pad_value_z):
    """
    PperL: np.ndarray of shape [B, L] with integer point counts
            PperL = np.array([
                [5, 3, 2, 0],   # Shower 0
                [1, 0, 4, 2],   # Shower 1
            ])  # shape: [B, L]
    Returns: np.ndarray of shape [B, max_N, 1] with padded layer indices per point
    """
    B, L = PperL.shape
    flat_counts = PperL.flatten()  # [B * L]

    # Create layer indices
    layer_ids = np.tile(np.arange(L), B)  # [B * L]

    # Repeat indices according to point counts
    layer_idx = np.repeat(layer_ids, flat_counts)  # [total_points]

    # Split layer indices per shower
    split_sizes = PperL.sum(axis=1)  # [B]
    layer_idx_per_shower = np.split(layer_idx, np.cumsum(split_sizes)[:-1])

    # Pad each shower's layer indices to max length
    max_N = max(len(x) for x in layer_idx_per_shower)
    padded = np.full((B, max_N), fill_value=pad_value_z, dtype=int)
    for i, arr in enumerate(layer_idx_per_shower):
        padded[i, : len(arr)] = arr

    if padded is None:
        print(layer_idx)
        print(max_N)
        print(padded)
        print(type(padded))
    else:
        return padded[..., np.newaxis]  # Shape: [B, max_N, 1]


def logE_and_padding_sampled_showers(
    cfg_data,
    shower,
    dataset,
    padding=None,
    mint_per_layer=None,
    en_idx=3,
):
    if cfg_data.log_energy:
        if dataset == "pions":
            shower[:, :, en_idx] *= cfg_data.log_var_pions
            shower[:, :, en_idx] += cfg_data.log_mean_pions
        elif dataset == "photons":
            shower[:, :, en_idx] *= cfg_data.log_var_photons
            shower[:, :, en_idx] += cfg_data.log_mean_photons
        shower[:, :, en_idx] = np.exp(shower[:, :, en_idx])

    if padding is not None:
        feats = shower.shape[2]
        for i in range(feats):
            shower[:, :, i][padding] = 0
    return shower


def _calculate_points_per_layer(events, axis=1):
    points_per_layer = []
    min_range, max_range, bins = 0, 78, 78

    for event in events:
        num_p, _ = np.histogram(
            event[:, axis][event[:, 3] > 0], bins=bins, range=(min_range, max_range)
        )
        points_per_layer.append(num_p)
    return np.array(points_per_layer)


def get_cmpr_showers_preprocessing(
    events, energies, n_points, XZmin=-450, XZmax=450, Ymin=0, Ymax=78
):
    max_num_points = n_points.max()
    ev = []
    for event in range(events.shape[0]):
        ev.append(events[event][:, -int(max_num_points) :])
    events = np.array(ev)

    padding_mask = np.zeros((events.shape[0], events.shape[-1])).astype(bool)
    col_indices = np.flip(np.arange(events.shape[-1]))
    padding_mask[col_indices >= np.array(n_points)[:, None]] = True

    events[:, 3, :] *= 1000  # energy scale
    events[:, 0, :] = (events[:, 0, :] - XZmin) * 2 / (
        XZmax - XZmin
    ) - 1  # x normalization
    events[:, 2, :] = (events[:, 2, :] - XZmin) * 2 / (
        XZmax - XZmin
    ) - 1  # z normalization

    events = np.moveaxis(events, -1, -2)
    events[events[:, :, 3] == 0] = 0

    points_per_layer = _calculate_points_per_layer(events)
    max_PperL = 400
    events = events[:, :, [0, 1, 2, 3]]
    events[:, :, 1][events[:, :, 3] <= 0] = 100
    sort_idx = np.argsort(events[:, :, 1], axis=1)

    batch_indices = np.arange(events.shape[0])[:, None]  # Shape: [B, 1], will broadcast
    events = events[batch_indices, sort_idx]
    y_cond = get_layers(points_per_layer.astype(int))  # B, max_N, 1
    y_cond = (y_cond - Ymin) * 2 / (Ymax - Ymin) - 1
    points_per_layer = points_per_layer / max_PperL
    return {
        "event": events,
        "energy": energies,
        "points_per_layer": points_per_layer,
        "points_per_layer_max": np.zeros(events.shape[0])
        + max_PperL,  # when using Dataloader this is automatic
        "n_points": n_points,
        "cond_ecal": np.zeros(events.shape[0]),
        "y_cond": y_cond,
        "padding_mask": padding_mask[batch_indices, sort_idx],
    }


def sort_by_layer_and_radius(
    events,
    layer_cond,
    pad_value_layer,
    x_idx=0,
    y_idx=1,
    layer_idx=2,
):
    """
    Args:
        events: array, shape (B, N, D)
        layer_cond: array, shape (B, N, 1) or (B, N)
        pad_value_z: value used for padded z entries
        x_idx, y_idx: indices of x and y coordinates

    Returns:
        events_sorted
        layer_cond_sorted
    """

    B, N, _ = events.shape

    if layer_cond.ndim == 3:
        layer_ids = layer_cond.squeeze(-1)
    else:
        layer_ids = layer_cond

    x = events[:, :, x_idx]
    y = events[:, :, y_idx]
    radius = np.sqrt(x**2 + y**2)
    is_padded = events[:, :, layer_idx] == pad_value_layer

    layer_ids = np.where(is_padded, np.inf, layer_ids)
    radius = np.where(is_padded, np.inf, radius)

    sorted_idx = np.empty((B, N), dtype=np.int64)

    for b in range(B):
        # Primary: layer, Secondary: radius
        sorted_idx[b] = np.lexsort(
            (
                radius[b],  # secondary key
                layer_ids[b],  # primary key
            )
        )

    batch_idx = np.arange(B)[:, None]
    events_sorted = events[batch_idx, sorted_idx]
    layer_cond_sorted = layer_cond[batch_idx, sorted_idx]

    return events_sorted, layer_cond_sorted


def _calculate_min_time_per_layer(
    events,
    layer_axis=1,
    bins=78,
    time_axis=4,
):
    time_per_layer = []
    for event in events:
        mask = event[:, 3] > 0
        vals = event[:, layer_axis][mask]  # binning axis
        times = event[:, time_axis][mask]  # "time" values

        t_p, _, _ = binned_statistic(
            vals, times, statistic="min", bins=bins, range=(0, bins)
        )
        t_p = np.nan_to_num(t_p, nan=-1.0)  # Replace NaNs with -1
        time_per_layer.append(t_p)
    return np.array(time_per_layer)


def get_preprocessing_events(
    events, Xmin, Xmax, Zmin, Zmax, layers_axis=1, include_time=True
):
    sort_energy_idx = np.argsort(events[:, :, 3], axis=1)
    batch_indices = np.arange(events.shape[0])[:, None]  # Shape: [B, 1], will broadcast
    events = events[batch_indices, sort_energy_idx]
    max_len = (events[:, :, 3] > 0).sum(axis=1).max()
    events = events[:, -(max_len):]
    coord = 2 if layers_axis == 1 else 1
    events[:, :, 0] = (events[:, :, 0] - Xmin) * 2 / (Xmax - Xmin) - 1
    events[:, :, coord] = (events[:, :, coord] - Zmin) * 2 / (Zmax - Zmin) - 1

    events[events[:, :, 3] == 0] = 0
    points_per_layer = _calculate_points_per_layer(events, axis=layers_axis)
    mint_per_layer = None
    if include_time:
        mint_per_layer = _calculate_min_time_per_layer(events, layer_axis=layers_axis)
    return events, points_per_layer, mint_per_layer


def get_sorted_events_layercond(
    events,
    pad_value_layer,
    points_per_layer,
    max_n_points=5376,
    y_idx=2,
    layer_idx=1,
):
    # sorting by layer axis
    events[:, :, layer_idx][events[:, :, 3] <= 0] = pad_value_layer
    sort_idx = np.argsort(events[:, :, layer_idx], axis=1)
    batch_indices = np.arange(events.shape[0])[:, None]
    events = events[batch_indices, sort_idx]
    # layer conditioning
    layer_cond = get_layers(
        points_per_layer.astype(int), pad_value_layer
    )  # B, max_N, 1

    # sort by radius within the layer
    events, layer_cond = sort_by_layer_and_radius(
        events,
        layer_cond,
        pad_value_layer,
        x_idx=0,
        y_idx=y_idx,
        layer_idx=layer_idx,
    )
    valid_energy_mask = events[:, :, 3] > 0
    n_points = (valid_energy_mask).sum(axis=1)[:, None] / max_n_points
    padding_mask = ~valid_energy_mask
    return (
        torch.Tensor(events),
        torch.Tensor(layer_cond),
        torch.Tensor(n_points),
        torch.Tensor(padding_mask),
    )


class PionClouds(Dataset):
    def __init__(
        self,
        files_path,
        n_files=None,
        tot_files=100,
        bs=32,
        phase="train",
        net="CaloHadronic",
        model_name="CaloHadronic",
        pad_value_layer=100,
        include_time=True,
        cfg=None,
    ):
        self.include_time = include_time
        self.Ymin, self.Ymax = Ymin, Ymax
        if "pions" in files_path:
            self.dataset = "pions"
            self.Xmin, self.Xmax = Xmin_pions, Xmax_pions
            self.Zmin, self.Zmax = Zmin_pions, Zmax_pions
            self.max_points_per_layer = max_points_per_layer_pions
            self.max_n_points = max_points_pions
        if "photons" in files_path:
            self.dataset = "photons"
            self.Xmin, self.Xmax = Xmin_photons, Xmax_photons
            self.Zmin, self.Zmax = Zmin_photons, Zmax_photons
            self.max_points_per_layer = max_points_per_layer_photons
            self.max_n_points = max_points_photons
        self.bs = bs
        self.phase = phase
        self.n_files = n_files
        self.tot_files = tot_files
        self.net = net
        self.model_name = model_name
        self.pad_value_layer = pad_value_layer
        if cfg is not None:
            self.cfg = cfg

        self.n_files_in_the_folder = len(os.listdir(Path(files_path).parent))

        if self.n_files is None:
            files_to_load = range(1, self.n_files_in_the_folder + 1)
        else:
            if self.phase == "train":
                from_file = 0
            elif self.phase == "val":
                from_file = int(self.tot_files - self.n_files * 2)
            elif self.phase == "test":
                from_file = int(self.tot_files - self.n_files)
            files_to_load = range(from_file, from_file + self.n_files)

        self.data_files = []
        self.cached_energy = {}  # small: one float per event, kept in RAM
        self._h5_handles = {}  # lazy per-worker file handles, never pickled

        is_ddp = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if is_ddp else 0

        # Read only n_points + energy metadata — event arrays are loaded on demand.
        print(f"[Rank {rank}] Scanning dataset metadata...")
        index_list = self._load_files(files_path, files_to_load)

        if is_ddp:
            dist.barrier()

        self.index_list = sorted(index_list, key=lambda x: x[0])
        print(f"[Rank {rank}] Dataset ready: {len(self.index_list)} events")

    def _load_files(self, files_path, files_to_load):
        """Read only metadata (n_points + energy). Event arrays are loaded on demand."""
        index_list = []
        for local_idx, file_idx in enumerate(files_to_load):
            path = files_path.format(file_idx + 1)
            self.data_files.append(path)
            with h5py.File(path, "r") as f:
                self.cached_energy[local_idx] = np.array(f["energy"])  # N floats, tiny
                n_points = f["n_points"][:]
                for i, n_point in enumerate(n_points):
                    if n_point < 20:
                        continue
                    index_list.append((n_point, local_idx, i))
        return index_list

    def _get_h5_events(self, local_idx, event_indices):
        """Return event rows for a given file, using a cached file handle per worker."""
        if local_idx not in self._h5_handles:
            self._h5_handles[local_idx] = h5py.File(self.data_files[local_idx], "r")
        f = self._h5_handles[local_idx]
        rows = np.array(event_indices)
        # h5py fancy indexing requires strictly increasing indices
        sort_order = np.argsort(rows)
        data = np.moveaxis(f["events"][rows[sort_order]], -1, -2)
        # restore original order so callers get rows in the requested sequence
        return data[np.argsort(sort_order)]

    def __getstate__(self):
        """Exclude open h5 handles from pickling so DataLoader workers start clean."""
        state = self.__dict__.copy()
        state["_h5_handles"] = {}
        return state

    def _get_batch(self, idx):
        idx_range = self._determine_idx_range(idx)  # (min_range, max_range)

        entries = [self.index_list[i] for i in range(idx_range[0], idx_range[1])]

        # Group by file to open each h5 handle at most once per batch
        from collections import defaultdict

        file_groups = defaultdict(list)
        for pos, (n_point, file_idx, event_idx) in enumerate(entries):
            file_groups[file_idx].append((pos, event_idx))

        raw_events = [None] * len(entries)
        for file_idx, items in file_groups.items():
            event_indices = [ei for _, ei in items]
            batch_events = self._get_h5_events(file_idx, event_indices)
            for k, (pos, _) in enumerate(items):
                raw_events[pos] = batch_events[k]

        events = []
        energy = []
        n_points = []
        for pos, (n_point, file_idx, event_idx) in enumerate(entries):
            event_data = raw_events[pos]
            energy_val = self.cached_energy[file_idx][event_idx]
            n_points.append((event_data[:, 3] > 0).sum())
            events.append(event_data)
            energy.append(energy_val)

        max_points_in_batch = max(event.shape[0] for event in events)
        padded_events = []
        for event in events:
            current_points = event.shape[0]
            if current_points < max_points_in_batch:
                pad = np.zeros(
                    (max_points_in_batch - current_points, event.shape[1]),
                    dtype=event.dtype,
                )
                event = np.vstack([event, pad])
            elif current_points > max_points_in_batch:
                # truncate if longer than max_points
                event = event[:max_points_in_batch]
            padded_events.append(event)

        events = np.array(padded_events)
        energy = np.array(energy)
        n_points = np.array(n_points)
        return events, energy, n_points

    def _determine_idx_range(self, idx):
        # non-overlapping batches: idx is the batch index, not the event index
        start = idx * self.bs
        end = min(start + self.bs, len(self.index_list))
        return (start, end)

    def get_preprocessing_events(
        self, events, Xmin, Xmax, Zmin, Zmax, layers_axis=1, include_time=True
    ):
        sort_energy_idx = np.argsort(events[:, :, 3], axis=1)
        batch_indices = np.arange(events.shape[0])[
            :, None
        ]  # Shape: [B, 1], will broadcast
        events = events[batch_indices, sort_energy_idx]
        max_len = (events[:, :, 3] > 0).sum(axis=1).max()
        events = events[:, -(max_len):]
        events[:, :, 0] = (events[:, :, 0] - Xmin) * 2 / (Xmax - Xmin) - 1
        events[:, :, 2] = (events[:, :, 2] - Zmin) * 2 / (Zmax - Zmin) - 1

        events[events[:, :, 3] == 0] = 0
        points_per_layer = _calculate_points_per_layer(events, axis=layers_axis)
        mint_per_layer = None
        if include_time:
            mint_per_layer = _calculate_min_time_per_layer(
                events, layer_axis=layers_axis
            )
        return events, points_per_layer, mint_per_layer

    def get_sorted_events_layercond(
        self,
        events,
        pad_value_layer,
        points_per_layer,
        max_n_points=5376,
        layer_idx=1,
    ):
        # sorting by layer axis
        events[:, :, layer_idx][events[:, :, 3] <= 0] = pad_value_layer
        sort_idx = np.argsort(events[:, :, layer_idx], axis=1)
        batch_indices = np.arange(events.shape[0])[:, None]
        events = events[batch_indices, sort_idx]
        # layer conditioning
        layer_cond = get_layers(
            points_per_layer.astype(int), pad_value_layer
        )  # B, max_N, 1

        # sort by radius within the layer
        events, layer_cond = sort_by_layer_and_radius(
            events,
            layer_cond,
            pad_value_layer,
            x_idx=0,
            y_idx=2,
            layer_idx=layer_idx,
        )
        valid_energy_mask = events[:, :, 3] > 0
        n_points = (valid_energy_mask).sum(axis=1)[:, None] / max_n_points
        padding_mask = ~valid_energy_mask
        return (
            torch.Tensor(events),
            torch.Tensor(layer_cond),
            torch.Tensor(n_points),
            torch.Tensor(padding_mask),
        )

    def __getitem__(self, idx):
        events, energy, n_points = self._get_batch(idx)
        events[:, :, 3] *= 1000  # energy scale

        events, points_per_layer, mint_per_layer = self.get_preprocessing_events(
            events,
            self.Xmin,
            self.Xmax,
            self.Zmin,
            self.Zmax,
            layers_axis=1,
            include_time=self.include_time,
        )

        if self.net == "CaloHadronic" and self.model_name == "CaloHadronic":
            layer_cond = np.array([0])  # not used

            # smear on the propagation axis
            smearing = np.random.uniform(-0.49, 0.49, size=events[:, :, 1].shape)
            events[:, :, 2] = events[:, :, 2] + smearing

            events[:, :, 2] = (events[:, :, 2] - self.Zmin) * 2 / (
                self.Zmax - self.Zmin
            ) - 1  # layer normalization
            events = torch.Tensor(events[:, :, [0, 1, 2, 3]])  # x, y, z, energy
            valid_energy_mask = events[:, :, 3] > 0
            n_points = (valid_energy_mask).sum(axis=1)[:, None] / self.max_n_points
            padding_mask = ~valid_energy_mask

            if self.cfg.data.norm_cond:
                energy = torch.Tensor(
                    energy / 100 * 2 - 1
                )  # assume max incident energy: 100 GeV

            cond_feats = torch.cat([energy, torch.Tensor(points_per_layer)], -1)

            for j in range(events.shape[0]):
                events[j, :, 3][padding_mask[j].bool()] = 0

            if self.cfg.data.log_energy:
                events[:, :, 3] = torch.log(events[:, :, 3] + 1e-20)
                events[:, :, 3] = events[:, :, 3] - self.cfg.data.log_mean
                events[:, :, 3] = events[:, :, 3] / self.cfg.data.log_var

        else:  # Compressor or CaloHadronicVAE
            events, layer_cond, n_points, padding_mask = (
                self.get_sorted_events_layercond(
                    events,
                    self.pad_value_layer,
                    points_per_layer,
                    # legacy CaloHadronic (Oct 2025) was trained with n_hits / 6000
                    6000 if self.net == "CaloHadronic" else self.max_n_points,
                )
            )
            # assume max incident energy: 100 GeV
            energy = torch.Tensor(energy / 100 * 2 - 1)
            if energy.dim() == 1:
                energy = energy.unsqueeze(-1)

            cond_feats = torch.cat([energy, n_points], -1)
            if self.include_time and self.net == "CaloHadronic":
                # legacy CaloHadronic: time relative to the per-layer minimum, applied
                # (as trained) only to hits whose normalised log-energy is > 0
                log_e = (
                    torch.log(events[:, :, 3] + 1e-20) - log_mean_pions
                ) / log_var_pions
                b, n = torch.nonzero(log_e > 0, as_tuple=True)
                events[b, n, 4] -= torch.Tensor(mint_per_layer)[
                    b, events[b, n, 1].long()
                ]
            if self.include_time:
                mint_per_layer = mint_per_layer / maximum_time

            # log energy and time transformations
            events[:, :, 3] = torch.log(events[:, :, 3] + 1e-20)
            if self.include_time:
                events[:, :, 4] = torch.log(events[:, :, 4] + 1e-20)
            if self.dataset == "pions":
                events[:, :, 3] = events[:, :, 3] - log_mean_pions
                events[:, :, 3] = events[:, :, 3] / log_var_pions
                if self.include_time:
                    events[:, :, 4] = events[:, :, 4] - log_mean_time_pions
                    events[:, :, 4] = events[:, :, 4] / log_var_time_pions
            elif self.dataset == "photons":
                events[:, :, 3] = events[:, :, 3] - log_mean_photons
                events[:, :, 3] = events[:, :, 3] / log_var_photons
                if self.include_time:
                    events[:, :, 4] = events[:, :, 4] - log_mean_time_photons
                    events[:, :, 4] = events[:, :, 4] / log_var_time_photons

            for j in range(events.shape[0]):
                events[j, :, 3][padding_mask[j].bool()] = 0
                if self.include_time:
                    events[j, :, 4][padding_mask[j].bool()] = 0
                layer_cond[j, :, 0][padding_mask[j].bool()] = 0

            # x, z, logE (+ logT); without time the model sees input_feats = 3
            events = events[:, :, [0, 2, 3, 4] if self.include_time else [0, 2, 3]]
        # per-layer minimum time (conditioning); without time it is a dummy tensor kept
        # for the default collate, never read since consumers gate on include_time
        if mint_per_layer is None:
            mint_per_layer = np.zeros_like(points_per_layer, dtype=np.float32)
        return {
            "events": events.to(torch.float32),
            "cond_feats": cond_feats.to(torch.float32),
            "padding_mask": padding_mask.to(torch.bool),
            "layer_cond": layer_cond.to(torch.long),
            "points_per_layer": torch.Tensor(points_per_layer).to(torch.float32),
            "mint_per_layer": torch.Tensor(mint_per_layer).to(torch.float32),
        }

    def __len__(self):
        return -(-len(self.index_list) // self.bs)  # ceil division
