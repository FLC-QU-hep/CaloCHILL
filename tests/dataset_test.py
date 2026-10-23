import unittest

import h5py
import numpy as np
import torch
from helpers import SHOWERS_PER_FILE, SyntheticData, load_configs, make_dataset
from utils.dataset import logE_and_padding_sampled_showers


class DatasetTest(SyntheticData, unittest.TestCase):
    def check_batch(self, item, n_feats, n_norm):
        events, pad = item["events"], item["padding_mask"]
        B, N, D = events.shape
        self.assertEqual((B, D), (4, n_feats))
        self.assertEqual(pad.shape, (B, N))
        self.assertEqual(pad.dtype, torch.bool)
        self.assertEqual(item["cond_feats"].shape, (B, 2))
        self.assertEqual(item["layer_cond"].shape, (B, N, 1))
        self.assertEqual(item["points_per_layer"].shape, (B, 78))
        self.assertEqual(item["mint_per_layer"].shape, (B, 78))
        for v in item.values():
            self.assertTrue(torch.isfinite(v.float()).all())

        n_valid = (~pad).sum(1)
        self.assertEqual(N, int(n_valid.max()))  # padded to the batch maximum only
        self.assertTrue((n_valid >= 20).all())
        self.assertTrue((events[pad] == 0).all())
        self.assertTrue((item["layer_cond"][pad] == 0).all())
        lc = item["layer_cond"][~pad]
        self.assertTrue(((lc >= 0) & (lc < 78)).all())
        torch.testing.assert_close(item["points_per_layer"].sum(1), n_valid.float())
        torch.testing.assert_close(item["cond_feats"][:, 1], n_valid / n_norm)
        energy = item["cond_feats"][:, 0]
        self.assertTrue(((energy > -1) & (energy < 1)).all())
        valid = events[~pad]
        self.assertTrue((valid[:, :2].abs() <= 1).all())  # x, z in [-1, 1]

    def test_compressor_batches(self):
        cfg_data, cfg_model = load_configs("configs_compressor.yaml", self.dataset_path)
        dset = make_dataset(cfg_data, cfg_model, "Compressor")
        self.assertEqual(len(dset), SHOWERS_PER_FILE // 4)
        with h5py.File(self.dataset_path.format(1)) as f:
            n_points = np.sort(f["n_points"][:])
        for i in range(len(dset)):
            item = dset[i]
            self.check_batch(item, n_feats=4, n_norm=cfg_data.max_points)
            # index_list is sorted by hit count, so batch i holds showers 4i..4i+3
            n_valid = (~item["padding_mask"]).sum(1).numpy()
            np.testing.assert_array_equal(n_valid, n_points[4 * i : 4 * i + 4])

        # inverse of the logE preprocessing gives back GeV * 1000 hit energies
        item = dset[0]
        shower = logE_and_padding_sampled_showers(
            cfg_data,
            item["events"].numpy().copy(),
            "pions",
            padding=item["padding_mask"].numpy(),
            en_idx=2,
        )
        e = shower[:, :, 2][~item["padding_mask"].numpy()]
        self.assertTrue(((e >= 1e-3 * 0.99) & (e <= 50 * 1.01)).all())

    def test_val_test_split(self):
        cfg_data, cfg_model = load_configs("configs_compressor.yaml", self.dataset_path)
        for phase in ("val", "test"):
            dset = make_dataset(cfg_data, cfg_model, "Compressor", phase=phase)
            self.assertEqual(len(dset.data_files), 1)
            self.check_batch(dset[0], n_feats=4, n_norm=cfg_data.max_points)

    def test_calohadronic_legacy_batches(self):
        cfg_data, cfg_model = load_configs(
            "configs_CaloHadronic.yaml", self.dataset_path
        )
        dset = make_dataset(cfg_data, cfg_model, "CaloHadronic")
        # the legacy model was trained with n_hits / 6000
        self.check_batch(dset[0], n_feats=4, n_norm=6000)

    def test_without_time(self):
        cfg_data, cfg_model = load_configs("configs_compressor.yaml", self.dataset_path)
        cfg_model.common.include_time = False
        item = make_dataset(cfg_data, cfg_model, "Compressor")[0]
        self.check_batch(item, n_feats=3, n_norm=cfg_data.max_points)
        self.assertTrue((item["mint_per_layer"] == 0).all())


if __name__ == "__main__":
    unittest.main()
