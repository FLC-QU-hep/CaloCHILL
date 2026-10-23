"""Shared test setup: synthetic hdf5 showers and the configs as training.py adjusts them."""

import os
import sys
import tempfile

import h5py
import numpy as np
import torch
import torch.nn.attention.flex_attention as flex

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

from models.CaloHadronic import CompressorVAE  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from utils.dataset import PionClouds  # noqa: E402
from utils.misc import Config  # noqa: E402

DEVICE = "cpu"
# torch.compile is a GPU speed-up here; inductor's CPU flex attention fails to lower
# and falls back to eager after a minute of compiling, so run everything eagerly.
torch._dynamo.config.disable = True
# Eager flex attention: call the reference implementation directly (it otherwise
# wraps itself in torch.compile), and let it run backward on CPU, which torch >= 2.13
# refuses for want of a fused CPU kernel. The reference gradients match SDPA to 1e-15.
flex._FLEX_ATTENTION_DISABLE_COMPILE_DEBUG = True
_validate_device = flex._validate_device
flex._validate_device = lambda q, k, v: _validate_device(
    q.detach(), k.detach(), v.detach()
)
N_FILES = 3
SHOWERS_PER_FILE = 32


def write_synthetic_files(folder, n_files=N_FILES, n_showers=SHOWERS_PER_FILE, seed=0):
    """Files shaped like the real ones: events (N, 5, 6000) with rows x, layer, z,
    energy [GeV], time [ns], hits right-aligned and time-sorted; n_points (N,);
    energy (N, 1) in GeV. Returns the path pattern ("{}" = file index from 1)."""
    rng = np.random.default_rng(seed)
    pattern = os.path.join(folder, "synthetic_pions_{}.hdf5")
    for i in range(1, n_files + 1):
        events = np.zeros((n_showers, 5, 6000), dtype=np.float32)
        n_points = rng.integers(150, 400, size=n_showers)
        for s, n in enumerate(n_points):
            # cell centres of the real 0.848 mm grid, around the shower axis
            x = 0.3175 + 0.848 * np.round(rng.normal(0, 40, n))
            z = 0.3175 + 0.848 * np.round(rng.normal(0, 40, n))
            hits = np.stack(
                [
                    np.clip(x, -445, 445),
                    rng.integers(0, 78, n),
                    np.clip(z, -445, 445),
                    np.clip(rng.lognormal(np.log(1.5e-4), 1.0, n), 1e-6, 0.05),
                    np.sort(rng.uniform(0.02, 200.0, n)),
                ]
            )
            events[s, :, -n:] = hits
        with h5py.File(pattern.format(i), "w") as f:
            f["events"] = events
            f["n_points"] = n_points.astype(np.float32)
            f["energy"] = rng.uniform(10, 90, (n_showers, 1)).astype(np.float32)
    return pattern


def load_configs(model_config, dataset_path):
    """cfg_data and cfg_model as load_and_adjust_configs() (--dataset pions --test)
    leaves them, on CPU and with the synthetic files."""
    cfg_data = Config.from_yaml(os.path.join(REPO, "configs/configs_data.yaml"))
    for field in (
        "log_mean",
        "log_var",
        "max_points",
        "max_points_per_layer",
        "dataset_path",
    ):
        setattr(cfg_data, field, getattr(cfg_data, f"{field}_pions"))
    cfg_data.dataset_path = dataset_path
    cfg_data.dataloader.total_files = N_FILES
    cfg_data.dataloader.training_files = 1
    cfg_data.dataloader.workers = 0

    cfg_model = Config.from_yaml(os.path.join(REPO, "configs", model_config))
    cfg_model.device = DEVICE
    for block in ("vae", "transformer"):
        if hasattr(cfg_model, block):
            getattr(cfg_model, block).device = DEVICE
    cfg_model.common.train_bs = 5  # 4 showers per batch
    cfg_model.log_comet = False
    cfg_model.multi_gpu = False
    return cfg_data, cfg_model


def build_vae(cfg_data, cfg_vae):
    """CompressorVAE as Train.get_model() builds it (cfg_vae = compressor config)."""
    cfg_vae.vae.calorimeter_layers = cfg_data.calorimeter_layers
    cfg_vae.vae.p_dim = cfg_vae.vae.p_dim_pions
    return CompressorVAE(cfg_data, cfg_vae.vae)


def make_dataset(cfg_data, cfg_model, net, phase="train", n_files=1):
    """The training dataset of initialize_dataloader()."""
    return PionClouds(
        files_path=cfg_data.dataset_path,
        tot_files=cfg_data.dataloader.total_files,
        bs=cfg_model.common.train_bs - 1,
        net=net,
        model_name=cfg_model.model.model_name,
        include_time=cfg_model.common.include_time,
        n_files=n_files,
        phase=phase,
    )


def make_loader(dset):
    return DataLoader(dset, batch_size=1, num_workers=0, shuffle=False)


def first_batch(loader):
    """One batch as Train.forward() unpacks it (drops the loader's batch dim)."""
    batch = next(iter(loader))
    return {k: v[0].to(DEVICE) for k, v in batch.items()}


def layer_kwargs(batch):
    """The block-mask kwargs Train.forward() passes to get_loss()."""
    ppl = batch["points_per_layer"]
    return {"n_points": ppl.sum(dim=1), "top_points_layers": ppl.topk(2)[0].sum(dim=1)}


class SyntheticData:
    """Mixin: synthetic files in a temp dir for the lifetime of the test class."""

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls._tmp = tempfile.TemporaryDirectory()
        cls.dataset_path = write_synthetic_files(cls._tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()
