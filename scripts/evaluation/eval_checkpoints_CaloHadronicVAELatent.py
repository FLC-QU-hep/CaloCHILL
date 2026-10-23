import argparse
import os
import sys

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import h5py
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from utils import gen_utils

torch.set_float32_matmul_precision(
    "high"
)  # TF32; used to be set on importing gen_utils


def setup_ddp(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def load_models_ddp(cfg, cfg_data, cfg_compressor, edm_dir, rank, _with_sf=True):
    local_device = torch.device(f"cuda:{rank}")
    cfg.device = local_device
    if _with_sf:
        # clone of https://github.com/FLC-QU-hep/LayerFM, weights from https://huggingface.co/FLC-QU-hep/LayerFM
        pcfm_root = os.environ.get("LAYERFM_ROOT", "../LayerFM")
        pcfm_ckpt = os.environ.get(
            "LAYERFM_CKPT",
            pcfm_root + "/results/20260619_124529_LayerFM_PandT_/checkpoint_distill.pt",
        )
        if pcfm_root not in sys.path:
            sys.path.insert(0, pcfm_root)
        from layerfm.inference_cond_file import load_from_checkpoint

        # distilled one-step student (NFE = 1)
        flow_dist = load_from_checkpoint(pcfm_ckpt, distill=True)
        flow_dist.eval()
    else:
        flow_dist = None

    model, model_vae = gen_utils.load_models(
        cfg, cfg_data, cfg_compressor, edm_dir, local_device
    )
    model = DDP(model, device_ids=[rank])
    model = torch.compile(model, dynamic=False)
    model_vae = DDP(model_vae, device_ids=[rank])
    model_vae = torch.compile(model_vae, dynamic=False)

    return model, model_vae, flow_dist


def main(cfg, dataset):
    # single node only
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    setup_ddp(local_rank, world_size)
    local_device = torch.device(f"cuda:{local_rank}")
    model, model_vae, flow_dist = load_models_ddp(
        cfg, cfg_data, cfg_compressor, edm_dir, local_rank, _with_sf
    )
    cfg.device = local_device
    return model, model_vae, flow_dist, cfg


parser = argparse.ArgumentParser()
parser.add_argument(
    "--inc_en",
    type=int,
    default=0,
    help="[5, 50, 500, 0]: 0 means all range from 1 to 1000 GeV, otherwise fixed energy in GeV",
)
parser.add_argument(
    "--num",
    "--n_showers",  # torchrun >= 2.9 rejects --num (ambiguous with its --numa-binding)
    type=int,
    default=100,
    help="total number of generated events",
)
parser.add_argument(
    "--fixed_energy",
    action="store_true",
    help="condition every generated shower on exactly --inc_en GeV (as a fixed-energy "
    "Geant4 sample) instead of the energies of the dataset showers within +-0.1 GeV; "
    "those showers, taken within +-0.5 GeV, are then only used for the reco",
)
parser.add_argument(
    "--num_steps",
    type=int,
    default=8,
    help="solver steps. NOT the NFE: the midpoint sampler used below spends 2 denoiser "
    "evaluations per step, so NFE = 2 * num_steps (default 8 steps = NFE 16)",
)
parser.add_argument("--bs", type=int, default=32, help="batch size")
parser.add_argument(
    "--energy_calibration_factor",
    type=float,
    nargs="+",
    default=[1.0],
    help="Rescales the incident energy condition fed to the diffusion transformer "
    "at inference (see configs_LatentDiffusion.yaml); 1.0 = no calibration",
)
parser.add_argument(
    "--energy_calibration_powers",
    type=float,
    nargs="+",
    default=None,
    help="powers of E (GeV) for the coefficients of --energy_calibration_factor; "
    "default with two coefficients: 0 -1 (a + b / E)",
)
parser.add_argument(
    "--calib",
    action="store_true",
    help="Cache energy_calibration_factor per incident-energy range in a txt file "
    "(files/generated_showers/<gen_folder>/calib_<inc_en>GeV.txt, gen_folder being "
    "'<latentdiff_dir>_<num_steps>'). If the file already exists, its factor is loaded "
    "and used instead of --energy_calibration_factor; otherwise --energy_calibration_factor "
    "is used and saved there for future runs.",
)
parser.add_argument(
    "--latentdiff_dir",
    type=str,
    default="lmf_large_ds256/",  # large LMF, d_s = 256 (paper)
    help="Name of the LatentDiff checkpoint directory under logs/ "
    "(must end with '/') to load the EDM/LatentDiff model from.",
)
parser.add_argument(
    "--save_latents",
    type=str,
    default=None,
    help="Write z_encoded / z_sampled and their per-shower labels to this .npz path.",
)
parser.add_argument(
    "--real_file",
    type=str,
    default=None,
    help="HGx9-format hdf5 (events, energy) whose first --num showers in the energy "
    "range are encoded and decoded for the reco, each once, instead of --num draws "
    "with replacement from test file 99.",
)
parser.add_argument(
    "--g4_conditioning",
    action="store_true",
    help="No LayerFM: per-layer point counts, earliest times and incident energy taken "
    "from the Geant4 showers (--real_file), as AllShowers does for its event displays.",
)
parser.add_argument(
    "--no_save",
    action="store_true",
    help="Do not write the generated showers hdf5 (e.g. for a plots-only run).",
)
args = parser.parse_args()

inc_en = args.inc_en
num = args.num
bs = args.bs

dataset = "pions"  # "pions" or "photons"
print("dataset: ", dataset)
print("incident energy: ", inc_en, "GeV")

# min and max energy of the generated events
if inc_en == 0:
    energy_range = [10, 90]
elif args.fixed_energy:
    # real showers for the reco only; +-0.5 GeV so that 10 GeV (dataset starts at
    # 10.14) still finds some
    energy_range = [inc_en - 0.5, inc_en + 0.5]
else:
    energy_range = [
        inc_en - 0.1,
        inc_en + 0.1,
    ]  # fixed energy with a small range to add some variability

n_files = 1
_with_sf = not args.g4_conditioning  # False: conditioning straight from Geant4
path_to_repo = "./"

caloclouds = "edm"  # 'ddpm, 'edm', 'cm'
edm_dir = gen_utils.resolve_edm_dir(args.latentdiff_dir)

# TF32 matmuls (precision "high"), not the timing script's bf16 "medium"
cfg, cfg_data, cfg_compressor = gen_utils.load_configs(edm_dir, "cuda", "high", dataset)

gen_folder = edm_dir.split("/")[-2] + "_" + str(args.num_steps)

energy_calibration_factor = (
    args.energy_calibration_factor
)  # [f], or [a, b] for a + b / E
energy_calibration_factor = (
    energy_calibration_factor[0]
    if len(energy_calibration_factor) == 1
    else energy_calibration_factor
)
if args.calib:
    calib_dir = path_to_repo + "files/generated_showers/" + gen_folder + "/"
    calib_file = calib_dir + f"calib_{inc_en}GeV.txt"
    os.makedirs(calib_dir, exist_ok=True)
    if os.path.exists(calib_file):
        with open(calib_file) as fh:
            energy_calibration_factor = float(fh.read().strip())
        print(
            f"loaded cached calibration factor for {inc_en} GeV: {energy_calibration_factor}"
        )
    else:
        # no fit yet for this energy range: generate uncalibrated showers now, then run
        # compute_energy_calibration.py on them to fit and save a factor to calib_file,
        # and rerun this script to pick it up and produce calibrated showers
        energy_calibration_factor = 1.0
        print("showers not calibrated yet!")
cfg.energy_calibration_factor = energy_calibration_factor
cfg.energy_calibration_powers = args.energy_calibration_powers
configs = cfg
print("PointCount FM: ", _with_sf)

float_type = "f4"  # float32: the network and the input data are float32 anyway
# Zero padding dominates these arrays, so gzip shrinks them ~10x at no cost on read.
_gz = dict(compression="gzip", compression_opts=4)
cfg.num_steps = args.num_steps
if getattr(cfg.model, "generative_model", "edm") == "mean_flow":
    # CaloHadronic_FM.sample reads mean_flow.num_steps, not cfg.num_steps / cfg.sampler
    # (one network evaluation per step, so NFE = num_steps; 1 = one-step sampling).
    cfg.mean_flow.num_steps = args.num_steps
cfg.sampler = "midpoint"
print(gen_folder)

iterations = 1  # number of iterations for timing
print("num", num, "bs", bs)
print("steps: ", cfg.num_steps)
print("incident energy range: ", energy_range)
n_scaling = True

os.makedirs(
    path_to_repo + "files/generated_showers/" + gen_folder,
    exist_ok=True,
)

model, model_vae, distribution, cfg = main(cfg, dataset)

(
    fake_showers,
    reco_showers,
    _,
    cond_E,
    cond_E_r,
    real_showers,
    _,
    z_encoded,
    z_sampled,
) = gen_utils.gen_showers_batch_CaloHadronicVAELatent(
    model,
    model_vae,
    distribution,
    energy_range[0],
    energy_range[1],
    config_data=cfg_data,
    config=cfg,
    config_compressor=cfg_compressor,
    num=num,
    dataset=dataset,
    bs=bs,
    enable_pointcount=_with_sf,
    cond_E=torch.full((num, 1), inc_en / 100 * 2 - 1) if args.fixed_energy else None,
    # the VAE checkpoint decides the feature layout, not this script
    with_time=cfg_compressor.vae.include_time,
    real_file=args.real_file,
)
# cond_E/cond_E_r come back normalised to [-1, 1]; convert to GeV to match the
# gen_energy/reco_energy convention the saved hdf5 already used for PointCountFM2
# runs (relied on by occ-scale.py and compute_energy_calibration.py).
cond_E = torch.tensor(cond_E, dtype=torch.float32)
cond_E = (cond_E + 1) / 2 * 100
cond_E_r = torch.tensor(cond_E_r, dtype=torch.float32)
cond_E_r = (cond_E_r + 1) / 2 * 100

if args.save_latents:
    os.makedirs(os.path.dirname(os.path.abspath(args.save_latents)), exist_ok=True)
    np.savez(
        args.save_latents,
        z_encoded=z_encoded.astype(np.float32),
        z_sampled=z_sampled.astype(np.float32),
        energy_enc=cond_E_r.numpy().reshape(-1),
        energy_samp=cond_E.numpy().reshape(-1),
        npoints_enc=(real_showers[:, :, 3] > 0).sum(axis=1),
        npoints_samp=(fake_showers[:, :, 3] > 0).sum(axis=1),
        layer_enc=real_showers[:, :, 2].astype(np.uint8),
        layer_samp=fake_showers[:, :, 2].astype(np.uint8),
    )
    print(f"saved latents: {args.save_latents}")

if args.no_save:
    print("--no_save: not writing the generated showers")
    sys.exit(0)

print("saving files...")

if not _with_sf:
    inc_en = str(inc_en) + "noSF_"

calib_tag = (
    "_calib" + "_".join(map(str, np.atleast_1d(cfg.energy_calibration_factor)))
    if cfg.energy_calibration_factor != 1.0
    else ""
)

names = ["gen", "reco"]
showers = [fake_showers, reco_showers]
inc_energies = [cond_E, cond_E_r]

file = h5py.File(
    path_to_repo
    + "files/generated_showers/"
    + gen_folder
    + "/showers_"
    + str(inc_en)
    + "GeV_"
    + str(num)
    + calib_tag
    + ".hdf5",
    "w",
)

for j in range(len(names)):
    if showers[j] is None:
        continue
    print(file, showers[j].shape, inc_energies[j].shape)
    n_hits_max, f = (
        int(showers[j].shape[1]),
        int(showers[j].shape[2]),
    )  # f = 4 for noTime VAEs
    events_goup = file.create_dataset(
        names[j] + "_events",
        (num, n_hits_max, f),
        maxshape=(None, n_hits_max, f),
        dtype=float_type,
        chunks=(min(32, num), n_hits_max, f),
        **_gz,
    )
    energy_goup = file.create_dataset(
        names[j] + "_energy", (num, 1), maxshape=(None, 1), dtype=float_type
    )
    file[names[j] + "_events"].write_direct(
        np.ascontiguousarray(showers[j], dtype=float_type)
    )
    file[names[j] + "_energy"].write_direct(
        np.ascontiguousarray(inc_energies[j], dtype=float_type)
    )
file.close()
