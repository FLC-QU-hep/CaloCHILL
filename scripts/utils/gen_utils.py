import os
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from models.CaloHadronic import CaloHadronic_Attention, CaloHadronic_FM, CompressorVAE
from tqdm import tqdm
from utils.constants import (
    Xmax_photons,
    Xmax_pions,
    Xmin_photons,
    Xmin_pions,
    Zmax_photons,
    Zmax_pions,
    Zmin_photons,
    Zmin_pions,
    max_points_photons,
    max_points_pions,
    maximum_time,
)
from utils.dataset import (
    _calculate_min_time_per_layer,
    _calculate_points_per_layer,
    get_layers,
    get_preprocessing_events,
    get_sorted_events_layercond,
)
from utils.misc import Config, load_checkpoint, load_model_state

plt.close("all")
plt.rcdefaults()
plt.style.use("seaborn-v0_8")
plt.rcParams.update(
    {
        "font.size": 20,  # default text size
        "axes.titlesize": 20,  # title size
        "axes.labelsize": 23,  # x and y label size
        "xtick.labelsize": 22,  # x tick label size
        "ytick.labelsize": 22,  # y tick label size
        "legend.fontsize": 22,  # legend size
        "figure.titlesize": 30,  # figure title size
    }
)


def resolve_edm_dir(run):
    """LMF run folder name under logs/, or an absolute path."""
    d = run if os.path.isabs(run) else "./logs/" + run
    return d.rstrip("/") + "/"


def load_configs(edm_dir, device, matmul_precision, dataset="pions"):
    """The LMF, data and VAE compressor configs of an LMF run, wired together the way
    the models expect them. (run from the repo root: vae_path is relative to logs/)"""
    cfg = Config.from_yaml(edm_dir + "configs_LatentDiffusion.yaml")
    cfg.energy_calibration_factor = 1.0
    cfg_data = Config.from_yaml(edm_dir + "configs_data.yaml")
    cfg_compressor = Config.from_yaml(
        "logs/" + os.path.dirname(cfg.model.vae_path) + "/configs_compressor.yaml"
    )

    cfg.device = cfg_compressor.device = cfg_compressor.vae.device = device
    cfg_compressor.multi_gpu = False
    # read by the models' __init__ on cuda (torch.set_float32_matmul_precision)
    cfg.matmul_precision = cfg_compressor.vae.matmul_precision = matmul_precision

    cfg.transformer.input_feats = int(
        cfg_compressor.vae.z_dim * cfg_compressor.vae.n_layers
    )
    cfg.transformer.calorimeter_layers = cfg_compressor.vae.calorimeter_layers = (
        cfg_data.calorimeter_layers
    )
    cfg_compressor.vae.p_dim = cfg.downsample_factor = (
        cfg_compressor.vae.p_dim_photons
        if dataset == "photons"
        else cfg_compressor.vae.p_dim_pions
    )
    cfg.calorimeter_layers = 78

    for field in (
        "log_mean",
        "log_var",
        "max_points",
        "max_points_per_layer",
        "dataloader.total_files",
        "dataloader.training_files",
        "dataset_path",
    ):
        *parents, attr = field.split(".")
        obj = cfg_data
        for p in parents:
            obj = getattr(obj, p)
        setattr(obj, attr, getattr(obj, f"{attr}_{dataset}"))

    return cfg, cfg_data, cfg_compressor


def _match_compiled(state_dict, model):
    """Checkpoints are saved from torch.compile'd models ('_orig_mod.' in the keys);
    on CPU the models are not compiled, so drop the prefix when the model has none."""
    if any("_orig_mod." in k for k in model.state_dict()):
        return state_dict
    return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}


def load_models(cfg, cfg_data, cfg_compressor, edm_dir, device):
    """The LMF (ckpt_latest.pt of edm_dir) and its VAE compressor, in eval mode."""
    # 'edm' (CaloHadronic_Attention) unless the run's config says mean_flow.
    if getattr(cfg.model, "generative_model", "edm") == "mean_flow":
        model = CaloHadronic_FM(cfg, "LatentDiff").to(device)
    else:
        model = CaloHadronic_Attention(cfg, "LatentDiff").to(device)
    checkpoint = torch.load(
        edm_dir + "ckpt_latest.pt",
        map_location=torch.device(device),
        weights_only=False,
    )
    model.load_state_dict(_match_compiled(checkpoint["others"]["model"], model))
    model.eval()

    model_vae = CompressorVAE(cfg_data, cfg_compressor.vae).to(device)
    new_state_dict, _, _ = load_checkpoint(
        cfg_data, cfg_compressor, model_vae, cfg.model.vae_path
    )
    model_vae = load_model_state(model_vae, _match_compiled(new_state_dict, model_vae))
    model_vae.eval()
    return model, model_vae


def get_shower(
    model,
    vae,
    num_points,
    energy,
    cond_Nperlayer=None,
    cond_mint_perlayer=None,
    cond_Ylayer=None,
    cond_T=None,
    bs=1,
    cond_N=None,
    pm=None,
    config=None,
    clock_dict=None,
):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    # Each rank handles a slice of the batch
    assert bs % world_size == 0, "total_bs must be divisible by world_size"
    local_bs = bs // world_size
    local_device = torch.device(f"cuda:{rank}")

    p = cond_N[rank * local_bs : (rank + 1) * local_bs].to(local_device)
    e = torch.ones((local_bs, 1), device=local_device) * energy
    e = e[rank * local_bs : (rank + 1) * local_bs]
    mint_per_layer = (
        cond_mint_perlayer[rank * local_bs : (rank + 1) * local_bs].to(local_device)
        if cond_mint_perlayer is not None
        else None
    )
    cond_feats = torch.cat([e, p], -1)
    if pm is not None:
        local_pm = pm[rank * local_bs : (rank + 1) * local_bs]
    with torch.no_grad():
        if cond_Ylayer is not None:
            local_cond_Ylayer = cond_Ylayer[rank * local_bs : (rank + 1) * local_bs].to(
                local_device
            )
            local_latent = model.module.sample(
                cond_feats,
                num_points,
                config,
                y=local_cond_Ylayer.to(torch.long),
                mint_per_layer=mint_per_layer,
                padding_mask=torch.Tensor(local_pm).to(local_device).bool(),
            )
            if getattr(model.module, "latent_target", "sample") == "mu":
                # model trained on the posterior mean (latent_target: mu): map the
                # sample back to the VAE decompressor's input
                local_latent = model.module.latent_to_decoder(local_latent)
            local_shower, _ = vae.module.sample(
                z=local_latent,
                N=local_cond_Ylayer.shape[1],
                c=cond_feats,
                y=local_cond_Ylayer.to(torch.long),
                mint_per_layer=mint_per_layer,
                padding=torch.Tensor(local_pm).to(local_device).bool(),
                device=local_device,
            )
    gather_list = (
        [torch.zeros_like(local_shower) for _ in range(world_size)]
        if rank == 0
        else None
    )
    gather_latent = (
        [torch.zeros_like(local_latent) for _ in range(world_size)]
        if rank == 0
        else None
    )
    dist.gather(local_shower, gather_list, dst=0)
    dist.gather(local_latent, gather_latent, dst=0)
    if rank == 0:
        full_shower = torch.cat(gather_list, dim=0)  # (total_bs, num_points, feat_dim)
        full_latent = torch.cat(gather_latent, dim=0)  # (total_bs, latent_dim)
        return full_latent, full_shower
    return None  # non-rank-0 processes return None


def get_reco_shower(
    model,
    showers,
    energy,
    max_points=3000,
    config_data=None,
    config=None,
    XZmin=-450,
    XZmax=450,
    scale_energy=False,
):
    if scale_energy:
        showers = showers.copy()
        showers[:, :, 3] *= 1000

    showers, points_per_layer, mint_per_layer = get_preprocessing_events(
        showers,
        XZmin,
        XZmax,
        XZmin,
        XZmax,
        layers_axis=2,
    )

    showers, layer_cond, n_points, padding_mask = get_sorted_events_layercond(
        showers,
        100,
        points_per_layer,
        max_points,
        y_idx=1,
        layer_idx=2,
    )

    showers = showers.to(config.device).float()
    e = torch.from_numpy(energy / 100 * 2 - 1).unsqueeze(1)

    cond_feats = torch.cat([e, n_points], -1)
    if config_data.log_energy:
        showers[:, :, 3] = torch.log(showers[:, :, 3] + 1e-20)
        showers[:, :, 3] = showers[:, :, 3] - config_data.log_mean
        showers[:, :, 3] = showers[:, :, 3] / config_data.log_var

    # the VAE itself is authoritative about whether it was trained with time
    include_time = getattr(model.module, "include_time", True)

    if include_time:
        showers[:, :, 4] = torch.log(showers[:, :, 4] + 1e-20)
        showers[:, :, 4] = showers[:, :, 4] - config_data.log_mean_time
        showers[:, :, 4] = showers[:, :, 4] / config_data.log_var_time

    for j in range(showers.shape[0]):
        showers[j, :, 3][padding_mask[j].bool()] = 0
        if include_time:
            showers[j, :, 4][padding_mask[j].bool()] = 0
        layer_cond[j, :, 0][padding_mask[j].bool()] = 0

    showers = showers[:, :, [0, 1, 3, 4] if include_time else [0, 1, 3]]
    mint_t = None
    if include_time:
        mint_t = (
            torch.from_numpy(mint_per_layer / maximum_time)
            .to(torch.float32)
            .to(config.device)
        )
    with torch.no_grad():
        reco_showers, z_reco = model.module.reco(
            showers.to(torch.float32).to(config.device),
            c=cond_feats.to(torch.float32).to(config.device),
            y=layer_cond.to(torch.long).to(config.device),
            mint_per_layer=mint_t,
            padding=padding_mask.to(torch.bool).to(config.device),
        )
        for j in range(showers.shape[0]):
            # energy is index 2 in both layouts (x, z, logE[, logT])
            reco_showers[j, :, 2][padding_mask[j].bool()] = 0
    return (
        reco_showers,
        layer_cond,
        padding_mask.bool().cpu().numpy(),
        z_reco[:, 0, :, :],
    )


def x_y_shift(showers, Xmin, Xmax, Ymin, Ymax):
    showers[:, :, 0] = (showers[:, :, 0] + 1) / 2
    showers[:, :, 1] = (showers[:, :, 1] + 1) / 2

    showers[:, :, 0] = showers[:, :, 0] * (Xmax - Xmin) + Xmin
    showers[:, :, 1] = showers[:, :, 1] * (Ymax - Ymin) + Ymin
    return showers


def gen_showers_batch_CaloHadronicVAELatent(
    model,
    vae,
    pointcount_model,
    e_min,
    e_max,
    config_data,
    config,
    config_compressor,
    num=2000,
    bs=8,
    dataset="photons",
    enable_pointcount=True,
    real_showers=None,
    cond_E=None,
    clock_dict=None,
    with_time=None,
    real_file=None,
):
    if with_time is None:
        # the VAE checkpoint decides the feature layout
        with_time = config_compressor.vae.include_time

    if dataset == "photons":
        Xmin, Xmax, Zmin, Zmax = Xmin_photons, Xmax_photons, Zmin_photons, Zmax_photons
        config_data.log_var, config_data.log_mean = (
            config_data.log_var_photons,
            config_data.log_mean_photons,
        )
        config_data.dataset_path = config_data.dataset_path_photons
    else:
        Xmin, Xmax, Zmin, Zmax = Xmin_pions, Xmax_pions, Zmin_pions, Zmax_pions
        config_data.log_var, config_data.log_mean = (
            config_data.log_var_pions,
            config_data.log_mean_pions,
        )
        config_data.dataset_path = config_data.dataset_path_pions

    if clock_dict is None:
        clock_dict_ = None

    if real_showers is None:
        # real_file: take its first `num` showers in [e_min, e_max] in order, each
        # once; otherwise draw `num` from test file 99 with replacement
        f = h5py.File(real_file or config_data.dataset_path.format(99), "r")
        events = np.array(f["events"])
        energy = np.array(f["energy"])[:, 0]
        events = np.moveaxis(events, -1, -2)
        events[:, :, [1, 2]] = events[:, :, [2, 1]]
        n_points = (events[:, :, 3] > 0).sum(axis=1)
        energy_mask = (energy.reshape(-1) > e_min) & (energy.reshape(-1) < e_max)
        if real_file:
            idx = np.arange(num)
        else:
            idx = np.sort(np.random.rand(1, num) * energy_mask.sum()).astype(int)[0]
        idx_sorted = np.argsort(n_points[energy_mask][idx])
        n_points = n_points[energy_mask][idx][idx_sorted]
        real_showers = events[energy_mask][idx][idx_sorted]
        e = energy[energy_mask][idx][idx_sorted]
        cond_E_real = torch.Tensor(e).to(config.device)
        real_showers[:, :, 3] *= 1000

    if not enable_pointcount or (cond_E is None):
        cond_E = cond_E_real.unsqueeze(1) / 100 * 2 - 1
        cond_E_r = cond_E.clone()
    else:
        cond_E_r = cond_E_real.unsqueeze(1) / 100 * 2 - 1

    real_showers[real_showers[:, :, 3] == 0] = 0
    samples_points_r = _calculate_points_per_layer(real_showers, axis=2).astype(int)
    samples_mint_r = _calculate_min_time_per_layer(
        real_showers,
        layer_axis=2,
        bins=78,
        time_axis=4,
    )
    samples_mint_r = samples_mint_r / maximum_time

    if clock_dict is not None:
        clock_dict_sf, clock_dict_ = clock_dict

    cond_E = cond_E.to(config.device)

    if not enable_pointcount:
        samples_points, samples_mint = samples_points_r, samples_mint_r
    else:
        # _CheckpointSampler: condition = raw energy in GeV (CPU), returns [num_points, min_time_ns]
        e_gev = ((cond_E + 1) / 2 * 100).cpu()
        pcfm_bs = 1024
        pts_list, mint_list = [], []
        start = time.time()
        for _i in range(0, e_gev.shape[0], pcfm_bs):
            batch_e = e_gev[_i : _i + pcfm_bs]
            with torch.no_grad():
                res = pointcount_model(batch_e)
            pts_list.append(np.round(res[0].numpy()).clip(0).astype(int))
            if with_time and len(res) > 1:
                mint_list.append(res[1].numpy() / maximum_time)
        samples_points = np.concatenate(pts_list, axis=0)
        samples_mint = np.concatenate(mint_list, axis=0) if mint_list else None
        tf = time.time() - start
        print(
            f"PointCountFM inference done in {tf:.2f}s  ({tf / e_gev.shape[0] * 1000:.1f} ms/shower)"
        )
    # check if the sum of samples along axis 1 is zero
    while len(np.where(samples_points.sum(axis=1) == 0)[0]) > 0:
        print("Warning: some samples have total number of points = zero")
        idx = np.where(samples_points.sum(axis=1) == 0)[0]
        print(f"Resampling {len(idx)} zero-hit showers...")
        with torch.no_grad():
            res = pointcount_model(e_gev[idx])
        samples_points[idx] = np.round(res[0].numpy()).clip(0).astype(int)
        if with_time and samples_mint is not None:
            samples_mint[idx] = res[1].numpy() / maximum_time
    # this is a control on the number of points per layer, and the total number of points
    p = 0
    # limits: max n points per shower = 5370, max n points per layer: 412
    max_points_layer = (
        config_data.max_points_per_layer_photons
        if dataset == "photons"
        else config_data.max_points_per_layer_pions
    )
    max_points = max_points_photons if dataset == "photons" else max_points_pions

    while (len(np.where(samples_points.sum(axis=1) > max_points)[0]) > 0) | (
        len(np.where(samples_points > max_points_layer)[0]) > 0
    ):
        indexes = np.unique(
            np.concatenate(
                [
                    np.where(samples_points.sum(axis=1) > max_points)[0],
                    np.where(samples_points > max_points_layer)[0],
                ]
            )
        )
        for i in indexes:
            with torch.no_grad():
                res = pointcount_model(e_gev[i : i + 1])
            samples_points[i, :] = np.round(res[0][0].numpy()).clip(0).astype(int)
            if with_time and samples_mint is not None:
                samples_mint[i] = res[1][0].numpy() / maximum_time
            p += 1
            if p > 10:
                break

    samples = samples_points.astype(int)
    mmaxx = samples_points.sum(axis=1).max()

    # ordering per number of hits
    idx_hits_sorted = np.argsort(samples.sum(axis=1))

    # conditioning
    hits_per_layer_all_tot = samples_points[idx_hits_sorted]
    cond_E = cond_E[idx_hits_sorted]
    if with_time:
        samples_mint = samples_mint[idx_hits_sorted]

    fake_showers_list, reco_showers_list = [], []
    z_encoded_list, z_sampled_list = [], []
    max_shower_length = mmaxx + 1
    max_shower_length_r = samples_points_r.sum(axis=1).max()

    for evt_id in tqdm(range(0, num, bs), mininterval=10, disable=False):
        bs = min(bs, num - evt_id)

        if clock_dict is not None:
            clock_dict_["sample timer"].reset()

        hits_per_layer_all = hits_per_layer_all_tot[evt_id : evt_id + bs]

        if len(np.where(hits_per_layer_all < 0)[0]) != 0:
            hits_per_layer_all[hits_per_layer_all < 0] = 0

        # padding mask for generated showers
        max_num_clusters = hits_per_layer_all.sum(axis=1).max()
        new_padding_mask = np.zeros((bs, max_num_clusters)).astype(bool)
        col_indices = np.arange(max_num_clusters)
        new_padding_mask[
            col_indices >= np.array(hits_per_layer_all.sum(axis=1))[:, None]
        ] = True

        # conditioning
        cond_E_batch = cond_E[evt_id : evt_id + bs]
        # Inference-only energy calibration of the incident energy seen by the latent model
        # and VAE decompressor; point-count conditioning and returned cond_E are untouched.
        cal = getattr(
            config, "energy_calibration_factor", 1.0
        )  # number, or coefficients (see powers)

        powers = getattr(config, "energy_calibration_powers", None) or (0, -1)

        def calib(e_gev):  # sum_i cal[i] * E^powers[i]; [a, b] alone = a + b / E
            if not isinstance(cal, (list, tuple)):
                return cal
            return sum(c * e_gev**k for c, k in zip(cal, powers))

        e_gev = (cond_E_batch + 1) / 2 * 100
        cond_E_batch_calibrated = (e_gev * calib(e_gev)) / 100 * 2 - 1
        cond_N = (
            torch.Tensor(hits_per_layer_all.sum(axis=1) / max_points)
            .to(config.device)
            .unsqueeze(-1)
        )
        y_cond = get_layers(hits_per_layer_all, 100)
        for j in range(y_cond.shape[0]):
            y_cond[j][new_padding_mask[j]] = 0
        cond_layer = torch.Tensor(y_cond).to(config.device)

        # generation: latent length to decode
        n_points = np.sum(~new_padding_mask, axis=1).min()
        if config_compressor.vae.downsampling_flag & (
            n_points > config_compressor.vae.p_dim
        ):
            n_points = config_compressor.vae.p_dim

        # reconstruction: same energy calibration as the generated path, since it goes
        # through the same VAE decompressor
        reco, cond_layer_r, new_padding_mask_r, z_reco = get_reco_shower(
            vae,
            real_showers[evt_id : evt_id + bs].copy(),
            energy=e[evt_id : evt_id + bs] * calib(e[evt_id : evt_id + bs]),
            config_data=config_data,
            config=config,
            max_points=max_points,
            XZmin=Xmin,
            XZmax=Xmax,
            scale_energy=False,
        )
        latent, fs = get_shower(
            model,
            vae,
            num_points=n_points,
            energy=cond_E_batch_calibrated,
            cond_N=cond_N,
            cond_mint_perlayer=torch.Tensor(samples_mint[evt_id : evt_id + bs]).to(
                config.device
            )
            if with_time
            else None,
            cond_Ylayer=cond_layer,
            bs=bs,
            pm=new_padding_mask,
            config=config,
            clock_dict=clock_dict_,
        )

        if config_data.log_energy:
            fs[:, :, 2] *= config_data.log_var
            fs[:, :, 2] += config_data.log_mean
            fs[:, :, 2] = torch.exp(fs[:, :, 2])
            reco[:, :, 2] *= config_data.log_var
            reco[:, :, 2] += config_data.log_mean
            reco[:, :, 2] = torch.exp(reco[:, :, 2])
        if with_time:
            fs[:, :, 3] *= config_data.log_var_time
            fs[:, :, 3] += config_data.log_mean_time
            fs[:, :, 3] = torch.exp(fs[:, :, 3])
            reco[:, :, 3] *= config_data.log_var_time
            reco[:, :, 3] += config_data.log_mean_time
            reco[:, :, 3] = torch.exp(reco[:, :, 3])
        fs = fs.cpu().numpy()
        reco = reco.cpu().numpy()
        latent = latent.cpu().numpy()
        z_reco = z_reco.cpu().numpy()

        fs = x_y_shift(fs, Xmin, Xmax, Zmin, Zmax)
        reco = x_y_shift(reco, Xmin, Xmax, Zmin, Zmax)

        for i in range(fs.shape[-1]):
            fs[:, :, i][new_padding_mask] = 0
            reco[:, :, i][new_padding_mask_r] = 0

        fs = np.concatenate(
            (fs[:, :, :2], cond_layer.cpu().numpy(), fs[:, :, 2:]),
            axis=-1,
        )

        reco = np.concatenate(
            (reco[:, :, :2], cond_layer_r.cpu().numpy(), reco[:, :, 2:]),
            axis=-1,
        )

        f = 5 if with_time else 4
        length = max_shower_length - fs.shape[1]
        fs = np.concatenate((fs, np.zeros((bs, length, f))), axis=1)  # B, max_points, 4
        latent = np.concatenate(
            (
                latent,
                np.zeros(
                    (bs, config_compressor.vae.p_dim - latent.shape[1], latent.shape[2])
                ),
            ),
            axis=1,
        )
        length = max_shower_length_r - reco.shape[1]
        reco = np.concatenate(
            (reco, np.zeros((bs, length, f))), axis=1
        )  # B, max_points, 4
        z_reco = np.concatenate(
            (
                z_reco,
                np.zeros(
                    (bs, config_compressor.vae.p_dim - z_reco.shape[1], z_reco.shape[2])
                ),
            ),
            axis=1,
        )

        fake_showers_list.append(fs)
        reco_showers_list.append(reco)
        z_encoded_list.append(z_reco)
        z_sampled_list.append(latent)

    fake_showers = np.vstack(fake_showers_list)
    reco_showers = np.vstack(reco_showers_list)
    z_encoded = np.vstack(z_encoded_list)
    z_sampled = np.vstack(z_sampled_list)

    if with_time:
        # time calibration
        min_t_per_l = np.load(
            "./files/npy_array/points_HGx9/min_time_per_layer_for_SF.npy"
        )

        for j in range(min_t_per_l.shape[1]):
            min_t = min_t_per_l[:, j][~np.isnan(min_t_per_l[:, j])].min()

            layer_mask = (fake_showers[:, :, 2] == j) & (fake_showers[:, :, 3] > 0)
            too_small = fake_showers[:, :, 4] < min_t
            fake_showers[:, :, 4][layer_mask & too_small] = min_t

            layer_mask = (reco_showers[:, :, 2] == j) & (reco_showers[:, :, 3] > 0)
            too_small = reco_showers[:, :, 4] < min_t
            reco_showers[:, :, 4][layer_mask & too_small] = min_t

        too_big = fake_showers[:, :, 4] > 200
        fake_showers[:, :, 4][too_big] = 200
        too_big = reco_showers[:, :, 4] > 200
        reco_showers[:, :, 4][too_big] = 200

        fake_showers[:, :, 4][fake_showers[:, :, 3] <= 0] = 0
        reco_showers[:, :, 4][reco_showers[:, :, 3] <= 0] = 0

    if clock_dict is not None:
        if config.only_hcal:
            clock_dict = clock_dict_
        else:
            clock_dict = [clock_dict_sf, clock_dict_]
        return (
            fake_showers,
            samples,
            cond_E.detach().cpu().numpy().astype("float32"),
            real_showers,
            cond_E_real.detach().cpu().numpy(),
            clock_dict,
        )
    else:
        print("Real showers shape: ", real_showers.shape)
        print("Fake showers shape: ", fake_showers.shape)
        print("Reconstructed showers shape: ", reco_showers.shape)
        return (
            fake_showers,
            reco_showers,
            samples,
            cond_E.detach().cpu().numpy().astype("float32"),
            cond_E_r.detach().cpu().numpy().astype("float32"),
            real_showers,
            cond_E_r,
            z_encoded,
            z_sampled,
        )  # (bs, 4, num_points), (bs, 1)
