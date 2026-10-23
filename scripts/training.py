import argparse
import logging
import os
import socket
import subprocess
import sys
import time
from itertools import islice

import comet_ml
import GPUtil
import k_diffusion as K
import numpy as np
import torch
import torch.distributed as dist
from adam_mini import Adam_mini
from models.CaloHadronic import (
    CaloHadronic_Attention,
    CaloHadronic_FM,
    CompressorVAE,
)
from models.CaloHadronic_legacy import CaloClouds2_Attention
from models.common import ShapeMismatchError
from rangerlite import RangerLite
from torch.distributed import destroy_process_group
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from utils.dataset import (
    PionClouds,
    logE_and_padding_sampled_showers,
)
from utils.misc import (
    CheckpointManager,
    Config,
    ddp_setup_torchrun,
    get_logger,
    get_new_log_dir,
    load_checkpoint,
    load_model_state,
    seed_all,
)
from utils.plotting_validation import (
    plot_DS_xye,
    plot_during_train,
    plot_energy_per_layer,
    plot_tsne,
)

os.environ["LIBRARY_PATH"] = "/usr/local/cuda/lib64"


def _weight_norms(module):
    """Parameter norms of the trainable model, as {metric name: float}. Total plus
    the matrices that ran away on the 2026-09-08 mean_flow run: every one of them
    grew ~10x in 85k iterations (ff.w_1 18.5 -> 148, point_embeding.2 9.2 -> 107,
    decoder.2 6.5 -> 46) with the loss flat and the function unchanged, because
    post-LN + qk_norm make them (asymptotically) scale-invariant and weight_decay
    was 0. The EDM run on the same VAE saturated at 1.6x. The growth shows up
    nowhere else -- not in the loss, not in the grad norm -- only as a shrinking
    effective learning rate, so it is logged alongside the loss."""
    named = {
        "wnorm_ff_w1": "inner_model.denoiser.0.layer.feed_forward.w_1.weight",
        "wnorm_point_emb": "inner_model.point_embeding.2.weight",
        "wnorm_q_proj": "inner_model.denoiser.0.layer.self_attn.attn.q_proj.weight",
        "wnorm_decoder": "inner_model.decoder.2.weight",
    }
    params = dict(module.named_parameters())
    out = {
        "wnorm_total": float(
            torch.stack([p.detach().float().norm() for p in params.values()])
            .norm()
            .item()
        )
    }
    for key, name in named.items():
        if name in params:
            out[key] = params[name].detach().float().norm().item()
    return out


def initialize_dataloader(cfg_data, cfg_model, params):
    tot_files = cfg_data.dataloader.total_files
    train_files = cfg_data.dataloader.training_files
    val_files = (tot_files - train_files) // 2

    shared_dset_kwargs = dict(
        files_path=cfg_data.dataset_path,
        tot_files=tot_files,
        bs=cfg_model.common.train_bs - 1,
        net=params.net,
        model_name=cfg_model.model.model_name,
        include_time=cfg_model.common.include_time,
    )

    train_dset = PionClouds(**shared_dset_kwargs, n_files=train_files, phase="train")
    val_dset = PionClouds(
        **shared_dset_kwargs,
        n_files=val_files,
        phase="val",
    )
    test_dset = PionClouds(
        **shared_dset_kwargs,
        n_files=tot_files - train_files - val_files,
        phase="test",
    )

    def make_dataloader(dset):
        return DataLoader(
            dset,
            batch_size=1,
            num_workers=cfg_data.dataloader.workers,
            shuffle=cfg_data.dataloader.shuffle and not cfg_model.multi_gpu,
            sampler=DistributedSampler(dset) if cfg_model.multi_gpu else None,
        )

    return (
        make_dataloader(train_dset),
        make_dataloader(val_dset),
        make_dataloader(test_dset),
    )


def load_and_adjust_configs(world_size, device_count):
    logger = get_logger(name="training.py")
    # None: comet_ml falls back to its own config (~/.comet.config); set log_comet: false to skip Comet
    key = os.environ.get("COMET_API_KEY")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--net",
        default="Compressor",
        type=str,
        choices=["Compressor", "LatentDiff", "CaloHadronic"],
    )
    parser.add_argument("--dataset", default="pions", type=str)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--DP", default=None, type=int)
    params = parser.parse_args()

    config_data_path = "configs/configs_data.yaml"
    cfg_data = Config.from_yaml(config_data_path)

    if params.net == "CaloHadronic":
        config_model_path = "configs/configs_CaloHadronic.yaml"
    elif params.net == "LatentDiff":
        config_model_path = "configs/configs_LatentDiffusion.yaml"
    elif params.net == "Compressor":
        config_model_path = "configs/configs_compressor.yaml"
    # CONFIG_MODEL points a job at its own copy of the model config (same basename,
    # so the copy saved in the run folder keeps the name later stages look for),
    # leaving the shared file free for other queued jobs
    config_model_path = os.environ.get("CONFIG_MODEL", config_model_path)
    cfg_model = Config.from_yaml(config_model_path)

    if cfg_model.model.model_path is not None:
        # Resuming a checkpoint: the architecture must match the run that
        # produced it, so reload cfg_model from the config saved alongside
        # that checkpoint rather than trusting the live config file.
        resumed_model_path = cfg_model.model.model_path
        resume_config_path = os.path.join(
            cfg_data.log_dir,
            os.path.dirname(resumed_model_path),
            os.path.basename(config_model_path),
        )
        cfg_model = Config.from_yaml(resume_config_path)
        cfg_model.model.model_path = resumed_model_path

    cfg_compressor = None
    if params.net == "LatentDiff":
        config_vae_path = (
            os.path.join(
                cfg_data.log_dir,
                os.path.dirname(cfg_model.model.vae_path),
            )
            + "/configs_compressor.yaml"
        )
        cfg_compressor = Config.from_yaml(config_vae_path)

    cfg_model.multi_gpu = False
    if (world_size > 1) or (device_count > 1):
        cfg_model.multi_gpu = True

    # for the ones that depend on the dataset
    all_fields = (
        "log_mean",
        "log_var",
        "max_points",
        "max_points_per_layer",
        "dataloader.total_files",
        "dataloader.training_files",
        "dataset_path",
    )

    def _resolve(obj, path):
        *parents, attr = path.split(".")
        for p in parents:
            obj = getattr(obj, p)
        return obj, attr

    for field in all_fields:
        obj, attr = _resolve(cfg_data, field)
        setattr(obj, attr, getattr(obj, f"{attr}_{params.dataset}"))
    seed_all(seed=cfg_model.common.seed)

    if params.test:
        logger.info("TESTING... ")
        cfg_data.dataloader.total_files = cfg_data.dataloader.total_files_test
        cfg_data.dataloader.training_files = cfg_data.dataloader.training_files_test
        cfg_model.common.train_bs = 3
        cfg_model.log_comet = False

    return (
        cfg_data,
        cfg_model,
        cfg_compressor,
        params,
        key,
        logger,
        config_data_path,
        config_model_path,
    )


# Train, validate and test
class Train(Module):
    def __init__(self, epochs, it_per_one_epoch, experiment, net, dataset):
        super().__init__()

        self.epochs = epochs
        self.it_per_one_epoch = it_per_one_epoch
        self.experiment = experiment
        self.net = net
        self.dataset = dataset

        self.local_rank = int(os.environ["LOCAL_RANK"]) if cfg_model.multi_gpu else 0
        self.global_rank = int(os.environ["RANK"]) if cfg_model.multi_gpu else 0

        self.model, self.model_ema, self.ema_sched, self.it, checkpoint, self.vae = (
            self.get_model(self.local_rank, net=net)
        )

        if self.net == "Compressor":
            self.optimizer, self.scheduler = self.get_optimizer_scheduler_compressor(
                self.model, checkpoint
            )
        else:
            self.optimizer, self.scheduler = self.get_optimizer_scheduler(
                self.model, checkpoint
            )

        if hasattr(cfg_model, "model_sigma"):
            # Sigma (time step) distibution --> lognormal distribution, so minimum value is 0
            self.sample_density = K.config.make_sample_density(
                cfg_model.to_dict(["model_sigma"])["model_sigma"]
            )

    def get_it(self):
        return self.it

    def get_model(self, gpu_id, net):
        cfg_model.downsample_factor = None
        if net == "LatentDiff":
            cfg_model.transformer.input_feats = int(
                cfg_compressor.vae.z_dim * cfg_compressor.vae.n_layers
            )
            cfg_model.downsample_factor = (
                cfg_compressor.vae.p_dim_photons
                if params.dataset == "photons"
                else cfg_compressor.vae.p_dim_pions
            )

        if net == "Compressor":
            cfg_model.vae.calorimeter_layers = cfg_data.calorimeter_layers
            cfg_model.vae.p_dim = (
                cfg_model.vae.p_dim_photons
                if params.dataset == "photons"
                else cfg_model.vae.p_dim_pions
            )
        elif net == "LatentDiff":
            cfg_model.transformer.calorimeter_layers = cfg_data.calorimeter_layers
            cfg_compressor.vae.calorimeter_layers = cfg_data.calorimeter_layers
            cfg_compressor.vae.p_dim = (
                cfg_compressor.vae.p_dim_photons
                if params.dataset == "photons"
                else cfg_compressor.vae.p_dim_pions
            )
        else:
            cfg_model.transformer.calorimeter_layers = cfg_data.calorimeter_layers

        # 'edm' -> Karras/EDM diffusion (CaloHadronic_Attention), 'mean_flow' ->
        # MeanFlow (CaloHadronic_FM, arXiv:2505.13447). Runs predating this option
        # were all trained with EDM, hence the default.
        generative_model = getattr(cfg_model.model, "generative_model", "edm")
        if generative_model == "edm":
            model_cls = CaloHadronic_Attention
        elif generative_model == "mean_flow":
            model_cls = CaloHadronic_FM
        else:
            raise NotImplementedError(
                f"model.generative_model must be 'edm' or 'mean_flow', got {generative_model!r}"
            )

        if net == "CaloHadronic":
            # Oct 2025 point-cloud EDM (run HGx9_ecal_hcal_withT_2025_10_09__13_46_37)
            model = CaloClouds2_Attention(cfg_model)
            model_ema = CaloClouds2_Attention(cfg_model)
        elif net == "Compressor":
            model = CompressorVAE(cfg_data, cfg_model.vae)
            model_ema = None
        elif net == "LatentDiff":
            vae = CompressorVAE(cfg_data, cfg_compressor.vae)
            model = model_cls(cfg_model, net)
            model_ema = model_cls(cfg_model, net)
        else:
            raise NotImplementedError("Model not implemented")

        num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        num_parameters_ema = 0
        if model_ema is not None:
            num_parameters_ema = sum(
                p.numel() for p in model_ema.parameters() if p.requires_grad
            )
        if self.local_rank == 0 and self.global_rank == 0:
            logger.info(
                f"number of parameters: {num_parameters}, {num_parameters_ema} "
            )

        if cfg_model.multi_gpu:
            if not isinstance(model, DDP):
                model = DDP(model.to(gpu_id), device_ids=[gpu_id])
                if model_ema is not None:
                    model_ema = DDP(model_ema.to(gpu_id), device_ids=[gpu_id])

            vae = (
                DDP(vae.to(gpu_id), device_ids=[gpu_id])
                if net == "LatentDiff"
                else None
            )
        else:
            model = model.to(gpu_id)
            model_ema = model_ema.to(gpu_id) if model_ema is not None else None
            vae = vae.to(gpu_id) if net == "LatentDiff" else None

        if cfg_model.model.model_path is not None:
            new_state_dict, checkpoint, it = load_checkpoint(
                cfg_data, cfg_model, model, cfg_model.model.model_path, logger
            )

            model = load_model_state(
                model, new_state_dict, multi_gpu_model=cfg_model.multi_gpu
            )
            model_ema = (
                load_model_state(
                    model_ema, new_state_dict, multi_gpu_model=cfg_model.multi_gpu
                )
                if model_ema is not None
                else None
            )

            # Set the starting lr to the last checkpoint lr
            cfg_model.scheduler.lr = checkpoint["others"]["optimizer"]["param_groups"][
                0
            ]["lr"]
        else:
            it = 1
            checkpoint = None

        while hasattr(model, "module"):
            model = model.module

        # initiate EMA (exponential moving average) model
        ema_sched = 0
        if model_ema is not None:
            while hasattr(model_ema, "module"):
                model_ema = model_ema.module

            model_ema.load_state_dict(model.state_dict())
            model_ema.eval().requires_grad_(False)
            assert cfg_model.ema_type == "inverse"
            ema_sched = K.utils.EMAWarmup(
                power=cfg_model.ema_power, max_value=cfg_model.ema_max_value
            )

        if net == "LatentDiff":
            new_state_dict, _, _ = load_checkpoint(
                cfg_data, cfg_model, vae, cfg_model.model.vae_path, logger
            )
            vae = load_model_state(
                vae, new_state_dict, multi_gpu_model=cfg_model.multi_gpu
            )
            vae.eval().requires_grad_(False)

        return model, model_ema, ema_sched, it, checkpoint, vae

    def get_optimizer_scheduler(self, model, checkpoint):
        base_model = model
        while hasattr(base_model, "module"):
            base_model = base_model.module

        # Decoupled weight decay on the matrices only (ndim >= 2). LayerNorm /
        # RMSNorm gains and biases are not scale-invariant, and at the per-step
        # rate needed here (lr * wd, see scheduler.weight_decay in the config) they
        # would be driven to zero over a run. A single group is kept when
        # weight_decay == 0 so archived runs resume with their optimizer state.
        wd = float(cfg_model.scheduler.weight_decay)
        params = list(base_model.diffusion.parameters())
        # scheduler.weight_decay_skip_cond excludes the conditioning embeddings from
        # decay: under mean_flow.analytic_skip their gradient is second order, so decay
        # wins and the network goes blind to (r, t). Off by default: the extra param
        # group changes the optimizer state layout, so a run cannot resume across it.
        cond_modules = (
            "t_embeding",
            "r_embeding",
            "timestep_embeding",
            "context_embeding",
            "embed_layers",
            "linear_y",
            # transformer.latent_pos_emb is zero-init and has to grow from nothing,
            # which is the one starting condition decay never loses against.
            "latent_pos_emb",
        )
        skip_cond = bool(getattr(cfg_model.scheduler, "weight_decay_skip_cond", False))
        if wd > 0:
            named = list(base_model.diffusion.named_parameters())
            if skip_cond:
                # Match on a whole module name, never a substring: "t_embeding" is
                # a substring of "point_embeding", which processes the latent
                # tokens themselves and must keep its decay.
                def _is_cond(name):
                    return any(part in cond_modules for part in name.split("."))

                mat_cond = [p for n, p in named if p.ndim >= 2 and _is_cond(n)]
                mat_rest = [p for n, p in named if p.ndim >= 2 and not _is_cond(n)]
                diff = [
                    {"params": mat_rest, "weight_decay": wd},
                    {"params": mat_cond, "weight_decay": 0.0},
                    {"params": [p for p in params if p.ndim < 2], "weight_decay": 0.0},
                ]
                if self.local_rank == 0 and self.global_rank == 0:
                    logger.info(
                        "weight_decay_skip_cond: wd %.3g on %d matrices, 0 on %d "
                        "conditioning matrices (%s) and %d 1-D params",
                        wd,
                        len(mat_rest),
                        len(mat_cond),
                        ", ".join(n for n, p in named if p.ndim >= 2 and _is_cond(n)),
                        len(params) - len(mat_rest) - len(mat_cond),
                    )
            else:
                diff = [
                    {"params": [p for p in params if p.ndim >= 2], "weight_decay": wd},
                    {"params": [p for p in params if p.ndim < 2], "weight_decay": 0.0},
                ]
        else:
            diff = params

        shared_opt_kwargs = dict(
            lr=cfg_model.scheduler.lr,
            weight_decay=cfg_model.scheduler.weight_decay,
            # RangerLite defaults to 1e-5; see optimizer.eps in the config for why
            # that turns a clipped gradient explosion into a silent freeze.
            eps=float(getattr(cfg_model.optimizer, "eps", 1e-8)),
        )
        # Optimizer and scheduler
        if cfg_model.optimizer.optimizer == "Adam":
            optimizer = torch.optim.Adam(diff, **shared_opt_kwargs)
        elif cfg_model.optimizer.optimizer == "RAdam":
            optimizer = torch.optim.RAdam(diff, **shared_opt_kwargs)
        elif cfg_model.optimizer.optimizer == "Ranger":
            optimizer = RangerLite(params=diff, **shared_opt_kwargs, lookahead_steps=6)
        elif cfg_model.optimizer.optimizer == "AdamMini":
            named_params = (
                model.module if cfg_model.multi_gpu else model
            ).diffusion.named_parameters()
            verbose = not cfg_model.multi_gpu or dist.get_rank() == 0
            optimizer = Adam_mini(
                named_parameters=named_params,
                **shared_opt_kwargs,
                betas=(0.9, 0.999),
                dim=cfg_model.transformer.d_model,
                n_heads=cfg_model.transformer.nhead,
                verbose=verbose,
            )
            optimizer.wqk_names.add("in_proj")
            optimizer.wv_names.add("in_proj")
            optimizer.attn_proj_names.add("out_proj")
        else:
            raise NotImplementedError("Optimizer not implemented")

        if cfg_model.scheduler.lr_scheduler == "onecyclelr":
            if self.local_rank == 0 and self.global_rank == 0:
                logger.info("One Cycle Scheduler \n")
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=cfg_model.scheduler.max_lr,
                total_steps=int(self.it_per_one_epoch * self.epochs) + 1,
                pct_start=cfg_model.scheduler.pct_start,
                final_div_factor=cfg_model.scheduler.final_div_factor,
            )
        elif cfg_model.scheduler.lr_scheduler == "cosineannealinglr":
            if self.local_rank == 0 and self.global_rank == 0:
                logger.info("Cosine Annealing Scheduler with Linear Warm Up \n")
            cosine_kwargs = dict(
                T_max=cfg_model.scheduler.sched_end_epoch
                - cfg_model.scheduler.sched_start_epoch,
                eta_min=cfg_model.scheduler.end_lr,
            )

            if cfg_model.scheduler.lr_warmup_epochs > 0:
                # the initial lr is determined by the optimizer
                scheduler1 = torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=1, gamma=cfg_model.scheduler.lr_warmup_factor
                )
                scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, **cosine_kwargs
                )
                # Align warm-up final LR with cosine start LR
                s = scheduler2.state_dict()
                s["base_lrs"] = [
                    cfg_model.scheduler.lr_warmup_factor
                    ** cfg_model.scheduler.sched_start_epoch
                    * cfg_model.scheduler.lr
                ]
                scheduler2.load_state_dict(s)

                scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=[scheduler1, scheduler2],
                    milestones=[cfg_model.scheduler.sched_start_epoch],
                )
                # bring back the optimizer to the initial lr
                for param_group in optimizer.param_groups:
                    param_group["lr"] = cfg_model.scheduler.lr
            else:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, **cosine_kwargs
                )
        if checkpoint is not None:
            optimizer.load_state_dict(checkpoint["others"]["optimizer"])
            scheduler.load_state_dict(checkpoint["others"]["scheduler"])
        return optimizer, scheduler

    def get_optimizer_scheduler_compressor(self, model, checkpoint):
        mod = model
        while hasattr(model, "module"):
            mod = model.module
        param_groups = mod.parameters()

        optimizer = RangerLite(
            params=param_groups,
            lr=cfg_model.scheduler.lr,
            betas=(cfg_model.scheduler.beta1, cfg_model.scheduler.beta2),
            weight_decay=cfg_model.scheduler.weight_decay,
            lookahead_steps=10,
        )
        if checkpoint is not None:
            optimizer.load_state_dict(checkpoint["others"]["optimizer"])

        # total_steps/last_epoch are recomputed from the current it_per_one_epoch
        # (rather than restored via scheduler.load_state_dict) since it_per_one_epoch
        # depends on world_size and can change across job resubmissions with a
        # different GPU count, which would otherwise desync the schedule from self.it.
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg_model.scheduler.max_lr,
            total_steps=int(self.it_per_one_epoch * self.epochs) + 1,
            pct_start=cfg_model.scheduler.pct_start,
            final_div_factor=cfg_model.scheduler.final_div_factor,
            last_epoch=self.it - 1 if checkpoint is not None else -1,
        )
        return optimizer, scheduler

    def warm_up(self, optimizer, itr):
        if itr < cfg_model.scheduler.warmup_iters:
            iter_frac = min(
                float(itr + 1) / max(cfg_model.scheduler.warmup_iters, 1), 1.0
            )
            lr = cfg_model.scheduler.lr * iter_frac
            for i, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = lr

    @torch.no_grad()
    def val(self, val_loader, it, epoch, device, test=False):
        cfg_model.device = device
        model = self.model
        while hasattr(self.model, "module"):
            model = self.model.module

        if self.net == "Compressor":
            tot = model.reconstruction_and_sampling(
                val_loader, epoch, device, test=test
            )
            if tot is not None:
                all_res, ref_pts, rec_pts, samp_pts, others = tot

                logger.info(
                    "[Val] Iter %04d | Loss %.6f | kl loss %.4f | rec loss %.4f"
                    % (it, all_res["loss"], all_res["kl_loss"], all_res["rec_loss"])
                )

                others["latentdiff_vae"] = None

                if cfg_model.log_comet:
                    val_metrics = {
                        "val/loss": float(all_res["loss"]),
                        "val/kl_loss": float(all_res["kl_loss"]),
                        "val/rec_loss": float(all_res["rec_loss"]),
                    }

                    for metric_name, value in val_metrics.items():
                        self.experiment.log_metric(metric_name, value, it)

                mask = None
                if others["pm"] is not None:
                    mask = others["pm"].numpy()
                ref_pts = torch.cat(
                    [ref_pts[:, :, :1], others["layer"], ref_pts[:, :, 1:]], dim=-1
                )
                samp_pts = torch.cat(
                    [samp_pts[:, :, :1], others["layer"], samp_pts[:, :, 1:]], dim=-1
                )
                rec_pts = torch.cat(
                    [rec_pts[:, :, :1], others["layer"], rec_pts[:, :, 1:]], dim=-1
                )
                plot_dict = {
                    "geant4": ref_pts,
                    "gen": samp_pts,
                    "rec": rec_pts,
                }

        else:
            fake_shower, cond_feats, layer, x, PpL_max, pm_list, mint = (
                [],
                [],
                [],
                [],
                [],
                [],
                [],
            )
            z_vae_, z_diff_ = [], []
            fake_shower_vae = []
            rec_shower = []
            others = {}
            others["all_DS"] = None
            max_length = 0

            for i, batch in enumerate(islice(val_loader, 8)):
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key][0].to(device)

                number_of_points = torch.sum(~batch["padding_mask"], dim=1).min()
                if (
                    cfg_compressor is not None
                    and cfg_compressor.vae.downsampling_flag
                    & (number_of_points > cfg_compressor.vae.p_dim)
                ):
                    number_of_points = cfg_compressor.vae.p_dim
                if self.net == "CaloHadronic":  # one generated point per hit slot
                    number_of_points = batch["layer_cond"].shape[1]

                fake_shower_ = model.sample(
                    batch["cond_feats"],
                    number_of_points,
                    cfg_model,
                    y=batch["layer_cond"],
                    mint_per_layer=batch["mint_per_layer"],
                    padding_mask=batch["padding_mask"],
                )

                if self.net == "LatentDiff":
                    vae_mod = self.vae
                    while hasattr(vae_mod, "module"):
                        vae_mod = vae_mod.module

                    z_vae = vae_mod.encode_for_latent(
                        batch["events"],
                        ctx=batch["cond_feats"],
                        y=batch["layer_cond"],
                        mint_per_layer=batch["mint_per_layer"],
                        padding=batch["padding_mask"],
                    )
                    z_diff = fake_shower_
                    z_vae_plot = z_vae
                    if getattr(model, "latent_target", "sample") == "mu":
                        # mean_flow on the posterior mean: the latent plots compare
                        # the flow's target (mu / latent_scale) with its raw output;
                        # the decoder gets latent_scale * output + posterior noise.
                        z_vae_plot = model.encode_latent_target(
                            vae_mod,
                            batch["events"],
                            ctx=batch["cond_feats"],
                            y=batch["layer_cond"],
                            mint_per_layer=batch["mint_per_layer"],
                            padding=batch["padding_mask"],
                        )
                        fake_shower_ = model.latent_to_decoder(fake_shower_)

                    fake_shower_, _ = vae_mod.sample(
                        N=batch["layer_cond"].shape[1],
                        c=batch["cond_feats"],
                        y=batch["layer_cond"],
                        mint_per_layer=batch["mint_per_layer"],
                        padding=batch["padding_mask"],
                        device=batch["cond_feats"].device,
                        z=fake_shower_,
                    )
                    for j in range(fake_shower_.shape[-1]):
                        fake_shower_[:, :, j][batch["padding_mask"]] = 0

                    # also decode a sample drawn directly from the VAE's own
                    # prior (z=None -> standard normal), bypassing the
                    # diffusion model, to compare against the LatentDiff sample
                    fake_shower_vae_, _ = vae_mod.sample(
                        B=batch["cond_feats"].shape[0],
                        N=batch["layer_cond"].shape[1],
                        c=batch["cond_feats"],
                        y=batch["layer_cond"],
                        mint_per_layer=batch["mint_per_layer"],
                        padding=batch["padding_mask"],
                        device=batch["cond_feats"].device,
                        z=None,
                    )
                    for j in range(fake_shower_vae_.shape[-1]):
                        fake_shower_vae_[:, :, j][batch["padding_mask"]] = 0

                    # true VAE reconstruction: decode the latent encoded from
                    # the real shower (z_vae), without going through diffusion
                    rec_shower_, _ = vae_mod.sample(
                        N=batch["layer_cond"].shape[1],
                        c=batch["cond_feats"],
                        y=batch["layer_cond"],
                        mint_per_layer=batch["mint_per_layer"],
                        padding=batch["padding_mask"],
                        device=batch["cond_feats"].device,
                        z=z_vae,
                    )
                    for j in range(rec_shower_.shape[-1]):
                        rec_shower_[:, :, j][batch["padding_mask"]] = 0

                layer.append(batch["layer_cond"].detach().cpu())
                mint.append(batch["mint_per_layer"].detach().cpu())
                x.append(batch["events"].detach().cpu())  # already on cpu
                pm_list.append(batch["padding_mask"].detach().cpu())
                fake_shower.append(fake_shower_.detach().cpu())
                cond_feats.append(batch["cond_feats"].detach().cpu())
                if self.net == "LatentDiff":
                    z_vae_.append(z_vae_plot.detach().cpu())
                    z_diff_.append(z_diff.detach().cpu())
                    fake_shower_vae.append(fake_shower_vae_.detach().cpu())
                    rec_shower.append(rec_shower_.detach().cpu())

                max_length = max(max_length, batch["layer_cond"].shape[1])

            def pad_tensor(tensor, i, pad, value):
                tensor[i] = torch.nn.functional.pad(
                    tensor[i],
                    pad,
                    mode="constant",
                    value=value,
                )

            for i in range(len(layer)):
                if layer[i].shape[1] < max_length:
                    pad_size = max_length - layer[i].shape[1]
                    pad_tensor(fake_shower, i, (0, 0, 0, pad_size), 0)
                    pad_tensor(x, i, (0, 0, 0, pad_size), 0)
                    pad_tensor(layer, i, (0, 0, 0, pad_size), -1)
                    pad_tensor(pm_list, i, (0, pad_size), 1)
                    if self.net == "LatentDiff":
                        pad_tensor(fake_shower_vae, i, (0, 0, 0, pad_size), 0)
                        pad_tensor(rec_shower, i, (0, 0, 0, pad_size), 0)

            layer = torch.cat(layer, dim=0)
            mint = torch.cat(mint, dim=0)
            x = torch.cat(x, dim=0)
            cond_feats = torch.cat(cond_feats, dim=0)
            fake_shower = torch.cat(fake_shower, dim=0)
            fake_shower_vae = (
                torch.cat(fake_shower_vae, dim=0) if self.net == "LatentDiff" else None
            )
            rec_shower = (
                torch.cat(rec_shower, dim=0) if self.net == "LatentDiff" else None
            )
            mask = torch.cat(pm_list, dim=0)

            if self.net == "LatentDiff":
                others["cond_e"] = cond_feats[:, 0]  # incident energy for t-SNE
                others["cond_feats"], others["latent"], others["posteriors"] = (
                    None,
                    None,
                    None,
                )
                # pad first the latent vectors to have the same number of points
                max_len = z_diff_[0].shape[1]
                for i in range(len(z_diff_)):
                    max_len = max(max_len, z_diff_[i].shape[1])

                # True where a token is real; the zero padding added below (batches
                # with a smaller effective_p_dim) must not enter the latent histograms.
                z_mask_ = [
                    (torch.arange(max_len) < z.shape[1]).expand(z.shape[0], -1)
                    for z in z_diff_
                ]
                for i in range(len(z_diff_)):
                    if z_diff_[i].shape[1] < max_len:
                        pad_size = max_len - z_diff_[i].shape[1]
                        z_diff_[i] = torch.nn.functional.pad(
                            z_diff_[i],
                            (0, 0, 0, pad_size),
                            mode="constant",
                            value=0,
                        )
                    if z_vae_[i].shape[1] < max_len:
                        pad_size = max_len - z_vae_[i].shape[1]
                        z_vae_[i] = torch.nn.functional.pad(
                            z_vae_[i],
                            (0, 0, 0, pad_size),
                            mode="constant",
                            value=0,
                        )
                others["latentdiff_sample"] = torch.cat(z_diff_, dim=0)
                others["latentdiff_vae"] = torch.cat(z_vae_, dim=0)
                others["latentdiff_mask"] = torch.cat(z_mask_, dim=0)

            max_cluster = layer.shape[1]
            if cfg_model.model.model_name == "CaloHadronicVAE":
                fake_shower = torch.cat(
                    [fake_shower[:, :, :1], layer, fake_shower[:, :, 1:]], dim=-1
                )
                x = torch.cat([x[:, :, :1], layer, x[:, :, 1:]], dim=-1)
                if fake_shower_vae is not None:
                    fake_shower_vae = torch.cat(
                        [fake_shower_vae[:, :, :1], layer, fake_shower_vae[:, :, 1:]],
                        dim=-1,
                    )
                if rec_shower is not None:
                    rec_shower = torch.cat(
                        [rec_shower[:, :, :1], layer, rec_shower[:, :, 1:]],
                        dim=-1,
                    )

            elif cfg_model.model.model_name == "CaloHadronic":
                PpL_max = torch.cat(PpL_max, dim=0)

                num_clusters = (
                    (cond_feats[:, 1:] * PpL_max)
                    .detach()
                    .cpu()
                    .numpy()
                    .sum(axis=1)
                    .astype(int)
                )
                padding2 = np.zeros((fake_shower.shape[0], max_cluster)).astype(bool)
                col_indices = np.arange(max_cluster)
                padding2[col_indices >= np.array(num_clusters.reshape(-1))[:, None]] = (
                    True
                )
            else:
                num_clusters = (cond_feats[:, 1] * cfg_data.x).numpy().astype(int)
                padding2 = np.zeros((fake_shower.shape[0], max_cluster)).astype(bool)
                col_indices = np.arange(max_cluster)
                padding2[col_indices > np.array(num_clusters.reshape(-1))[:, None]] = (
                    True
                )

            plot_dict = {
                "geant4": x,
                "gen": fake_shower,
                "rec": rec_shower,
                "samp_vae": fake_shower_vae,
            }

        if cfg_model.log_comet:
            processed = {
                key: logE_and_padding_sampled_showers(
                    cfg_data,
                    tensor.cpu().numpy(),
                    self.dataset,
                    padding=mask,
                )
                if tensor is not None
                else None
                for key, tensor in plot_dict.items()
            }
            g4, gen, rec = processed["geant4"], processed["gen"], processed["rec"]
            samp_vae = processed.get("samp_vae")

            if "all_DS" in others and "all_notDS" in others:
                for key, downsampled, threshold_op in [
                    ("all_DS", True, lambda n: n > cfg_model.vae.p_dim),
                    ("all_notDS", False, lambda n: n <= cfg_model.vae.p_dim),
                ]:
                    if others[key] is not None:
                        shower = logE_and_padding_sampled_showers(
                            cfg_data,
                            others[key].cpu().numpy(),
                            self.dataset,
                            padding=others[f"mask_{'DS' if downsampled else 'notDS'}"]
                            .cpu()
                            .numpy(),
                            en_idx=2,
                        )
                        n_points = (g4[:, :, 3] > 0).sum(axis=1)
                        # g4 is [x, layer, z, logE(, logT)]: drop the layer column,
                        # keeping the time column only when the model has one
                        g4_cols = (
                            [0, 2, 3, 4] if cfg_model.common.include_time else [0, 2, 3]
                        )
                        x2 = g4[:, :, g4_cols][threshold_op(n_points)]
                        plot_DS_xye(
                            x2,
                            shower,
                            downsampled,
                            experiment=self.experiment,
                            it=self.it,
                        )

            plot_during_train(
                g4,
                rec,
                samp_shower=gen,
                samp_shower_vae=samp_vae,
                others=others,
                plot_name="VAL",
                experiment=self.experiment,
                it=self.it,
            )
            plot_energy_per_layer(
                g4,
                rec,
                samp_shower=gen,
                samp_shower_vae=samp_vae,
                experiment=self.experiment,
                it=self.it,
            )

            plot_tsne(others, g4, self.net, self.experiment, self.it)

    def beta_fn(self, it, start=1e-4, stop=1, warmup_epochs=50, iters_per_epoch=1000):
        warmup_iters = warmup_epochs * iters_per_epoch
        return min(stop, start + (stop - start) * it / warmup_iters)

    def forward(self, batch, it, device):
        self.it = it
        cfg_model.device = device
        global time_tick
        time_tick = time.time()
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key][0].to(device)
        kwargs = {
            "n_points": batch["points_per_layer"].sum(dim=1),
            "top_points_layers": batch["points_per_layer"].topk(2)[0].sum(dim=1),
        }

        if self.net == "LatentDiff":
            vae_mod = self.vae
            while hasattr(vae_mod, "module"):
                vae_mod = vae_mod.module
            model_mod = self.model
            while hasattr(model_mod, "module"):
                model_mod = model_mod.module

            if getattr(model_mod, "latent_target", "sample") == "mu":
                # mean_flow on the posterior mean (mean_flow.latent_target: mu)
                x = model_mod.encode_latent_target(
                    vae_mod,
                    batch["events"],
                    ctx=batch["cond_feats"],
                    y=batch["layer_cond"],
                    mint_per_layer=batch["mint_per_layer"],
                    padding=batch["padding_mask"],
                )
            else:
                x = vae_mod.encode_for_latent(
                    batch["events"],
                    ctx=batch["cond_feats"],
                    y=batch["layer_cond"],
                    mint_per_layer=batch["mint_per_layer"],
                    padding=batch["padding_mask"],
                )
            if getattr(model_mod, "gmm_prior", None) is not None:
                noise = model_mod.gmm_prior.sample_eps(
                    x, cond_feats=batch["cond_feats"]
                )
            else:
                noise = torch.randn_like(x).to(device)  # noise for forward diffusion
            sigma = self.sample_density([x.shape[0]], device=device)  # time steps
            loss = self.model.get_loss(
                x,
                noise,
                sigma,
                batch["cond_feats"],
                padding_mask=batch["padding_mask"],
                y=batch["layer_cond"],
                mint_per_layer=batch["mint_per_layer"],
                it=it,
                **kwargs,
            )

            # Backward and optimize
            loss.backward()
            orig_grad_norm = clip_grad_norm_(
                self.model.parameters(), cfg_model.optimizer.max_grad_norm
            )
            self.optimizer.step()
            self.scheduler.step()

            # Update EMA model
            ema_decay = self.ema_sched.get_value()
            K.utils.ema_update(self.model, self.model_ema, ema_decay)
            self.ema_sched.step()

            if self.local_rank == 0 and self.global_rank == 0:
                if (it % cfg_model.log_iter == 0) | (it == 10):
                    time_elapsed = time.time() - time_tick
                    logger.info(
                        "[Train] Iter %04d | Time %.4f | Loss %.6f | Grad %.4f | KLWeight %.4f | EMAdecay %.4f"
                        % (
                            it,
                            time_elapsed,
                            loss.item(),
                            orig_grad_norm,
                            cfg_model.model.kl_diffusion_weight,
                            ema_decay,
                        )
                    )
                    if self.experiment is not None:
                        self.experiment.log_metric("train/loss", loss, it)
                        self.experiment.log_metric(
                            "train/kl_diffusion_weight",
                            cfg_model.model.kl_diffusion_weight,
                            it,
                        )
                        self.experiment.log_metric(
                            "train/lr", self.optimizer.param_groups[0]["lr"], it
                        )
                        self.experiment.log_metric(
                            "train/grad_norm", orig_grad_norm, it
                        )
                        self.experiment.log_metric("train/ema_decay", ema_decay, it)
                        if getattr(model_mod, "gmm_prior", None) is not None:
                            with torch.no_grad():
                                pi_mean = model_mod.gmm_prior.forward(
                                    batch["cond_feats"]
                                )[0].mean(dim=0)
                            for k in range(pi_mean.shape[0]):
                                self.experiment.log_metric(
                                    f"train/gmm_prior_pi_{k}", pi_mean[k].item(), it
                                )

                    # MeanFlow one-step diagnostics; empty for the EDM model.
                    mf_metrics = getattr(model_mod, "mf_metrics", None)
                    if mf_metrics:
                        logger.info(
                            "[Train] Iter %04d | %s"
                            % (
                                it,
                                " | ".join(
                                    "%s %.4f" % (k.split("/")[-1], v)
                                    for k, v in mf_metrics.items()
                                ),
                            )
                        )
                        if self.experiment is not None:
                            for name, value in mf_metrics.items():
                                self.experiment.log_metric("train/" + name, value, it)

                    # Weight norms, for the EDM control as well (see _weight_norms).
                    wnorms = _weight_norms(model_mod.diffusion)
                    logger.info(
                        "[Train] Iter %04d | %s"
                        % (
                            it,
                            " | ".join("%s %.2f" % (k, v) for k, v in wnorms.items()),
                        )
                    )
                    if self.experiment is not None:
                        for name, value in wnorms.items():
                            self.experiment.log_metric("train/" + name, value, it)

                    time_tick = time.time()
                return (
                    self.model,
                    self.model_ema,
                    self.vae,
                    self.ema_sched,
                    self.optimizer,
                    self.scheduler,
                )

        elif self.net == "Compressor":
            self.warm_up(self.optimizer, self.it)
            beta = self.beta_fn(
                self.it, warmup_epochs=50, iters_per_epoch=self.it_per_one_epoch
            )

            loss_dict = self.model.get_loss(
                batch["events"],
                ctx=batch["cond_feats"],
                y=batch["layer_cond"],
                mint_per_layer=batch["mint_per_layer"]
                if cfg_model.common.include_time
                else None,
                padding_mask=batch["padding_mask"],
                beta=beta,
                it=self.it,
                **kwargs,
            )

            if loss_dict is None:
                logger.info("Skipping batch ")
                return self.model, 0, 0, 0, self.optimizer, self.scheduler
            loss = loss_dict["loss"]

            if torch.isnan(loss):
                logger.warning("Loss is NaN")
                logger.info(loss.item())
                for key, value in loss_dict.items():
                    print(key, value)
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        print(key, torch.isnan(batch[key]).any())
                import sys

                sys.exit()

            self.optimizer.zero_grad()
            loss.backward()

            if cfg_model.scheduler.grad_norm_clip_value is not None:
                params = self.model.parameters()
                grad_norm = clip_grad_norm_(
                    params, cfg_model.scheduler.grad_norm_clip_value
                )
            else:
                grad_norm = 0.0

            self.optimizer.step()
            self.scheduler.step()

            if self.local_rank == 0 and self.global_rank == 0:
                if (it % cfg_model.log_iter == 0) | (it == 10):
                    loss = loss.detach().cpu().item()  # unavoidable sync for logging
                    kl = loss_dict.get("kl_loss")
                    rec = loss_dict.get("rec_loss")
                    kl_scalar = kl.detach().cpu().item() if kl is not None else 0.0
                    rec_scalar = rec.detach().cpu().item() if rec is not None else 0.0

                    time_elapsed = time.time() - time_tick  # in seconds

                    logger.info(
                        f"[Train] Iter {it:04d} | Time {time_elapsed:.4f} | Grad {grad_norm:.2f} | "
                        f"Loss {loss:.5f} | kl {kl_scalar:.4f} | rec {rec_scalar:.4f}"
                    )

                    if self.experiment is not None:
                        self.experiment.log_metric("train/loss", loss, it)
                        self.experiment.log_metric("train/grad", float(grad_norm), it)
                        self.experiment.log_metric("train/kl_loss", kl_scalar, it)
                        self.experiment.log_metric("train/beta", float(beta), it)
                        self.experiment.log_metric("train/rec_loss", rec_scalar, it)
                        self.experiment.log_metric(
                            "train/lr", self.optimizer.param_groups[0]["lr"], it
                        )
                return self.model, 0, 0, 0, self.optimizer, self.scheduler

        elif self.net == "CaloHadronic":
            x = batch["events"]
            noise = torch.randn_like(x)  # noise for forward diffusion
            sigma = self.sample_density([x.shape[0]], device=x.device)  # time steps

            loss = self.model.get_loss(
                x,
                noise,
                sigma,
                batch["cond_feats"],
                padding_mask=batch["padding_mask"],
                y=batch["layer_cond"],
                mint_per_layer=batch["mint_per_layer"],
            )

            if torch.isnan(loss):
                logger.warning("Loss is NaN")
                logger.info((~batch["padding_mask"]).sum(axis=1))
                sys.exit()

            # Backward and optimize (no zero_grad, as in the original run)
            loss.backward()
            orig_grad_norm = clip_grad_norm_(
                self.model.parameters(), cfg_model.optimizer.max_grad_norm
            )
            self.optimizer.step()
            self.scheduler.step()

            # Update EMA model
            ema_decay = self.ema_sched.get_value()
            K.utils.ema_update(self.model, self.model_ema, ema_decay)
            self.ema_sched.step()

            if self.local_rank == 0 and self.global_rank == 0:
                if (it % cfg_model.log_iter == 0) | (it == 10):
                    time_elapsed = time.time() - time_tick
                    logger.info(
                        "[Train] Iter %04d | Time %.4f | Loss %.6f | Grad %.4f | EMAdecay %.4f"
                        % (it, time_elapsed, loss.item(), orig_grad_norm, ema_decay)
                    )
                    if self.experiment is not None:
                        self.experiment.log_metric("train/loss", loss, it)
                        self.experiment.log_metric(
                            "train/lr", self.optimizer.param_groups[0]["lr"], it
                        )
                        self.experiment.log_metric(
                            "train/grad_norm", orig_grad_norm, it
                        )
                        self.experiment.log_metric("train/ema_decay", ema_decay, it)
                    time_tick = time.time()
                return (
                    self.model,
                    self.model_ema,
                    0,
                    self.ema_sched,
                    self.optimizer,
                    self.scheduler,
                )


# Main loop
def main(rank, global_rank, node_rank, world_size):
    if global_rank != 0:
        logging.getLogger("comet_ml").setLevel(logging.WARNING)
    log_dir = None
    if params.net == "LatentDiff":
        the_name = params.dataset + "_" + cfg_model.model.vae_path.split("/")[0] + "_"
        the_name = the_name.replace("VAE", "LatentDiff")
        # Tag the run with the generative objective, so 'edm' and 'mean_flow' runs off
        # the same VAE checkpoint get distinct log dirs and comet experiment names.
        the_name += getattr(cfg_model.model, "generative_model", "edm") + "_"
    else:
        the_name = params.dataset + "_" + cfg_model.name

    if (rank == 0) & (global_rank == 0):
        logger.info(f"DATASET: {params.dataset}  |  NET: {params.net}")
        if cfg_model.model.model_path is not None:
            log_dir = cfg_data.log_dir + cfg_model.model.model_path.split("/")[0]
        else:
            log_dir = get_new_log_dir(
                cfg_data.log_dir,
                prefix=the_name,
                postfix="_",
                start_time=time.localtime(),
            )

    torch.cuda.set_device(rank)
    ddp_setup_torchrun()
    if world_size > 1:
        store = dist.TCPStore(  # type: ignore
            host_name=os.environ["MASTER_ADDR"],
            port=12345,
            world_size=world_size,
            is_master=(rank == 0),
        )
        if (rank == 0) & (global_rank == 0):
            store.set("log_dir", log_dir)
        else:
            store.wait(["log_dir"])
            log_dir = store.get("log_dir").decode()

    if (rank == 0) & (global_rank == 0):
        logger.info(
            f"copying the config files inside the checkpoint directory: \n {log_dir}"
        )
        bash_command = f"cp {config_data_path} {log_dir}/"
        subprocess.run(bash_command, shell=True, capture_output=True, text=True)
        ckpt_mgr = CheckpointManager(log_dir)
        if cfg_model.model.model_path is None:
            bash_command2 = f"cp {config_model_path} {log_dir}/"
            subprocess.run(bash_command2, shell=True, capture_output=True, text=True)

    if cfg_model.common.include_time and (rank == 0) and (global_rank == 0):
        logger.info(" INCLUDING TIME! ")

    if (rank == 0) & (global_rank == 0):
        if params.test:
            logger.info("Setting Small Dataloader for testing...")
        else:
            logger.info("Setting Dataloader...")
    dataloader, dataloader_val, dataloader_test = initialize_dataloader(
        cfg_data, cfg_model, params
    )
    start_time = time.time()
    print("len dataloader: ", len(dataloader))
    it_per_one_epoch = len(dataloader)

    if (rank == 0) & (global_rank == 0):
        logger.info(
            f"dataloader size: {len(dataloader)}  |  it_per_one_epoch: {it_per_one_epoch:.2f}"
        )
        if cfg_model.log_comet:
            logger.info("Logging on CometML --> SWITCHED ON")
        else:
            logger.info("Logging on CometML --> SWITCHED OFF")
    experiment, exp_key = None, None
    if cfg_model.log_comet and (global_rank == 0):
        experiment = comet_ml.start(
            project_name=cfg_model.comet_project,
            api_key=key,
        )
        exp_key = experiment.get_key()  # unique key for this experiment

        experiment.log_parameters(cfg_model.__dict__)
        experiment.set_name(log_dir.split("/")[-1])
        if rank == 0:
            if params.net == "Compressor":
                experiment.log_code(file_name=config_model_path)
            elif params.net == "LatentDiff":
                experiment.log_code(file_name="configs/configs_LatentDiffusion.yaml")
            else:
                experiment.log_code(file_name=config_model_path)
            experiment.log_code(file_name="scripts/training.py")

    if cfg_model.multi_gpu and cfg_model.log_comet:
        exp_key_list = [exp_key] if global_rank == 0 else [None]
        dist.broadcast_object_list(exp_key_list, src=0)
        exp_key = exp_key_list[0]
        if global_rank != 0:
            experiment = comet_ml.ExistingExperiment(
                previous_experiment=exp_key,
                api_key=key,
                project_name=cfg_model.comet_project,
            )
    if cfg_model.log_comet and experiment is not None:
        experiment.log_parameters(
            {
                "global_rank": global_rank,
                "local_rank": rank,
                "node_rank": node_rank,
                "hostname": os.uname().nodename,
            }
        )
        experiment.add_tag(f"node_{node_rank}")

    epochs = cfg_model.common.epochs
    # initializing Train class
    if (rank == 0) & (global_rank == 0):
        if params.test:
            logger.info("Start Testing the Training...")
        else:
            logger.info("Start Training...")
    train = Train(
        epochs, it_per_one_epoch, experiment, net=params.net, dataset=params.dataset
    )
    it = train.get_it() - 1
    if (cfg_model.model.model_path is not None) & (rank == 0 and global_rank == 0):
        logger.info(f"Resuming from it={it}, it_per_one_epoch={it_per_one_epoch}")
    model_mod = train.model
    while hasattr(model_mod, "module"):
        model_mod = model_mod.module
    if getattr(model_mod, "latent_target", "sample") == "mu":
        # mean_flow on the posterior mean: measure latent_scale / latent_noise_std
        # once (no-op when resuming, the buffers come with the checkpoint). All
        # ranks take part (all-reduce inside).
        vae_mod = train.vae
        while hasattr(vae_mod, "module"):
            vae_mod = vae_mod.module
        model_mod.measure_latent_stats(
            vae_mod,
            dataloader,
            rank,
            logger=logger if (rank == 0 and global_rank == 0) else None,
        )
        if train.model_ema is not None:
            ema_mod = train.model_ema
            while hasattr(ema_mod, "module"):
                ema_mod = ema_mod.module
            ema_bufs = dict(ema_mod.named_buffers())
            for name, buf in model_mod.named_buffers():
                # latent_* on the model, diffusion.sigma_data_dim (per-dim skip)
                if name.startswith("latent_") or name.endswith("sigma_data_dim"):
                    ema_bufs[name].copy_(buf)
    if cfg_model.multi_gpu:
        dist.barrier()
    start_epoch = int(it // int(it_per_one_epoch)) if int(it_per_one_epoch) != 0 else 0

    for epoch in range(start_epoch, epochs):
        # note: the batch here are changed dinamically. From the dataloader I have indeces which are
        # the number of the input data (2M). Then every time the dataloader chooses an index, I take the showers
        # from that index to index + batch_size.
        for batch in dataloader:
            current_epoch = it // int(it_per_one_epoch)
            should_val = (
                rank == 0 and global_rank == 0 and it % int(it_per_one_epoch) == 0
            )
            if should_val:
                logger.info("EPOCH: %d" % current_epoch)

            if cfg_model.multi_gpu:
                dist.barrier()  # all ranks sync BEFORE val

            if should_val:
                train.val(dataloader_val, it, current_epoch, rank, test=params.test)

            if cfg_model.multi_gpu:
                dist.barrier()
            try:
                if rank == 0 and global_rank == 0:
                    model, model_ema, _, ema_sched, optimizer, scheduler = train(
                        batch, it, rank
                    )
                else:
                    train(batch, it, rank)
            except ShapeMismatchError as e:
                logger.error(e)
                continue

            if params.test:
                break

            torch.cuda.empty_cache()
            it += 1

            if rank == 0 and global_rank == 0:
                if (it % int(it_per_one_epoch) == 0) or (it == 100):
                    print(f"SAVING at it={it} in {log_dir}")
                    mod = model if params.net == "Compressor" else model_ema
                    while hasattr(mod, "module"):
                        mod = mod.module
                    mod = mod.state_dict()

                    opt_states = {
                        "model": mod,  # save the model in case of compressor, model_ema for diffusion
                        "ema_sched": ema_sched.state_dict()
                        if params.net != "Compressor"
                        else None,
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                    }

                    ckpt_mgr.save(model, cfg_model, 0, others=opt_states, step=it)
                    ckpt_mgr.save(
                        model, cfg_model, 0, others=opt_states, save_latest=True
                    )
                    ckpt_mgr.save(
                        model, cfg_model, 0, others=opt_states, save_best=True
                    )

            if cfg_model.multi_gpu:
                dist.barrier()
        if params.test:
            break

    if params.net == "Compressor":
        logger.info("Final --> running test data...")
        train.val(dataloader_test, it, epoch, rank)

    if cfg_model.log_comet:
        experiment.end()

    if cfg_model.multi_gpu:
        destroy_process_group()
    logger.info("training done in %.2f seconds" % (time.time() - start_time))
    if params.test:
        logger.info("TESTING... DONE ")


if __name__ == "__main__":
    # count the GPUs
    node_name = socket.gethostname()
    device_count = torch.cuda.device_count()
    gpus = GPUtil.getGPUs()
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    node_rank = int(os.environ.get("GROUP_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.set_float32_matmul_precision("high")

    (
        cfg_data,
        cfg_model,
        cfg_compressor,
        params,
        key,
        logger,
        config_data_path,
        config_model_path,
    ) = load_and_adjust_configs(world_size, device_count)

    if global_rank != 0:
        sys.stdout = open(os.devnull, "w")  # silence other ranks
    main(local_rank, global_rank, node_rank, world_size)
    if local_rank == 0 and global_rank == 0:
        logger.info("all finished. Tschüüüssss")
