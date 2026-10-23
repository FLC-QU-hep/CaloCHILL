import datetime
import logging
import logging.handlers
import os
import random
import socket
import time

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.distributed import init_process_group

THOUSAND = 1000
MILLION = 1000000


class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                value = Config(value)
            setattr(self, key, value)

    @staticmethod
    def from_yaml(file_path):
        cfg_dict = OmegaConf.load(file_path)
        cfg_dict = OmegaConf.to_container(cfg_dict, resolve=True)
        return Config(cfg_dict)

    def to_dict(self, keys):
        result = {}
        for key in keys:
            value = getattr(self, key, None)
            if isinstance(value, Config):
                value = value.to_dict(value.__dict__.keys())
            result[key] = value
        return result


class Configs:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                value = Config(value)
            setattr(self, key, value)

    @staticmethod
    def from_yaml(file_path):
        cfg_dict = OmegaConf.load(file_path)
        cfg_dict = OmegaConf.to_container(cfg_dict, resolve=True)
        return Config(cfg_dict)

    def to_dict(self, keys):
        result = {}
        for key in keys:
            value = getattr(self, key, None)
            if isinstance(value, Config):
                value = value.to_dict(value.__dict__.keys())
            result[key] = value
        return result


class BlackHole(object):
    def __setattr__(self, name, value):
        pass

    def __call__(self, *args, **kwargs):
        return self

    def __getattr__(self, name):
        return self


class CheckpointManager(object):
    def __init__(self, save_dir, logger=BlackHole()):
        super().__init__()
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.ckpts = []
        self.logger = logger

        for f in os.listdir(self.save_dir):
            if f[:4] != "ckpt":
                continue
            if f != "ckpt_latest.pt" and f != "ckpt_best.pt":
                _, score, it = f.split("_")
                it = it.split(".")[0]
                self.ckpts.append(
                    {
                        "score": float(score),
                        "file": f,
                        "iteration": int(it),
                    }
                )

    def save(
        self,
        model,
        args,
        score,
        others=None,
        step=None,
        save_latest=False,
        save_best=False,
    ):
        if save_latest:
            fname = "ckpt_latest.pt"
        elif save_best:
            fname = "ckpt_best.pt"
        else:
            if step is None:
                fname = "ckpt_%.6f_.pt" % float(score)
            else:
                fname = "ckpt_%.6f_%d.pt" % (float(score), int(step))
        path = os.path.join(self.save_dir, fname)

        torch.save(
            {"args": args, "state_dict": model.state_dict(), "others": others}, path
        )

        self.ckpts.append({"score": score, "file": fname})

        return True


def load_model_state(model, state_dict, multi_gpu_model=True):
    while hasattr(model, "module"):
        model = model.module

    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    def remap_keys(state_dict):
        new_state = {}
        for k, v in state_dict.items():
            if k.startswith("model.ion."):
                k = k.replace("model.ion.", "diffusion.", 1)
            elif k.startswith("module."):
                k = k.replace("module.", "", 1)
            new_state[k] = v
        return new_state

    state_dict = remap_keys(state_dict)
    model.load_state_dict(state_dict)
    return model


def load_checkpoint(cfg_data, cfg_model, model, model_path, logger=None):
    """
    Load a checkpoint, handle multi-GPU / single-GPU key differences, and set scheduler start.

    Returns:
        state_dict: dict ready to load into model
        it: int, iteration to start from
    """
    model_name = model_path.split("/")[-1]

    if len(model_name.split("_")) == 2:
        # Auto find latest iteration in folder
        it = 0
        folder = os.path.join(cfg_data.log_dir, model_path.split("/")[0])
        for file in os.listdir(folder):
            if file in ("ckpt_best.pt", "ckpt_lastest.pt"):
                continue
            if file.endswith(".pt") and len(file.split("_")) == 3:
                it_temp = int(file.split("_")[2].split(".")[0])
                if it_temp > it:
                    it = it_temp - 1  # start from it-1 to get plot at start
    else:
        it = int(model_name.split("_")[2].split(".")[0])

    cfg_model.scheduler.sched_start_epoch = it

    checkpoint_path = os.path.join(cfg_data.log_dir, model_path)
    checkpoint = torch.load(
        checkpoint_path, map_location=torch.device(cfg_model.device), weights_only=False
    )
    if logger is not None:
        logger.info("Keys in checkpoint: %s", list(checkpoint["state_dict"].keys())[:5])
        logger.info("Keys in model: %s", list(model.state_dict().keys())[:5])

    # Handle multi-GPU / single-GPU mismatches
    state_dict = {}
    if cfg_model.multi_gpu and not list(checkpoint["state_dict"].keys())[0].startswith(
        "module."
    ):
        # Multi-GPU, checkpoint from single-GPU
        if logger is not None:
            logger.info("Checkpoint from single-GPU, loading on multi-GPU")
        for k, v in checkpoint["state_dict"].items():
            state_dict["module." + k] = v
    elif torch.cuda.device_count() == 1 and cfg_model.multi_gpu:
        # Single-GPU, checkpoint from multi-GPU
        if logger is not None:
            logger.info("Checkpoint from multi-GPU, loading on single-GPU")
        for k, v in checkpoint["state_dict"].items():
            state_dict[k.replace("module.model", "module.model")] = v
    else:
        state_dict = {}
        if torch.cuda.device_count() > 1 and cfg_model.multi_gpu:
            if logger is not None:
                logger.info("Multi-GPU, checkpoint from multi-GPU")
            for k, v in checkpoint["state_dict"].items():
                if k.startswith("module.module."):
                    new_key = k.replace("module.module.", "module.module.")
                elif k.startswith("module."):
                    new_key = k.replace("module.", "module.module.")
                else:
                    new_key = k
                state_dict[new_key] = v
        else:
            if logger is not None:
                logger.info("Single-GPU, checkpoint from single-GPU")
            for k, v in checkpoint["state_dict"].items():
                state_dict[k] = v

    return state_dict, checkpoint, it


def seed_all(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True


def module_to_dict(module: nn.Module):
    children = {}
    for name, child in module.named_children():
        children[name] = module_to_dict(child)
    return {
        "type": module.__class__.__name__,
        "params": {
            k: str(v) for k, v in module.__dict__.items() if not k.startswith("_")
        },
        "children": children,
    }


def cfg_to_dict(obj):
    if isinstance(obj, dict):
        return {k: cfg_to_dict(v) for k, v in obj.items()}
    elif hasattr(obj, "__dict__"):  # handles nested Config objects
        return {k: cfg_to_dict(v) for k, v in vars(obj).items()}
    elif isinstance(obj, list):
        return [cfg_to_dict(v) for v in obj]
    else:
        return obj


def get_logger(name, log_dir=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s::%(name)s::%(levelname)s] %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        file_handler = logging.FileHandler(os.path.join(log_dir, "log.txt"))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_new_log_dir(root="./logs", postfix="", prefix="", start_time=time.localtime()):
    log_dir = os.path.join(
        root, prefix + time.strftime("%Y_%m_%d__%H_%M_%S", start_time) + postfix
    )
    os.makedirs(log_dir)
    return log_dir


def default_rendezvous():
    """torch.distributed needs an address and a port even for one process; torchrun sets
    them, a plain `python` run on one GPU gets localhost and a free port."""
    os.environ.setdefault("MASTER_ADDR", "localhost")
    if "MASTER_PORT" not in os.environ:
        with socket.socket() as s:
            s.bind(("", 0))
            os.environ["MASTER_PORT"] = str(s.getsockname()[1])


def ddp_setup_torchrun():
    default_rendezvous()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    init_process_group(
        backend="nccl",
        timeout=datetime.timedelta(minutes=60),
        init_method="env://",
        rank=rank,
        world_size=world_size,
        device_id=local_rank,
    )
    print(f"[Global Rank {rank}] DDP initialized on local rank {local_rank}")


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))
