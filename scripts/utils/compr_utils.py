import argparse

import torch
import torch.nn as nn


class Compute_log_probabilities:
    def __init__(self):
        self.c = 0.9189385332  # log(2 * pi)

    def log_p_var_normal(self, samples, mu, logvar):
        log_p = (
            -0.5 * torch.square(samples - mu) / torch.exp(logvar)
            - 0.5 * logvar
            - self.c
        )
        return log_p

    def log_p_normal(self, samples):
        log_p = -0.5 * torch.square(samples) - self.c
        return log_p

    def compute_kl(self, mu, logvar):
        # closed-form KL to N(0, 1)
        return -0.5 * (logvar - torch.exp(logvar) - mu.pow(2) + 1).view(
            mu.shape[0], -1
        ).sum(dim=-1)

    def sample(self, mu, logvar):
        eps = torch.randn(mu.shape).to(mu)
        z = mu + torch.exp(logvar / 2.0) * eps
        return eps, z


def get_activation(activation):
    if activation is None:
        return nn.Identity()
    if activation.lower() == "gelu":
        return nn.GELU()
    elif activation.lower() == "rrelu":
        return nn.RReLU(inplace=True)
    elif activation.lower() == "selu":
        return nn.SELU(inplace=True)
    elif activation.lower() == "silu":
        return nn.SiLU(inplace=True)
    elif activation.lower() == "hardswish":
        return nn.Hardswish(inplace=True)
    elif activation.lower() == "leakyrelu":
        return nn.LeakyReLU(inplace=True)
    elif activation.lower() == "leakyrelu0.2":
        return nn.LeakyReLU(negative_slope=0.2, inplace=True)
    elif activation.lower() == "swish":
        return nn.SiLU()
    else:
        return nn.ReLU(inplace=True)


def get_norm(channels, type="group_norm", groups=16, elementwise_affine=False, axis=1):
    if type is None:
        return nn.Identity()
    type = type.lower()
    if type == "batch_norm":
        norm = BatchNorm1d(channels)
    elif type == "layer_norm":
        norm = LayerNorm(channels, elementwise_affine)
    elif type == "group_norm":
        assert groups > 0
        norm = nn.GroupNorm(min(channels // 4, groups), channels, eps=1e-6)
    else:
        raise TypeError("norm not support")
    return norm


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


class LabelEmbedding(nn.Module):
    def __init__(self, num_categorys, dim_embed, dim_out):
        super(LabelEmbedding, self).__init__()
        self.label_emb = nn.Embedding(num_categorys, dim_embed)
        self.mlp = nn.Sequential(
            nn.Linear(dim_embed, dim_out), nn.SiLU(), nn.Linear(dim_out, dim_out)
        )
        self.t_emb_dim = dim_embed

    def forward(self, label):
        return self.mlp(self.label_emb(label))


class LayerNorm(nn.Module):
    def __init__(self, channels, elementwise_affine):
        super().__init__()
        self.norm = nn.LayerNorm(
            channels, elementwise_affine=elementwise_affine, eps=1e-6
        )

    def forward(self, x):
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class BatchNorm1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.BatchNorm1d(channels)

    def forward(self, x):
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class FinalLayer(nn.Module):
    def __init__(self, dim_in, dim_out, dim_c, norm):
        super(FinalLayer, self).__init__()
        self.norm = get_norm(
            dim_in,
            norm,
            16,
            elementwise_affine=True if dim_c is None else False,
            axis=2,
        )
        if dim_c is not None:
            self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim_c, 2 * dim_in))
        self.ln = nn.Conv1d(dim_in, dim_out, 1)

    def forward(self, x, c=None):
        if c is not None:
            c = c[:, None, :] if c.dim() == 2 else c.transpose(1, 2)
            shift, scale = self.adaLN(c).transpose(1, 2).chunk(2, dim=1)
            x = modulate(self.norm(x), shift, scale)
            x = self.ln(x)
        else:
            x = self.ln(self.norm(x))
        return x
