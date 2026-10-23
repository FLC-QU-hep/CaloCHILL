import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from geomloss import SamplesLoss
from torch.nn import Linear, Module, init


def pad_tensor_to_max(
    tensor, max_shape, pad_dim, extra_dim_condition=None, pad_value=-100
):
    """
    Pads a tensor along a specified dimension with a constant value.

    Args:
        tensor:              The tensor to pad.
        max_shape:           The target size along pad_dim.
        pad_dim:             The dimension along which to pad.
        extra_dim_condition: Optional callable that returns the last dim size.
                             If None, no extra dim is appended to the pad shape.
        pad_value:           The fill value for the padding (default: -100).

    Returns:
        Concatenated tensor padded to max_shape along pad_dim.
    """
    pad_size = max_shape - tensor.shape[pad_dim]
    if pad_size <= 0:
        return tensor

    shape = list(tensor.shape)
    shape[pad_dim] = pad_size

    if extra_dim_condition is not None:
        shape.append(extra_dim_condition(tensor))

    padding = pad_value + torch.zeros(shape, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, padding], dim=pad_dim)


class KLDloss(nn.Module):
    def __init__(self):
        super(KLDloss, self).__init__()
        self.use_cuda = torch.cuda.is_available()

    def forward(self, mu, logvar):
        B = logvar.size(0)
        KLD = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())) / (B)
        return KLD


class ConcatSquashLinear(Module):
    def __init__(self, dim_in, dim_out, dim_ctx):
        super(ConcatSquashLinear, self).__init__()
        self._layer = Linear(dim_in, dim_out)
        self._hyper_bias = Linear(dim_ctx, dim_out, bias=False)
        self._hyper_gate = Linear(dim_ctx, dim_out)

    def forward(self, ctx, x):
        gate = torch.sigmoid(self._hyper_gate(ctx))
        bias = self._hyper_bias(ctx)
        ret = self._layer(x) * gate + bias
        return ret


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class Embedder(nn.Module):
    "Implement the fourier layer function."

    def __init__(self, d_model, **kwargs):
        super(Embedder, self).__init__()

        self.d = d_model
        self.kwargs = kwargs

        self.periodic_fns = [
            getattr(torch, fn.split(".")[1]) for fn in self.kwargs["periodic_fns"]
        ]
        self.create_embedding_fn()

    @staticmethod
    def apply_function(x, p_fn, freq):
        """Apply a periodic function with a given frequency."""
        return p_fn(x * freq)

    def create_embedding_fn(self):
        embed_fns = []
        out_dim = 0
        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += self.d

        if self.kwargs["log_sampling"]:
            freq_bands = 2.0 ** torch.linspace(
                0, self.kwargs["max_freq"], self.kwargs["N_freqs"]
            )
        else:
            freq_bands = torch.linspace(
                2.0**0.0, 2.0 ** self.kwargs["max_freq"], self.kwargs["N_freqs"]
            )

        for freq in freq_bands:
            for p_fn in self.periodic_fns:
                embed_fns.append(partial(self.apply_function, p_fn=p_fn, freq=freq))
                out_dim += self.d

        self.embed_fns = embed_fns
        self.out_dim = out_dim
        self.dim_final = len(self.embed_fns)

    def forward(self, x):
        return torch.cat([fn(x) for fn in self.embed_fns], dim=-1)


class ShapeMismatchError(RuntimeError):
    pass


def masked_fill(tensor_bnc, mask_bn=None, value=0.0):
    return (
        tensor_bnc.masked_fill(mask_bn.unsqueeze(-1), value)
        if (mask_bn is not None)
        else tensor_bnc
    )


def check(x):
    isinf = torch.isinf(x).any()
    isnan = torch.isnan(x).any()
    assert not (isinf or isnan), (
        f"Tensor of shape [{x.shape}] is isinf:{isinf} or isnan:{isnan}"
    )


class MaskedBatchNorm1d(nn.Module):
    """A masked version of nn.BatchNorm1d. Only tested for 3D inputs.
    Source https://gist.github.com/yangkky/364413426ec798589463a3a88be24219
    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, L)`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics and always uses batch
            statistics in both training and eval modes. Default: ``True``
    Shape:
        - x: (N, C, L)
        - x_mask: (N, L) tensor of ones and zeros, where the zeros indicate locations not to use.
        - Output: :math:`(N, C)` or :math:`(N, C, L)` (same shape as input)
    """

    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    ):
        super(MaskedBatchNorm1d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.Tensor(num_features, 1))
            self.bias = nn.Parameter(torch.Tensor(num_features, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.track_running_stats = track_running_stats
        if self.track_running_stats:
            self.register_buffer("running_mean", torch.zeros(1, num_features, 1))
            self.register_buffer("running_var", torch.ones(1, num_features, 1))
            self.register_buffer(
                "num_batches_tracked", torch.tensor(0, dtype=torch.long)
            )
        else:
            self.register_parameter("running_mean", None)
            self.register_parameter("running_var", None)
            self.register_parameter("num_batches_tracked", None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, x, x_mask=None):
        x_mask = (~x_mask).unsqueeze(1)
        # Calculate the masked mean and variance
        B, C, L = x.shape
        if x_mask is not None and x_mask.shape != (B, 1, L):
            raise ValueError("Mask should have shape (B, 1, L).")
        if C != self.num_features:
            raise ValueError(
                "Expected %d channels but input has %d channels"
                % (self.num_features, C)
            )
        if x_mask is not None:
            masked = x * x_mask
            n = x_mask.sum()
        else:
            masked = x
            n = B * L
        masked_sum = masked.sum(dim=0, keepdim=True).sum(dim=2, keepdim=True)
        current_mean = masked_sum / n
        current_var = ((masked - current_mean) ** 2).sum(dim=0, keepdim=True).sum(
            dim=2, keepdim=True
        ) / n
        # Update running stats
        if self.track_running_stats and self.training:
            if self.num_batches_tracked == 0:
                self.running_mean = current_mean
                self.running_var = current_var
            else:
                self.running_mean = (
                    1 - self.momentum
                ) * self.running_mean + self.momentum * current_mean
                self.running_var = (
                    1 - self.momentum
                ) * self.running_var + self.momentum * current_var
            self.num_batches_tracked += 1
        if self.track_running_stats and not self.training:
            normed = (masked - self.running_mean) / (
                torch.sqrt(self.running_var + self.eps)
            )
        else:
            normed = (masked - current_mean) / (torch.sqrt(current_var + self.eps))
        if self.affine:
            normed = normed * self.weight + self.bias
        return normed


def chamfer_loss_v2(output_set, target_set, valid_mask=None, return_dists=False):
    B, N, C = output_set.shape
    _, M, _ = target_set.shape
    large_value = 1e6

    dists = torch.cdist(output_set, target_set, p=2)  # [B, N, M]

    if valid_mask is not None:
        valid_mask = valid_mask.float()  # [B, N]

        # Expand masks for broadcasting
        mask1 = valid_mask.unsqueeze(2)  # [B, N, 1]
        mask2 = valid_mask.unsqueeze(1)  # [B, 1, M]
        combined_mask = mask1 * mask2  # [B, N, M]

        # Mask invalid distances by setting them to a large number
        dists_masked = dists.clone()
        dists_masked[combined_mask == 0] = large_value

        # Compute min distances (valid only)
        min_dist1 = dists_masked.min(dim=2)[0] * valid_mask  # [B, N]
        min_dist2 = dists_masked.min(dim=1)[0] * valid_mask  # [B, M]
        # Avoid division by zero
        eps = 1e-8
        loss_ot = (min_dist1.sum(dim=1) / (valid_mask.sum(dim=1) + eps)).mean()
        loss_to = (min_dist2.sum(dim=1) / (valid_mask.sum(dim=1) + eps)).mean()
    else:
        # standard chamfer loss
        min_dist1 = dists.min(dim=2)[0]
        min_dist2 = dists.min(dim=1)[0]
        loss_ot = min_dist1.mean()
        loss_to = min_dist2.mean()

    if return_dists:
        return (
            loss_ot + loss_to,
            min_dist1,
            min_dist2,
            dists.min(dim=2)[1],
            dists.min(dim=1)[1],
        )
    else:
        return loss_ot + loss_to, loss_ot, loss_to


def distChamfer(a, b):
    x, y = a, b
    bs, num_points, points_dim = x.size()
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))
    diag_ind = torch.arange(0, num_points).to(a).long()
    rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
    ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
    P = rx.transpose(2, 1) + ry - 2 * zz
    return P.min(1)[0], P.min(2)[0]


def L2_ChamferEval_1000(array1, array2):
    dist1, dist2 = distChamfer(array1, array2)
    dist = torch.mean(dist1) + torch.mean(dist2)
    return dist * 1000


def F1Score(array1, array2, threshold=0.001):
    dist1, dist2 = distChamfer(array1, array2)
    precision_1 = torch.mean((dist1 < threshold).float(), dim=1)
    precision_2 = torch.mean((dist2 < threshold).float(), dim=1)
    fscore = 2 * precision_1 * precision_2 / (precision_1 + precision_2)
    fscore[torch.isnan(fscore)] = 0
    return fscore, precision_1, precision_2


def dcd_loss_from_v2(
    output_set, target_set, alpha=1000, n_lambda=1.0, valid_mask=None, non_reg=False
):
    B, N, _ = output_set.shape
    _, M, _ = target_set.shape

    _, dist1, dist2, idx1, idx2 = chamfer_loss_v2(
        output_set, target_set, valid_mask=valid_mask, return_dists=True
    )

    if non_reg:
        frac_12 = max(1, N / M)
        frac_21 = max(1, M / N)
    else:
        frac_12 = N / M
        frac_21 = M / N

    exp_dist1 = torch.exp(-alpha * dist1**2)  # [B, N]
    exp_dist2 = torch.exp(-alpha * dist2**2)  # [B, M]

    loss1, loss2 = [], []

    for b in range(B):
        # Count how often each target point is matched (for source -> target)
        count1 = torch.bincount(idx1[b], minlength=M).float()
        weight1 = count1[idx1[b]].pow(n_lambda).clamp_min(1e-6)
        weight1 = 1.0 / weight1 * frac_21
        term1 = (1.0 - weight1 * exp_dist1[b]).mean()
        loss1.append(term1)

        # Count how often each predicted point is matched (for target -> source)
        count2 = torch.bincount(idx2[b], minlength=N).float()
        weight2 = count2[idx2[b]].pow(n_lambda).clamp_min(1e-6)
        weight2 = 1.0 / weight2 * frac_12
        term2 = (1.0 - weight2 * exp_dist2[b]).mean()
        loss2.append(term2)

    loss1 = torch.stack(loss1)
    loss2 = torch.stack(loss2)

    loss = 0.5 * (loss1 + loss2)

    return loss.mean(), loss1.mean(), loss2.mean()


def sliced_wasserstein_per_layer_fast(
    output_set,  # (B, N, D)
    target_set,  # (B, N, D)
    layer_cond,  # (B, N) or (B, N, 1)
    valid_mask,  # (B, N)
    num_projections=50,
    p=2,
    directions=None,  # (P, D) fixed unit directions; None -> num_projections random ones
):
    B, N, D = output_set.shape
    device = output_set.device

    if layer_cond.ndim == 3:
        layer_cond = layer_cond.squeeze(-1)

    # Sample projections once
    if directions is None:
        P = num_projections
        directions = torch.randn(P, D, device=device)
        directions = directions / (directions.norm(dim=1, keepdim=True) + 1e-8)
    else:
        P = directions.shape[0]

    # Project entire batch
    proj_x = output_set @ directions.T  # (B, N, P)
    proj_y = target_set @ directions.T

    # Flatten batch dimension
    proj_x = proj_x.reshape(-1, P)  # (B*N, P)
    proj_y = proj_y.reshape(-1, P)

    layers = layer_cond.reshape(-1)  # (B*N)
    mask = valid_mask.reshape(-1)  # (B*N)

    layers = layers[mask]
    proj_x = proj_x[mask]
    proj_y = proj_y[mask]

    unique_layers = torch.unique(layers)

    swd_values = []
    # layer_cond is 0-based and valid_mask already drops padding, so no label-0 filter
    for layer in unique_layers:
        layer_mask = layers == layer
        if layer_mask.sum() < 2:
            continue

        x_l = proj_x[layer_mask]
        y_l = proj_y[layer_mask]

        # sort along point dimension
        x_sorted, _ = torch.sort(x_l, dim=0)
        y_sorted, _ = torch.sort(y_l, dim=0)

        min_len = min(x_sorted.shape[0], y_sorted.shape[0])

        diff = (x_sorted[:min_len] - y_sorted[:min_len]).abs() ** p
        swd = diff.mean()

        swd_values.append(swd)

    if len(swd_values) == 0:
        return torch.tensor(0.0, device=device)

    return torch.stack(swd_values).mean()


def sliced_wasserstein_per_shower_layer(
    output_set,  # (B, N, D)
    target_set,  # (B, N, D)
    layer_cond,  # (B, N) or (B, N, 1), 0-based layers
    valid_mask,  # (B, N)
    num_projections=50,
    p=2,
    directions=None,  # (P, D) fixed unit directions; None -> num_projections random ones
):
    """Like sliced_wasserstein_per_layer_fast, but points are matched within each
    (shower, layer) instead of across the whole batch, so every reconstruction is
    compared with its own shower. Output and target share layer_cond and valid_mask,
    so each group has the same size on both sides: sort the values, then stable-sort
    by group id, and position i on both sides is the i-th smallest of the same group.
    Same weighting as the batch-pooled version: mean over points per layer, then over layers."""
    B, N, D = output_set.shape
    if layer_cond.ndim == 3:
        layer_cond = layer_cond.squeeze(-1)
    if directions is None:
        directions = torch.randn(num_projections, D, device=output_set.device)
        directions = directions / (directions.norm(dim=1, keepdim=True) + 1e-8)

    mask = valid_mask.reshape(-1)
    layers = layer_cond.reshape(-1)[mask].long()
    shower = torch.arange(B, device=output_set.device).repeat_interleave(N)[mask]
    L = 128  # group id = shower * L + layer is unique for any L > 77 (78 layers); a constant avoids a GPU sync on layers.max()
    group = shower * L + layers  # (M,)

    def sort_in_groups(s):  # (M, P) -> sorted within each group, groups in id order
        vals, idx = torch.sort(s, dim=0)
        _, order = torch.sort(group[idx], dim=0, stable=True)
        return torch.gather(vals, 0, order)

    x = sort_in_groups(
        (output_set @ directions.T).reshape(-1, directions.shape[0])[mask]
    )
    y = sort_in_groups(
        (target_set @ directions.T).reshape(-1, directions.shape[0])[mask]
    )
    per_point = ((x - y).abs() ** p).mean(dim=1)  # (M,), rows in group order

    layer_rows = torch.sort(group)[0] % L  # layer of each row, rows are in group order
    n_l = torch.bincount(layer_rows, minlength=L)
    s_l = per_point.new_zeros(L).index_add(
        0, layer_rows, per_point
    )  # bincount has no gradient
    present = n_l > 0
    return (s_l[present] / n_l[present]).mean()


def part_chamfer_distance(pred, target, layer_index):
    """
    pred: (B, N, 3)
    target: (B, N, 3)
    layer_index: (B, N, 1)
    """

    B, N, _ = pred.shape

    layer_index = layer_index.squeeze(-1)  # (B, N)

    # Pairwise distances (B, N, N)
    dist = torch.cdist(pred, target, p=2) ** 2

    # Build layer mask: only compare points in same layer
    layer_eq = layer_index.unsqueeze(2) == layer_index.unsqueeze(1)  # (B, N, N)

    # Mask out cross-layer distances
    BIG = 1e10
    dist_masked = torch.where(layer_eq, dist, BIG)

    # Chamfer components
    min_pred = dist_masked.min(dim=2).values  # (B, N)
    min_tgt = dist_masked.min(dim=1).values  # (B, N)

    # Only keep valid points (i.e., points that had at least one match)
    valid_pred = layer_eq.sum(dim=2) > 0
    valid_tgt = layer_eq.sum(dim=1) > 0

    min_pred = min_pred * valid_pred
    min_tgt = min_tgt * valid_tgt

    # Normalize per batch (avoid bias from different counts)
    pred_norm = valid_pred.sum(dim=1).clamp(min=1)
    tgt_norm = valid_tgt.sum(dim=1).clamp(min=1)

    loss_per_batch = (
        min_pred.sum(dim=1) / pred_norm + min_tgt.sum(dim=1) / tgt_norm
    ) / 2.0

    return loss_per_batch.mean()


class EBSW(nn.Module):
    def __init__(self, L=100, device="cuda"):
        super().__init__()
        self.L = L
        self.device = device
        self.p = 2
        self.T = 100
        self.eps = 0
        self.kappa = 1.0
        self.rho = 2
        self.M = 0
        self.N = 1
        self.s_lr = 1e-4

    def minibatch_rand_projections(
        self, batchsize, dim, num_projections=64, device="cuda"
    ):
        projections = torch.randn((batchsize, num_projections, dim), device=device)
        projections = projections / torch.sqrt(
            torch.sum(projections**2, dim=2, keepdim=True)
        )
        return projections

    def compute_projected_distances(self, x, y, projections, degree=2.0):
        """
        x, y: [batch_size, num_points, dim]
        num_projections: integer number
        """
        # projs.shape: [batchsize, num_projs, dim]

        xproj = x.bmm(projections.transpose(1, 2))

        yproj = y.bmm(projections.transpose(1, 2))

        _sort = (
            torch.sort(xproj.transpose(1, 2))[0] - torch.sort(yproj.transpose(1, 2))[0]
        )

        _sort_pow_p_get_sum = torch.sum(torch.pow(torch.abs(_sort), degree), dim=2)

        return _sort_pow_p_get_sum

    def forward(self, x, y, **kwargs):
        dim = x.size(2)
        batch_size = x.size(0)

        projections = self.minibatch_rand_projections(
            batch_size, dim, self.L, device=self.device
        )
        distances = self.compute_projected_distances(x, y, projections)
        weights = torch.softmax(distances, dim=1)
        weights = weights.detach()  # gradient type =  indipendent

        return torch.pow(torch.sum(weights * distances, dim=1), 1.0 / self.p).mean(
            dim=0
        )


def l2_loss_per_layer_per_shower(out, tar, layer, reduction="mean"):
    """
    out, tar : [B, N, 3]
    layer    : [B, N]  (int64)
    """
    # squared L2 distance per point
    # [B, N]
    sq_err = ((out - tar) ** 2).sum(dim=-1)

    B, N = layer.shape
    L = int(layer.max()) + 1

    loss = torch.zeros(B, L, device=out.device)

    loss.scatter_reduce_(
        dim=1,
        index=layer,
        src=sq_err,
        reduce="mean" if reduction == "mean" else "sum",
        include_self=False,
    )

    return loss.mean()


def sinkhorn_ot_cost(x, y, eps=0.05, n_iters=50):
    """
    x: (N, D)
    y: (M, D)
    returns: scalar OT cost
    """
    N, M = x.size(0), y.size(0)

    # Cost matrix: (N, M)
    C = torch.cdist(x, y, p=2)  # Euclidean
    K = torch.exp(-C / eps)
    with torch.no_grad():  # fast iterations
        mu = torch.full((N,), 1.0 / N, device=x.device)
        nu = torch.full((M,), 1.0 / M, device=x.device)

        u = torch.ones_like(mu)
        v = torch.ones_like(nu)

        for _ in range(n_iters):
            u = mu / (K @ v + 1e-18)
            v = nu / (K.t() @ u + 1e-18)

        T = u[:, None] * K * v[None, :]  # Transport plan
    loss = torch.sum(T * C)
    return loss


def ot_per_layer_loss(
    pred,
    target,
    layer_index,
    num_layers=None,
    eps=0.05,
    n_iters=10,
):
    """
    pred:        (B, N, D)
    target:      (B, N, D)
    layer_index: (B, N) integer layer ids
    """
    B, N, D = pred.shape

    if num_layers is None:
        num_layers = int(layer_index.max().item()) + 1

    total_loss = 0.0
    valid_layers = 0

    for b in range(B):
        for ll in range(num_layers):
            mask = layer_index[b] == ll
            if mask.sum() < 2:
                continue

            x = pred[b, mask]  # (Nl, D)
            y = target[b, mask]  # (Nl, D)

            if x.std() < 1e-8 and y.std() < 1e-8:
                continue
            total_loss = total_loss + sinkhorn_ot_cost(x, y, eps=eps, n_iters=n_iters)
            valid_layers += 1

    if valid_layers > 0:
        total_loss = total_loss / valid_layers

    return total_loss


class PerLayerOTLoss(torch.nn.Module):
    def __init__(self, loss: str = "sinkhorn", p: int = 2, blur: float = 0.05):
        super().__init__()
        self.loss_fn = SamplesLoss(
            loss="sinkhorn",
            p=p,
            blur=blur,
            debias=True,
            scaling=0.9,
            diameter=10.0,
            backend="tensorized",  # "online" or "tensorized"
        )

    def forward(
        self,
        pred: torch.Tensor,  # (B, N, D)
        target: torch.Tensor,  # (B, N, D)
        layer_index: torch.Tensor,  # (B, N)
        valid_mask: torch.Tensor,  # (B, N)
        per_layer: bool = True,  # if False, compute global Sinkhorn per batch
        num_projections: int = None,  # optional low-D projection
    ) -> torch.Tensor:
        """
        Compute Sinkhorn distance per-layer or globally per batch.
        """
        B, N, D = pred.shape
        device = pred.device
        total_loss = []

        if per_layer:
            B, N, D = pred.shape
            device = pred.device
            mask = valid_mask
            x_all = pred[mask]  # (total_points, D)
            y_all = target[mask]
            layers = layer_index[mask]  # (total_points,)

            if x_all.numel() == 0:
                return torch.tensor(0.0, device=device)

            sorted_layers, perm = torch.sort(layers)
            x_all = x_all[perm]
            y_all = y_all[perm]

            unique_layers, counts = torch.unique_consecutive(
                sorted_layers, return_counts=True
            )

            # Keep only layers with >=2 points
            valid_layers_mask = counts >= 2
            counts = counts[valid_layers_mask]

            if counts.numel() == 0:
                return torch.tensor(0.0, device=device)

            # Rebuild starts after filtering — need to also filter x_all/y_all rows
            # to only include rows belonging to valid layers
            valid_layer_ids = unique_layers[valid_layers_mask]
            row_mask = torch.isin(sorted_layers, valid_layer_ids)
            x_all = x_all[row_mask]
            y_all = y_all[row_mask]

            L = counts.shape[0]
            max_points = counts.max()

            starts = torch.cumsum(
                torch.cat([torch.tensor([0], device=device), counts[:-1]]), dim=0
            )  # (L,)

            # Build index tensors vectorized — no Python loop
            # arange over max_points, compare against counts to build the padding mask
            col_idx = (
                torch.arange(max_points, device=device).unsqueeze(0).expand(L, -1)
            )  # (L, max_points)
            count_col = counts.unsqueeze(1)  # (L, 1)
            valid_mask_batch = col_idx < count_col  # (L, max_points)

            # Flat source indices: for each (i, j), source row = starts[i] + j (clamped for safety)
            flat_idx = (starts.unsqueeze(1) + col_idx).clamp(
                max=x_all.shape[0] - 1
            )  # (L, max_points)

            x_batch = x_all[flat_idx] * valid_mask_batch.unsqueeze(
                -1
            )  # (L, max_points, D)
            y_batch = y_all[flat_idx] * valid_mask_batch.unsqueeze(
                -1
            )  # (L, max_points, D)

            weights = (
                valid_mask_batch.float() / count_col.float()
            )  # (L, max_points), uniform 1/c per row

            loss = self.loss_fn(weights, x_batch, weights, y_batch)
            return loss.mean()

        else:
            # --- GLOBAL (per batch) computation ---
            for b in range(B):
                m = valid_mask[b]
                if m.sum() < 2:
                    continue
                x_b = pred[b][m]
                y_b = target[b][m]

                # Optional projection
                if num_projections is not None and num_projections < D:
                    proj = torch.randn((D, num_projections), device=device)
                    proj = proj / (proj.norm(dim=0, keepdim=True) + 1e-8)
                    x_b = x_b @ proj
                    y_b = y_b @ proj

                    a = torch.ones(x_b.shape[0], 1, device=device) / x_b.shape[0]
                    b_ = torch.ones(y_b.shape[0], 1, device=device) / y_b.shape[0]
                    total_loss.append(self.loss_fn(x_b, y_b, a, b_))
                else:
                    total_loss.append(self.loss_fn(x_b, y_b))
        if len(total_loss) == 0:
            return torch.tensor(0.0, device=device)
        return torch.stack(total_loss).mean()


if __name__ == "__main__":
    import os
    import random
    import time

    import numpy as np
    import torch
    from flash_sinkhorn import SamplesLoss

    seed = 42
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

    eps = 0.05
    B, N, D = 2, 10000, 2
    num_layers = 5
    pred = torch.rand(B, N, D, device="cuda", requires_grad=True)

    # Create target as a permutation of pred along the points dimension
    target = torch.stack([p[torch.randperm(N)] for p in pred]).to("cuda")

    layer_index = torch.randint(0, num_layers, (B, N), device="cuda")

    # Compute OT loss per layer
    s = time.time()
    loss = ot_per_layer_loss(
        pred, target, layer_index, num_layers=num_layers, eps=eps, n_iters=10
    )
    e = time.time()
    print("OT loss per layer:", loss.item(), " time: ", e - s)

    B, N, D = pred.shape
    loss_fn = SamplesLoss(loss="sinkhorn", blur=np.sqrt(eps), n_iters=10, debias=False)
    s = time.time()
    total_loss = ot_per_layer_loss(
        loss_fn, pred, target, layer_index, num_layers=num_layers
    )
    e = time.time()
    print("fast OT loss per layer:", total_loss.item(), " time: ", e - s)

    total_loss.backward(retain_graph=True)
    print("fast ot loss", total_loss.item())

    loss.backward(retain_graph=True)
    print("ot loss", loss.item())
