import torch
import torch._dynamo
from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
)

torch._dynamo.config.cache_size_limit = 1024
create_block_mask = torch.compile(create_block_mask, dynamic=True)

# flex-attention's block granularity; sequences are padded to a multiple of this
FLEX_BLOCK_SIZE = 128

# Up to this many dense entries (B * Q_LEN * KV_LEN) the compiled dense build beats the
# block-grid path, whose eager kernel launches cost a roughly flat overhead per call.
DENSE_MAX_ENTRIES = 2**24

# Granularity at which a batch's padded length is rounded up (multiple of FLEX_BLOCK_SIZE).
# Every distinct length is a new shape and costs a dynamo compile; coarser buckets trade
# a little extra padding for far fewer compiles.
MAX_LEN_BUCKET = 512

# Shared across BlockMaskCache instances on purpose: the LMF and the VAE decompressor
# build the latent mask from identical inputs. Keyed on shape/type, then confirmed by
# comparing the actual inputs, so a hit never returns a mask built for a different batch.
_MASK_MEMO = {}


def _memo_lookup(key, pad_q, pad_kv, neigh):
    hit = _MASK_MEMO.get(key)
    if hit is None:
        return None
    p_q, p_kv, n, block_mask = hit
    if (
        p_q.shape == pad_q.shape
        and p_kv.shape == pad_kv.shape
        and torch.equal(n, neigh)
        and torch.equal(p_q, pad_q)
        and torch.equal(p_kv, pad_kv)
    ):
        return block_mask
    return None


def _local_mask_mod(pad_q_flat, pad_kv_flat, neigh, B, Q_LEN, KV_LEN):
    """The per-entry mask flex_attention evaluates inside partial blocks."""
    valid_len_q, valid_len_kv = pad_q_flat.sum(axis=1), pad_kv_flat.sum(axis=1)
    if pad_q_flat.device.type == "cpu" and (
        _is_prefix(pad_q_flat, valid_len_q) and _is_prefix(pad_kv_flat, valid_len_kv)
    ):
        return _local_mask_mod_lengths(valid_len_q, valid_len_kv, neigh, Q_LEN, KV_LEN)
    neigh = neigh.unsqueeze(1).expand(B, max(Q_LEN, KV_LEN)).contiguous()

    if Q_LEN == KV_LEN:
        # self-attention: straightforward local window
        def mask_fn(b, h, q_idx, kv_idx):
            neighbor = torch.abs(q_idx - kv_idx) <= neigh[b, q_idx]
            padding = pad_q_flat[b, q_idx] & pad_kv_flat[b, kv_idx]
            return neighbor & padding

    elif Q_LEN < KV_LEN:

        def mask_fn(b, h, q_idx, kv_idx):
            scale = valid_len_kv[b] / valid_len_q[b]
            q_center = (q_idx.float() * scale).long()
            neighbor = torch.abs(kv_idx - q_center) <= neigh[b, q_idx]
            padding = pad_q_flat[b, q_idx] & pad_kv_flat[b, kv_idx]
            return neighbor & padding
    elif Q_LEN > KV_LEN:

        def mask_fn(b, h, q_idx, kv_idx):
            scale = valid_len_q[b] / valid_len_kv[b]
            kv_center = (kv_idx.float() * scale).long()
            neighbor = torch.abs(q_idx - kv_center) <= neigh[b, kv_idx]
            padding = pad_q_flat[b, q_idx] & pad_kv_flat[b, kv_idx]
            return neighbor & padding

    return mask_fn


def _local_mask_mod_lengths(n_q, n_kv, w, Q_LEN, KV_LEN):
    """_local_mask_mod for prefix padding, from the per-row valid lengths and window
    alone. Same mask: with prefix padding pad[b, i] is i < n[b], and the window is
    constant along a row. Used on CPU, where the torch 2.10 Inductor flex-attention
    template fails to compile (dynamic=True) any mask_mod that indexes a buffer by
    q_idx or kv_idx ("cur_kvSplitSize<N> was not declared"); per-row lookups compile."""
    if Q_LEN == KV_LEN:

        def mask_fn(b, h, q_idx, kv_idx):
            neighbor = torch.abs(q_idx - kv_idx) <= w[b]
            return neighbor & (q_idx < n_q[b]) & (kv_idx < n_kv[b])

    elif Q_LEN < KV_LEN:

        def mask_fn(b, h, q_idx, kv_idx):
            scale = n_kv[b] / n_q[b]
            q_center = (q_idx.float() * scale).long()
            neighbor = torch.abs(kv_idx - q_center) <= w[b]
            return neighbor & (q_idx < n_q[b]) & (kv_idx < n_kv[b])

    else:

        def mask_fn(b, h, q_idx, kv_idx):
            scale = n_q[b] / n_kv[b]
            kv_center = (kv_idx.float() * scale).long()
            neighbor = torch.abs(q_idx - kv_center) <= w[b]
            return neighbor & (q_idx < n_q[b]) & (kv_idx < n_kv[b])

    return mask_fn


@torch.compiler.disable()
def _make_local_mask_dense(pad_q_flat, pad_kv_flat, neigh, B, Q_LEN, KV_LEN, device):
    """Reference builder: evaluates the mask on all B * Q_LEN * KV_LEN entries and
    reduces it to blocks. At 90 GeV the last-layer mask alone is 128 * 5376^2 entries,
    ~40% of the decoder's sample time. Used for small masks and when the padding is
    not a prefix."""
    return create_block_mask(
        mask_mod=_local_mask_mod(pad_q_flat, pad_kv_flat, neigh, B, Q_LEN, KV_LEN),
        B=B,
        H=None,
        Q_LEN=Q_LEN,
        KV_LEN=KV_LEN,
        device=device,
    )


def _is_prefix(valid, n):
    """True when each row's valid positions are exactly its first n."""
    return torch.equal(
        valid, torch.arange(valid.shape[1], device=valid.device)[None] < n[:, None]
    )


def _dense_to_ordered(blocks):
    """Same layout as torch's flex_attention._dense_to_ordered, which create_block_mask
    hands to BlockMask.from_kv_blocks: per-row block counts and stable-sorted indices."""
    blocks = blocks.to(torch.int32)
    return (
        blocks.sum(dim=-1).to(torch.int32, memory_format=torch.contiguous_format),
        torch.argsort(blocks, dim=-1, descending=True, stable=True).to(
            torch.int32, memory_format=torch.contiguous_format
        ),
    )


def _window_blocks_self(n_q, n_kv, w, Q_LEN, KV_LEN, bs):
    """(any, full) [B, Q blocks, KV blocks] for |q - kv| <= w, q < n_q, kv < n_kv."""
    dev = n_q.device
    iq = torch.arange(-(-Q_LEN // bs), device=dev)
    ikv = torch.arange(-(-KV_LEN // bs), device=dev)
    # valid span of each block per row, [start, end]; empty when end < start
    q_start, kv_start = (iq * bs)[None, :, None], (ikv * bs)[None, None, :]
    q_end = torch.minimum(iq * bs + bs - 1, n_q[:, None] - 1)[:, :, None]
    kv_end = torch.minimum(ikv * bs + bs - 1, n_kv[:, None] - 1)[:, None, :]
    W = w[:, None, None]
    gap = torch.clamp(torch.maximum(kv_start - q_end, q_start - kv_end), min=0)
    any_ = (q_start <= q_end) & (kv_start <= kv_end) & (gap <= W)
    # full: both blocks entirely valid and the farthest pair, (|i - j| + 1) * bs - 1,
    # still inside the window
    span = ((iq[:, None] - ikv[None, :]).abs() + 1) * bs - 1
    full = (
        ((iq + 1) * bs <= n_q[:, None])[:, :, None]
        & ((ikv + 1) * bs <= n_kv[:, None])[:, None, :]
        & (span[None] <= W)
    )
    return any_, full


def _window_blocks_cross(n_s, n_l, w, S_LEN, L_LEN, bs):
    """(any, full) [B, short blocks, long blocks] for |l - floor(s * n_l / n_s)| <= w,
    s < n_s, l < n_l -- the Q_LEN != KV_LEN masks, s on the shorter axis.

    The kernel computes the centre in float32, which can differ from the exact
    floor(s * n_l / n_s) by one where the exact value sits within float error of an
    integer. There both possible centres are allowed: their union for `any` and their
    intersection for `full`, so no attended entry is ever outside a partial or full
    block. Elsewhere the result is exactly what the dense build gives."""
    dev = n_s.device
    s = torch.arange(S_LEN, device=dev)[None]
    num = s * n_l[:, None]
    den = n_s.clamp(min=1)[:, None]
    c, frac = num // den, num % den
    # n_l / n_s is exact in float32 iff its reduced denominator is a power of two
    red = (n_s.clamp(min=1) // torch.gcd(n_l, n_s.clamp(min=1)))[:, None]
    inexact = (red & (red - 1)) != 0
    tol = n_l[:, None] * den  # float error <= n_l * 2^-23, tested with margin 2^-22
    c_lo = c - (inexact & (s > 0) & (frac * 2**22 < tol)).long()
    c_hi = c + (inexact & ((den - frac) * 2**22 < tol)).long()

    W, last = w[:, None], n_l[:, None] - 1
    lo_any, hi_any = torch.clamp(c_lo - W, min=0), torch.minimum(c_hi + W, last)
    lo_full, hi_full = c_hi - W, torch.minimum(c_lo + W, last)
    start = (torch.arange(-(-L_LEN // bs), device=dev) * bs)[None, None]
    end = start + bs - 1
    valid = (s < n_s[:, None])[..., None]
    row_any = (
        valid
        & (lo_any[..., None] <= end)
        & (hi_any[..., None] >= start)
        & (lo_any <= hi_any)[..., None]
    )
    row_full = valid & (lo_full[..., None] <= start) & (hi_full[..., None] >= end)

    # rows -> short-axis blocks; padded rows count as False, as in the dense build
    n_sb = -(-S_LEN // bs)
    pad = (0, 0, 0, n_sb * bs - S_LEN)
    row_any = torch.nn.functional.pad(row_any, pad).view(-1, n_sb, bs, start.shape[-1])
    row_full = torch.nn.functional.pad(row_full, pad).view(
        -1, n_sb, bs, start.shape[-1]
    )
    return row_any.any(dim=2), row_full.all(dim=2)


@torch.compiler.disable()
def _make_local_mask(pad_q_flat, pad_kv_flat, neigh, B, Q_LEN, KV_LEN, device):
    """Dense build for small masks, block-grid build otherwise; both give the same
    BlockMask."""
    if B * Q_LEN * KV_LEN <= DENSE_MAX_ENTRIES:
        return _make_local_mask_dense(
            pad_q_flat, pad_kv_flat, neigh, B, Q_LEN, KV_LEN, device
        )
    return _make_local_mask_blocks(
        pad_q_flat, pad_kv_flat, neigh, B, Q_LEN, KV_LEN, device
    )


@torch.compiler.disable()
def _make_local_mask_blocks(pad_q_flat, pad_kv_flat, neigh, B, Q_LEN, KV_LEN, device):
    """Same BlockMask as _make_local_mask_dense, computed at block granularity: with
    prefix padding the window mask's block occupancy follows from n_q, n_kv and the
    window width, so the B * Q_LEN * KV_LEN evaluation is skipped."""
    valid_len_q, valid_len_kv = pad_q_flat.sum(axis=1), pad_kv_flat.sum(axis=1)
    if not (
        _is_prefix(pad_q_flat, valid_len_q) and _is_prefix(pad_kv_flat, valid_len_kv)
    ):
        return _make_local_mask_dense(
            pad_q_flat, pad_kv_flat, neigh, B, Q_LEN, KV_LEN, device
        )
    bs = FLEX_BLOCK_SIZE
    w = neigh.long()
    if Q_LEN == KV_LEN:
        any_, full = _window_blocks_self(
            valid_len_q, valid_len_kv, w, Q_LEN, KV_LEN, bs
        )
    elif Q_LEN < KV_LEN:
        any_, full = _window_blocks_cross(
            valid_len_q, valid_len_kv, w, Q_LEN, KV_LEN, bs
        )
    else:
        any_, full = _window_blocks_cross(
            valid_len_kv, valid_len_q, w, KV_LEN, Q_LEN, bs
        )
        any_, full = any_.transpose(1, 2), full.transpose(1, 2)
    partial = (any_ & ~full)[:, None]
    kv_num, kv_idx = _dense_to_ordered(partial)
    full_num, full_idx = _dense_to_ordered(full[:, None])
    return BlockMask.from_kv_blocks(
        kv_num,
        kv_idx,
        full_num,
        full_idx,
        BLOCK_SIZE=(bs, bs),
        mask_mod=_local_mask_mod(pad_q_flat, pad_kv_flat, neigh, B, Q_LEN, KV_LEN),
        seq_lengths=(Q_LEN, KV_LEN),
    )


class BlockMaskCache:
    def __init__(self, neighbors=30, p_dim=1024, k_scale=8, device="cuda"):
        self.neighbors = neighbors
        self.p_dim = p_dim
        self.k_scale = k_scale
        self.k_min = self.k_scale // 2
        self.device = device

    def clear(self):
        _MASK_MEMO.clear()

    def get(
        self,
        pad_q,
        pad_kv,
        max_len_q,
        max_len_kv,
        type,
        top_points_layers=None,
        n_points=None,
    ):
        if type == "local":
            neighbors_vector = torch.full(
                (pad_q.shape[0],),
                self.neighbors,
                device=pad_q.device,
                dtype=torch.int32,
            )
        elif type == "layer":  # block mask uses +- this number
            neighbors_vector = (top_points_layers // 2).to(torch.int32)
        elif type == "half_layer":
            neighbors_vector = (top_points_layers // 4).to(torch.int32)
        elif type == "p_dim_based":
            if n_points is None:
                raise ValueError("Point dim based local mask requires number of points")
            neighbors_vector = torch.clamp(
                torch.sqrt(n_points.float() / self.p_dim) * self.k_scale,
                min=torch.full_like(n_points.float(), self.k_min),
                max=torch.minimum(
                    n_points, torch.full_like(n_points, self.p_dim)
                ).float(),  # can't attend to more than what exists
            ).to(torch.int32)
        else:
            raise NotImplementedError("Layer-based local mask not implemented yet")

        pad_q_flat, pad_kv_flat = pad_q.flatten(1), pad_kv.flatten(1)
        key = (type, max_len_q, max_len_kv, pad_q.shape[0])
        block_mask = _memo_lookup(key, pad_q_flat, pad_kv_flat, neighbors_vector)
        if block_mask is None:
            block_mask = _make_local_mask(
                pad_q_flat,
                pad_kv_flat,
                neighbors_vector,
                pad_q.shape[0],
                max_len_q,
                max_len_kv,
                str(self.device),
            )
            _MASK_MEMO[key] = (pad_q_flat, pad_kv_flat, neighbors_vector, block_mask)
        return block_mask
