import torch
import torch._dynamo
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import (
    BlockMask,
    flex_attention,
)

torch._dynamo.config.cache_size_limit = 1024
flex_attention_compiled = torch.compile(flex_attention, dynamic=True)


def pad_to_max_len_fast_padding(padding, max_len):
    """
    Fast version of pad_to_max_len without repeated torch.cat.

    padding: (B, N)  True = PAD, False = VALID
    max_len: int

    Returns:
        padding_pad:(B, max_len)
    """
    B, N = padding.shape
    assert N <= max_len, "Sequence length exceeds max_len"

    if N == max_len:
        return padding

    padding_pad = torch.ones(B, max_len, device=padding.device, dtype=torch.bool)
    padding_pad[:, :N] = padding
    return padding_pad


def pad_to_max_len_fast_x(x, max_len):
    """
    Fast version of pad_to_max_len without repeated torch.cat.

    x:       (B, N, D)
    max_len: int

    Returns:
        x_pad:      (B, max_len, D)
    """
    B, N, D = x.shape
    assert N <= max_len, "Sequence length exceeds max_len"

    if N == max_len:
        return x

    x_pad = torch.zeros(B, max_len, D, device=x.device, dtype=x.dtype)
    x_pad[:, :N, :] = x

    return x_pad


class MHFlexEncoderLayer(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        max_len=6704,
        dim_feedforward=None,
        dropout=0.1,
        device="cuda",
        qk_norm=False,
        qk_norm_affine=True,
        track_qk_scale=False,
    ):
        super().__init__()
        self.max_len = max_len
        self.self_attn = FlexAttentionWithInternalPadding2(
            dim,
            num_heads,
            max_len_q=max_len,
            max_len_kv=max_len,
            device=device,
            qk_norm=qk_norm,
            qk_norm_affine=qk_norm_affine,
            track_qk_scale=track_qk_scale,
        )
        self.feed_forward = PositionwiseFeedForward(
            dim, dim_feedforward or 4 * dim, dropout=dropout
        )
        self.layer_norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x,
        block_mask: BlockMask = None,
        effective_p_dim: int = None,
    ):
        """
        x: (B, N, D)
        padding: (B, N) True=PAD, False=valid
        """
        Nq = x.shape[1]
        residual = x

        x = self.self_attn(
            Nq,
            query=x,
            block_mask=block_mask,
            effective_p_dim=effective_p_dim,
            effective_p_dim_kv=effective_p_dim,
        )
        x = self.dropout(x)
        x = x + residual
        x = self.layer_norm1(x)

        x = self.feed_forward(x)
        x = self.layer_norm2(x)
        return x


class MHFlexDecoderLayer(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        max_len_q=6704,
        max_len_kv=6704,
        dim_feedforward=None,
        dropout=0.1,
        self_attn=False,
        device="cuda",
    ):
        super().__init__()
        self.max_len_q = max_len_q
        self.max_len_kv = max_len_kv
        self.self_attn_enabled = self_attn
        if self.self_attn_enabled:
            self.self_attn = FlexAttentionWithInternalPadding2(
                dim,
                num_heads,
                max_len_q=max_len_q,
                max_len_kv=max_len_q,
                device=device,
            )
            self.layer_norm1 = nn.LayerNorm(dim, eps=1e-6)

        self.cross_attn = FlexAttentionWithInternalPadding2(
            dim,
            num_heads,
            max_len_q=max_len_q,
            max_len_kv=max_len_kv,
            device=device,
        )
        self.layer_norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.feed_forward = PositionwiseFeedForward(
            dim, dim_feedforward or 4 * dim, dropout=dropout
        )
        self.layer_norm3 = nn.LayerNorm(dim, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x,
        kv,
        block_mask: BlockMask = None,
        effective_p_dim: int = None,
        effective_p_dim_kv: int = None,
    ):
        """
        x: (B, Nq, D)       # target sequence
        memory: (B, Nk, D)  # encoder output
        padding_x: (B, Nq)  True=PAD for target
        padding_memory: (B, Nk) True=PAD for memory
        """
        Nq = x.shape[1]

        residual = x
        if self.self_attn_enabled:
            x = self.self_attn(
                Nq,
                query=x,
                block_mask=block_mask,
                effective_p_dim=effective_p_dim,
                effective_p_dim_kv=effective_p_dim,
            )
            x = self.dropout(x)
            x = x + residual
            x = self.layer_norm1(x)
            residual = x
        x = self.cross_attn(
            Nq,
            query=x,
            kv=kv,
            block_mask=block_mask,
            effective_p_dim=effective_p_dim,
            effective_p_dim_kv=effective_p_dim_kv,
        )
        x = self.dropout(x)
        x = x + residual
        x = self.layer_norm2(x)
        residual = x
        x = self.feed_forward(x)
        x = self.layer_norm3(x)
        return x


class FlexAttentionWithInternalPadding2(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        max_len_q: int = 6704,
        max_len_kv: int = 6704,
        device: str = "cuda",
        qk_norm: bool = False,
        qk_norm_affine: bool = True,
        track_qk_scale: bool = False,
    ):
        super().__init__()
        self.attn = MHFlexAttention(
            dim,
            num_heads,
            max_len_q=max_len_q,
            max_len_kv=max_len_kv,
            device=device,
            qk_norm=qk_norm,
            qk_norm_affine=qk_norm_affine,
            track_qk_scale=track_qk_scale,
        )

        self.max_len_q = max_len_q
        self.max_len_kv = max_len_kv
        self.num_heads = num_heads

    def forward(
        self,
        Nq,
        query,
        kv=None,
        block_mask: BlockMask = None,
        effective_p_dim: int = None,
        effective_p_dim_kv: int = None,
    ):
        if effective_p_dim is not None:
            query = pad_to_max_len_fast_x(query, max_len=effective_p_dim)
        else:
            query = pad_to_max_len_fast_x(query, max_len=self.max_len_q)

        if kv is not None:
            if effective_p_dim_kv is not None:
                kv = pad_to_max_len_fast_x(kv, max_len=effective_p_dim_kv)
            else:
                kv = pad_to_max_len_fast_x(kv, max_len=self.max_len_kv)
            out = self.attn(query, key=kv, value=kv, block_mask=block_mask)
        else:
            out = self.attn(query, block_mask=block_mask)
        return out[:, :Nq]


class MHFlexAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        max_len_q=6704,
        max_len_kv=6704,
        bias=False,
        return_lse=False,
        device="cuda",
        qk_norm=False,
        qk_norm_affine=True,
        track_qk_scale=False,
    ):
        super().__init__()
        assert dim % num_heads == 0

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.return_lse = return_lse
        self.max_len_q = max_len_q
        self.max_len_kv = max_len_kv
        self.track_qk_scale = track_qk_scale

        # RMS-normalise q and k per head so the attention logits cannot grow without bound
        # and saturate the softmax. qk_norm_affine=False also drops the (undecayed) RMSNorm
        # gains, fixing the logit scale at sqrt(head_dim); the default keeps them for strict loading.
        if qk_norm:
            self.q_norm = nn.RMSNorm(
                self.head_dim, eps=1e-6, elementwise_affine=qk_norm_affine
            )
            self.k_norm = nn.RMSNorm(
                self.head_dim, eps=1e-6, elementwise_affine=qk_norm_affine
            )
        else:
            self.q_norm = None
            self.k_norm = None

        self.q_proj = nn.Linear(dim, dim, bias=bias)
        self.k_proj = nn.Linear(dim, dim, bias=bias)
        self.v_proj = nn.Linear(dim, dim, bias=bias)
        self.out_proj = nn.Linear(dim, dim, bias=bias)

        # initialize to identity for stable training at the start
        nn.init.eye_(self.q_proj.weight)
        nn.init.eye_(self.k_proj.weight)
        nn.init.eye_(self.v_proj.weight)
        if self.q_proj.bias is not None:
            nn.init.zeros_(self.q_proj.bias)
        if self.k_proj.bias is not None:
            nn.init.zeros_(self.k_proj.bias)
        if self.v_proj.bias is not None:
            nn.init.zeros_(self.v_proj.bias)

        # Monitor: std of the attention logits, estimated as rms(q) * rms(k) * sqrt(head_dim).
        # Non-persistent so it stays out of the state_dict; allocated on CPU and moved by .to().
        self.register_buffer("qk_logit_scale", torch.zeros(()), persistent=False)

    def _reshape(self, x, B, N):
        return x.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        query,  # (B, Nq, D)
        key=None,  # (B, Nk, D)
        value=None,  # (B, Nk, D)
        block_mask=None,
    ):
        B, Nq, D = query.shape
        if key is None:
            key = query
            value = query
        Nk = key.shape[1]

        q = self._reshape(self.q_proj(query), B, Nq)
        k = self._reshape(self.k_proj(key), B, Nk)
        v = self._reshape(self.v_proj(value), B, Nk)

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if self.track_qk_scale:
            with torch.no_grad():
                self.qk_logit_scale.copy_(
                    q.detach().pow(2).mean().sqrt()
                    * k.detach().pow(2).mean().sqrt()
                    * self.head_dim**0.5
                )

        out_ = flex_attention_compiled(
            q,
            k,
            v,
            block_mask=block_mask,
            return_lse=self.return_lse,
        )

        if self.return_lse:
            out, lse = out_
            out = out.transpose(1, 2).contiguous().view(B, Nq, D)
            return self.out_proj(out), lse
        else:
            out_ = out_.transpose(1, 2).contiguous().view(B, Nq, D)
            return self.out_proj(out_)


class PositionwiseFeedForward(nn.Module):
    """A two-feed-forward-layer module"""

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x
