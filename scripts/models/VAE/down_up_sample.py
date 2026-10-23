import math

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import BlockMask
from utils.flex_attn import (
    MHFlexDecoderLayer,
)


def sinusoidal_pe(num_positions, dim, device):
    pe = torch.zeros(num_positions, dim, device=device)
    position = torch.arange(0, num_positions, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device) * (-math.log(10000.0) / dim)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class CrossAttnDownsample(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_queries,
        num_heads,
        max_outputs=None,
        device=None,
    ):
        super().__init__()
        self.num_queries = num_queries

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        # .to() returns an unregistered copy: fixed random queries, not trained or saved
        # (learnable queries gave worse reconstruction); copy=True makes CPU match CUDA
        self.queries = nn.Parameter(
            torch.randn(num_queries, embed_dim) * embed_dim**-0.5
        ).to(device, copy=True)
        self.register_buffer("query_pe", sinusoidal_pe(num_queries, embed_dim, device))

        self.decoder_attn = MHFlexDecoderLayer(
            embed_dim,
            num_heads,
            max_len_q=self.num_queries,
            max_len_kv=max_outputs,
            dim_feedforward=embed_dim * 4,
        )
        self.print_mask = True

    def forward(
        self,
        x,
        ctx_emb=None,
        block_mask: BlockMask = None,
        effective_p_dim: int = None,
    ):
        B, N, D = x.shape
        p = effective_p_dim if effective_p_dim is not None else self.num_queries
        q = self.queries[:p].unsqueeze(0).expand(B, -1, -1)
        q = q + self.query_pe[:p].unsqueeze(0)
        if ctx_emb is not None:
            q = q + ctx_emb
        out = self.decoder_attn(
            q,
            x,
            block_mask=block_mask,
            effective_p_dim=effective_p_dim,
        )
        return out


class CrossAttnUpsample(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_queries,  # max_outputs
        z_scales,
        num_heads,
        query_type="static",
        device=None,
    ):
        """query_type: one of ["static", "input_cond", "dynamic_comb", "struct_cond", "lang_cond"]"""

        super().__init__()
        self.query_type = query_type
        self.num_queries = num_queries

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.queries_up = nn.Parameter(  # see CrossAttnDownsample
            torch.randn(num_queries, embed_dim) * embed_dim**-0.5
        ).to(device, copy=True)

        self.gamma_zglobal = nn.Linear(embed_dim, embed_dim)
        self.beta_zglobal = nn.Linear(embed_dim, embed_dim)

        self.decoderUP_attn = MHFlexDecoderLayer(
            embed_dim,
            num_heads,
            max_len_q=self.num_queries,
            max_len_kv=z_scales,
            dim_feedforward=embed_dim * 4,
        )

        self.print_mask = True
        # Optional query generators
        if self.query_type == "dynamic_comb" or self.query_type == "struct_cond":
            self.query_mlp = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
            )

    def generate_queries(self, z, N, cond=None, structure_info=None, text_feat=None):
        B = z.size(0)

        if self.query_type == "static":
            # just slice fixed queries
            q = self.queries_up[:N].unsqueeze(0).expand(B, -1, -1)

        elif self.query_type == "input_cond":
            # like BLIP-2 / Q-Former
            q = self.queries_up.unsqueeze(0).expand(B, -1, -1)
            if cond is not None:
                cond_proj = cond.mean(dim=1, keepdim=True)
                q = q + cond_proj
            q = q[:, :N, :]

        elif self.query_type == "dynamic_comb":
            # like DQ-Det: combine base queries based on global input
            global_feat = z.mean(dim=1)  # (B, D)
            coeffs = torch.sigmoid(self.query_mlp(global_feat))  # (B, D)
            q = self.queries_up.unsqueeze(0) * coeffs.unsqueeze(
                1
            )  # (B, num_queries, D)
            q = q[:, :N, :]

        elif self.query_type == "struct_cond":
            # like TraCQ: condition queries on structure
            if structure_info is None:
                structure_info = z  # fallback
            q = self.query_mlp(structure_info.mean(dim=1, keepdim=True))  # (B, 1, D)
            q = q.expand(B, N, -1)

        elif self.query_type == "lang_cond":
            # like ReferFormer: condition queries on text
            if text_feat is None:
                text_feat = z  # fallback
            q = text_feat.mean(dim=1, keepdim=True)
            q = q.expand(B, N, -1)

        else:
            raise ValueError(f"Unknown query_type: {self.query_type}")

        return q

    def forward(
        self,
        z,
        N,
        o=None,
        q=None,
        y=None,
        ctx=None,
        pos_emb=None,
        block_mask: BlockMask = None,
        effective_p_dim: int = None,
    ):
        """
        z: (B, M, D)
        N: target number of points (<= num_queries)
        """
        B, M, D = z.shape

        if q is None:
            q = self.generate_queries(z, N, ctx, ctx, ctx)

        q = q + y if y is not None else q
        q = q + pos_emb if pos_emb is not None else q

        # add z in a global way (better than nothing)
        gamma_zglobal = self.gamma_zglobal(z.mean(dim=1)).unsqueeze(1)
        beta_zglobal = self.beta_zglobal(z.mean(dim=1)).unsqueeze(1)
        q = gamma_zglobal * q + beta_zglobal

        # attention for final upsampling
        q = self.decoderUP_attn(
            q,
            z,
            block_mask=block_mask,
            effective_p_dim_kv=effective_p_dim,
        )

        # add conditioning
        if ctx is not None:
            q = q + ctx
        return q
