"""Toy sparse-attention masks figure (paper fig:attn): VAE encoder, LMF, VAE decoder.

usage: python plot_flex_attn_latent.py [out.pdf]
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Rectangle

N, N_PAD, DS = 1515, 2048, 256  # points, padded length, latent size
HW = 32.9  # compress half-width sqrt(W*N/ds), in points
W_LAT = 183  # latent self-attention half-width
VIOLET, PAD, HATCH = "#a992d4", "#e6e6e6", "#b3b3b3"
out = sys.argv[1] if len(sys.argv) > 1 else "flex_attn_latent.pdf"

plt.rcParams.update({"font.size": 11, "hatch.color": HATCH, "hatch.linewidth": 0.8})
fig = plt.figure(figsize=(12.4, 4.4))
# [left, bottom, width, height] in figure fractions; decoder is the tall one
ax_enc = fig.add_axes([0.07, 0.56, 0.25, 0.30])
ax_lmf = fig.add_axes([0.47, 0.40, 0.16, 0.46])
ax_dec = fig.add_axes([0.77, 0.12, 0.13, 0.74])


def padding(ax, xy, w, h):
    ax.add_patch(Rectangle(xy, w, h, fc=PAD, ec=HATCH, lw=0, hatch="////"))


def frame(ax, title, sub, xlabel, ylabel, xt, yt):
    ax.set_title(f"$\\bf{{{title}}}$\n{sub}", loc="left", fontsize=12, color="0.15")
    ax.set(
        xlabel=xlabel,
        ylabel=ylabel,
        xticks=xt,
        yticks=yt,
        xlim=(0, xt[-1]),
        ylim=(yt[-1], 0),
    )
    ax.tick_params(length=3)


q = np.linspace(0, DS, 400)
ax_enc.fill_betweenx(
    q,
    np.clip(q * N / DS - HW, 0, N),
    np.clip(q * N / DS + HW, 0, N),
    color=VIOLET,
    lw=0,
)
padding(ax_enc, (N, 0), N_PAD - N, DS)
frame(
    ax_enc,
    "VAE\\ encoder",
    "compress · latent ← full",
    "Full-length shower (KV)",
    "(Learnable)\nLatent vectors (Q)",
    [0, N, N_PAD],
    [0, DS],
)

ax_lmf.fill_betweenx(
    q, np.clip(q - W_LAT, 0, DS), np.clip(q + W_LAT, 0, DS), color=VIOLET, lw=0
)
frame(
    ax_lmf,
    "Latent\\ Mean\\ Flow\\ (LMF)",
    "latent self-attention",
    "Latent vectors (KV)",
    "Latent vectors (Q)",
    [0, DS],
    [0, DS],
)

ax_dec.fill_between(
    q,
    np.clip(q * N / DS - HW, 0, N),
    np.clip(q * N / DS + HW, 0, N),
    color=VIOLET,
    lw=0,
)
padding(ax_dec, (0, N), DS, N_PAD - N)
frame(
    ax_dec,
    "VAE\\ decoder",
    "decompress · full ← latent",
    "Latent vectors (KV)",
    "(Learnable) Full-res shower (Q)",
    [0, DS],
    [0, N, N_PAD],
)

for x in (0.365, 0.68):
    fig.text(x, 0.66, "→", fontsize=18, color="0.55", ha="center", va="center")
fig.legend(
    handles=[
        Patch(fc=VIOLET, label="Attended"),
        Patch(fc=PAD, ec=HATCH, lw=0, hatch="////", label="Padding"),
    ],
    loc="center",
    bbox_to_anchor=(0.35, 0.16),
    ncol=2,
    frameon=False,
    fontsize=12,
    handlelength=1.6,
    handleheight=1.2,
    columnspacing=2.5,
)
fig.savefig(out, bbox_inches="tight")
print("saved", out)
