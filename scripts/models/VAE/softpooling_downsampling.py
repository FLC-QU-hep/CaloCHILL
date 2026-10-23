import torch
import torch.nn as nn
import torch.nn.functional as F


# Step 0: PointNet-style embedding
class PointEmbed(nn.Module):
    def __init__(self, in_dim=3, embed_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        # x: [B, N, 3]
        return self.mlp(x)  # [B, N, embed_dim]


# Step 1: Farthest Point Sampling (FPS)
def farthest_point_sampling(pos, M):
    """
    pos: [B, N, 3]
    M: number of sampled points
    returns: indices [B, M]
    """
    B, N, _ = pos.shape
    idx = torch.zeros(B, M, dtype=torch.long, device=pos.device)
    dist = torch.ones(B, N, device=pos.device) * 1e10
    farthest = torch.randint(0, N, (B,), device=pos.device)

    for i in range(M):
        idx[:, i] = farthest
        centroid = pos[torch.arange(B), farthest].unsqueeze(1)
        d = ((pos - centroid) ** 2).sum(-1)
        dist = torch.minimum(dist, d)
        farthest = dist.max(-1).indices

    return idx


# Step 2: Soft pooling
def soft_pool(x, pos, M=1024, temperature=0.1):
    """
    x: [B, N, D] point features
    pos: [B, N, 3] positions
    M: target pooled points
    Returns:
        pooled_x: [B, M, D]
        pooled_pos: [B, M, 3]
    """
    B, N, D = x.shape
    idx = farthest_point_sampling(pos, M)  # [B, M]
    pooled_pos = pos[torch.arange(B).unsqueeze(1), idx]  # [B, M, 3]

    # Soft aggregation weights
    dists = torch.cdist(pooled_pos, pos)  # [B, M, N]
    weights = torch.softmax(-dists / temperature, dim=-1)  # [B, M, N]

    pooled_x = weights @ x  # [B, M, D]
    pooled_pos = weights @ pos  # smooth position aggregation
    return pooled_x, pooled_pos


# Step 3: Latent MLP (no attention)
class LatentMLP(nn.Module):
    def __init__(self, in_dim, latent_dim=128, num_latents=64, num_pooled=1024):
        super().__init__()
        self.num_latents = num_latents
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, latent_dim), nn.ReLU(), nn.Linear(latent_dim, latent_dim)
        )
        # Learnable soft aggregation weights
        self.weights = nn.Parameter(torch.randn(num_latents, num_pooled))

    def forward(self, x):
        """
        x: [B, M, D] pooled features
        Returns: [B, num_latents, latent_dim]
        """
        B, M, D = x.shape
        x_emb = self.mlp(x)  # [B, M, latent_dim]
        # Soft aggregation to latent queries with normalized learnable weights
        soft_w = F.softmax(self.weights, dim=-1)  # [num_latents, M]
        latent = torch.einsum(
            "bml, nm->bml", x_emb, soft_w
        )  # [B, num_latents, latent_dim]
        return latent


# Full Encoder: 3k -> 1k -> latent
class PointCloudDownsamplingNoAttn(nn.Module):
    def __init__(
        self, embed_dim=128, pooled_points=1024, latent_dim=128, num_latents=64
    ):
        super().__init__()
        self.embed = PointEmbed(in_dim=embed_dim, embed_dim=embed_dim)
        self.embed_y = PointEmbed(in_dim=8, embed_dim=embed_dim)
        self.pooled_points = pooled_points
        self.latent_mlp = LatentMLP(embed_dim, latent_dim, num_latents, pooled_points)

    def forward(self, x, y_emb=None, ctx_emb=None, padding=None):
        """
        x: [B, N, 3] input point cloud
        Returns:
            latent: [B, num_latents, latent_dim]
            pooled_x: [B, M, embed_dim]
            pooled_pos: [B, M, 3]
        """
        B, N, _ = x.shape
        x_embed = self.embed(x)  # [B, N, D]
        if y_emb is not None:
            x_embed = x_embed + self.embed_y(y_emb)

        # Step 2: soft pooling to 1k points
        pooled_x, pooled_pos = soft_pool(x_embed, x, M=self.pooled_points)

        if ctx_emb is not None:
            pooled_x = pooled_x + ctx_emb

        # Step 3: latent MLP aggregation (no attention)
        latent = self.latent_mlp(pooled_x)
        return latent
