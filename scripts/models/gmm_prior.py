import torch
import torch.nn as nn


class ConditionalGMMPrior(nn.Module):
    """Condition-dependent Gaussian-Mixture noise prior for the LMF EDM stage.

    Generalizes the isotropic eps ~ N(0, I) noise used for forward corruption /
    x_T initialization to eps = mu_k(c) + N(0, I), k ~ Categorical(pi(c)) — an
    additive per-component mean shift with unit variance, so the EDM
    preconditioning (derived assuming unit-variance noise) stays valid.
    """

    def __init__(self, cond_dim, z_dim, n_components, hidden_dim=128):
        super().__init__()
        self.z_dim = z_dim
        self.n_components = n_components
        self.trunk = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.to_pi = nn.Linear(hidden_dim, n_components)
        self.to_mu = nn.Linear(hidden_dim, n_components * z_dim)
        self.c = 0.9189385332  # 0.5 * log(2 * pi)

    def forward(self, cond_feats):
        h = self.trunk(cond_feats)
        pi = torch.softmax(self.to_pi(h), dim=-1)  # [B, K]
        mu = self.to_mu(h).view(-1, self.n_components, self.z_dim)  # [B, K, D]
        return pi, mu

    def sample_eps(self, like, cond_feats):
        """Draw eps shaped like `like` ([B, N, D]), one mixture component per point."""
        pi, mu = self.forward(cond_feats)  # [B, K], [B, K, D]
        B, N, D = like.shape
        component = torch.multinomial(pi, N, replacement=True)  # [B, N]
        mu_sel = torch.gather(
            mu, dim=1, index=component.unsqueeze(-1).expand(-1, -1, D)
        )  # [B, N, D]
        return mu_sel + torch.randn_like(like)

    def nll(self, x, cond_feats):
        """Mixture negative log-likelihood of latents x ([B, N, D]) given cond_feats."""
        pi, mu = self.forward(cond_feats)  # [B, K], [B, K, D]
        x = x.unsqueeze(2)  # [B, N, 1, D]
        mu = mu.unsqueeze(1)  # [B, 1, K, D]
        log_probs = (-0.5 * (x - mu) ** 2 - self.c).sum(dim=-1)  # [B, N, K]
        weighted = log_probs + torch.log(pi + 1e-10).unsqueeze(1)  # [B, N, K]
        log_px = torch.logsumexp(weighted, dim=-1)  # [B, N]
        return -log_px.mean()
