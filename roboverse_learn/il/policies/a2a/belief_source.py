from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiagGaussianBeliefSource(nn.Module):
    """Diagonal Gaussian belief source for Belief-to-Action flow matching."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 512,
        use_prev_latent: bool = False,
        residual_mean: bool = True,
        min_log_std: float = -4.0,
        max_log_std: float = 0.0,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.use_prev_latent = use_prev_latent
        self.residual_mean = residual_mean
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

        input_dim = 2 * latent_dim
        if use_prev_latent:
            input_dim += latent_dim

        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.delta_head = nn.Linear(hidden_dim, latent_dim)
        self.gate_head = nn.Linear(hidden_dim, latent_dim)
        self.logstd_head = nn.Linear(hidden_dim, latent_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

        # Start as a near-A2A source: mu equals history_latents and std is exp(-2).
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, -2.0)
        nn.init.zeros_(self.logstd_head.weight)
        nn.init.constant_(self.logstd_head.bias, -2.0)

    def forward(
        self,
        history_latents: torch.Tensor,
        obs_latents: torch.Tensor,
        prev_latents: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        inputs = [history_latents, obs_latents]

        if self.use_prev_latent:
            if prev_latents is None:
                prev_latents = torch.zeros_like(history_latents)
            inputs.append(prev_latents)

        x = torch.cat(inputs, dim=-1)
        h = self.trunk(x)

        delta = self.delta_head(h)
        gate = torch.sigmoid(self.gate_head(h))

        if self.residual_mean:
            mu = history_latents + gate * delta
        else:
            mu = delta

        log_std = self.logstd_head(h)
        log_std = torch.clamp(log_std, self.min_log_std, self.max_log_std)
        std = torch.exp(log_std)

        if deterministic:
            z = mu
        else:
            z = mu + std * torch.randn_like(std)

        stats = {
            "mu": mu,
            "log_std": log_std,
            "std": std,
            "gate": gate,
        }
        return z, stats

    def belief_losses(
        self,
        target_latents: torch.Tensor,
        stats: Dict[str, torch.Tensor],
        entropy_floor: float = 0.02,
    ) -> Dict[str, torch.Tensor]:
        """Return belief-source losses and diagnostics."""
        mu = stats["mu"]
        log_std = stats["log_std"]
        std = stats["std"]
        target = target_latents.detach()

        raw_nll = 0.5 * (((target - mu) / (std + 1e-6)) ** 2 + 2.0 * log_std).mean()
        # Shift by a constant so the reported/weighted loss is non-negative.
        # This preserves gradients because log_std is clamped to min_log_std or above.
        nll = raw_nll - self.min_log_std
        w2_proxy = ((mu - target) ** 2 + std**2).mean()

        avg_std = std.mean()
        entropy_floor_loss = F.relu(entropy_floor - avg_std).pow(2)

        return {
            "belief_nll": nll,
            "belief_nll_raw": raw_nll.detach(),
            "belief_w2_proxy": w2_proxy,
            "belief_entropy_floor_loss": entropy_floor_loss,
            "belief_mean_error": ((mu - target) ** 2).mean().detach(),
            "belief_avg_std": avg_std.detach(),
            "belief_avg_log_std": log_std.mean().detach(),
            "belief_avg_gate": stats["gate"].mean().detach(),
        }



class KalmanGaussianBeliefSource(nn.Module):
    """
    Diagonal Gaussian posterior belief source for Kalman-style B2A.

    The module treats A2A's history latent as a prior mean and learns a
    visual-proprioceptive innovation likelihood. The source is the closed-form
    product of two diagonal Gaussians:

        q_h(z | h) = N(h, diag(sigma_h^2))
        q_v(z | h, o) = N(h + delta_v, diag(sigma_v^2))
        q_b(z | h, o) proportional to q_h(z | h) q_v(z | h, o)

    This makes the correction reliability explicit through a Kalman-gain-like
    precision ratio instead of adding another hand-shaped source loss.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 512,
        use_prev_latent: bool = False,
        min_log_std: float = -4.0,
        max_log_std: float = 0.0,
        prior_log_std_init: float = -1.5,
        visual_log_std_init: float = -1.5,
        innovation_gate_init: float = -2.0,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.use_prev_latent = use_prev_latent
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        self.prior_log_std_init = prior_log_std_init
        self.visual_log_std_init = visual_log_std_init
        self.innovation_gate_init = innovation_gate_init

        input_dim = 2 * latent_dim
        if use_prev_latent:
            input_dim += latent_dim

        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.visual_delta_head = nn.Linear(hidden_dim, latent_dim)
        self.visual_gate_head = nn.Linear(hidden_dim, latent_dim)
        self.prior_logstd_head = nn.Linear(hidden_dim, latent_dim)
        self.visual_logstd_head = nn.Linear(hidden_dim, latent_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

        # Initialize as a conservative A2A-like posterior: both component means
        # agree with history_latents, and uncertainty is learned from there.
        nn.init.zeros_(self.visual_delta_head.weight)
        nn.init.zeros_(self.visual_delta_head.bias)
        nn.init.zeros_(self.visual_gate_head.weight)
        nn.init.constant_(self.visual_gate_head.bias, self.innovation_gate_init)
        nn.init.zeros_(self.prior_logstd_head.weight)
        nn.init.constant_(self.prior_logstd_head.bias, self.prior_log_std_init)
        nn.init.zeros_(self.visual_logstd_head.weight)
        nn.init.constant_(self.visual_logstd_head.bias, self.visual_log_std_init)

    def forward(
        self,
        history_latents: torch.Tensor,
        obs_latents: torch.Tensor,
        prev_latents: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        inputs = [history_latents, obs_latents]

        if self.use_prev_latent:
            if prev_latents is None:
                prev_latents = torch.zeros_like(history_latents)
            inputs.append(prev_latents)

        features = self.trunk(torch.cat(inputs, dim=-1))

        history_mu = history_latents
        innovation_gate = torch.sigmoid(self.visual_gate_head(features))
        innovation = innovation_gate * self.visual_delta_head(features)
        visual_mu = history_latents + innovation

        history_log_std = torch.clamp(
            self.prior_logstd_head(features),
            self.min_log_std,
            self.max_log_std,
        )
        visual_log_std = torch.clamp(
            self.visual_logstd_head(features),
            self.min_log_std,
            self.max_log_std,
        )

        history_precision = torch.exp(-2.0 * history_log_std)
        visual_precision = torch.exp(-2.0 * visual_log_std)
        posterior_precision = history_precision + visual_precision
        posterior_var = torch.reciprocal(posterior_precision + 1e-8)
        posterior_mu = posterior_var * (
            history_precision * history_mu + visual_precision * visual_mu
        )

        posterior_log_std = 0.5 * torch.log(posterior_var + 1e-8)
        posterior_log_std = torch.clamp(
            posterior_log_std,
            self.min_log_std,
            self.max_log_std,
        )
        posterior_std = torch.exp(posterior_log_std)

        if deterministic:
            z = posterior_mu
        else:
            z = posterior_mu + posterior_std * torch.randn_like(posterior_std)

        kalman_gain = visual_precision / (posterior_precision + 1e-8)
        stats = {
            "mu": posterior_mu,
            "log_std": posterior_log_std,
            "std": posterior_std,
            "gate": kalman_gain,
            "history_mu": history_mu,
            "visual_mu": visual_mu,
            "history_log_std": history_log_std,
            "visual_log_std": visual_log_std,
            "history_std": torch.exp(history_log_std),
            "visual_std": torch.exp(visual_log_std),
            "innovation": innovation,
            "innovation_gate": innovation_gate,
            "kalman_gain": kalman_gain,
        }
        return z, stats

    def belief_losses(
        self,
        target_latents: torch.Tensor,
        stats: Dict[str, torch.Tensor],
        entropy_floor: float = 0.02,
    ) -> Dict[str, torch.Tensor]:
        """Return posterior belief losses plus Kalman diagnostics."""
        mu = stats["mu"]
        log_std = stats["log_std"]
        std = stats["std"]
        target = target_latents.detach()

        raw_nll = 0.5 * (((target - mu) / (std + 1e-6)) ** 2 + 2.0 * log_std).mean()
        nll = raw_nll - self.min_log_std
        w2_proxy = ((mu - target) ** 2 + std**2).mean()

        avg_std = std.mean()
        entropy_floor_loss = F.relu(entropy_floor - avg_std).pow(2)

        history_error = ((stats["history_mu"] - target) ** 2).mean().detach()
        visual_error = ((stats["visual_mu"] - target) ** 2).mean().detach()

        return {
            "belief_nll": nll,
            "belief_nll_raw": raw_nll.detach(),
            "belief_w2_proxy": w2_proxy,
            "belief_entropy_floor_loss": entropy_floor_loss,
            "belief_mean_error": ((mu - target) ** 2).mean().detach(),
            "belief_avg_std": avg_std.detach(),
            "belief_avg_log_std": log_std.mean().detach(),
            "belief_avg_gate": stats["kalman_gain"].mean().detach(),
            "kalman_history_error": history_error,
            "kalman_visual_error": visual_error,
            "kalman_history_avg_std": stats["history_std"].mean().detach(),
            "kalman_visual_avg_std": stats["visual_std"].mean().detach(),
            "kalman_gain_mean": stats["kalman_gain"].mean().detach(),
            "kalman_gain_min": stats["kalman_gain"].min().detach(),
            "kalman_gain_max": stats["kalman_gain"].max().detach(),
            "kalman_innovation_mse": (stats["innovation"] ** 2).mean().detach(),
            "kalman_innovation_gate_mean": stats["innovation_gate"].mean().detach(),
        }
