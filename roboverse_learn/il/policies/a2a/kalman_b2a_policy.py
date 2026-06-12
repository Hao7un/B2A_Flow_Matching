"""Kalman-style Belief-to-Action Flow Matching policy."""

from roboverse_learn.il.policies.a2a.b2a_policy import B2AImagePolicy
from roboverse_learn.il.policies.a2a.belief_source import KalmanGaussianBeliefSource


class KalmanB2AImagePolicy(B2AImagePolicy):
    """B2A policy with a structured Gaussian posterior belief source.

    This keeps B2A training/inference unchanged except for the source module:
    A2A history latents define a prior, visual-proprioceptive features define an
    innovation likelihood, and the flow starts from their closed-form diagonal
    Gaussian posterior.
    """

    def __init__(
        self,
        *args,
        latent_dim=512,
        belief_hidden_dim=512,
        belief_use_prev_latent=False,
        belief_min_log_std=-4.0,
        belief_max_log_std=0.0,
        kalman_prior_log_std_init=-1.5,
        kalman_visual_log_std_init=-1.5,
        kalman_innovation_gate_init=-2.0,
        **kwargs,
    ):
        super().__init__(
            *args,
            latent_dim=latent_dim,
            belief_hidden_dim=belief_hidden_dim,
            belief_use_prev_latent=belief_use_prev_latent,
            belief_min_log_std=belief_min_log_std,
            belief_max_log_std=belief_max_log_std,
            **kwargs,
        )
        self.belief_source = KalmanGaussianBeliefSource(
            latent_dim=latent_dim,
            hidden_dim=belief_hidden_dim,
            use_prev_latent=belief_use_prev_latent,
            min_log_std=belief_min_log_std,
            max_log_std=belief_max_log_std,
            prior_log_std_init=kalman_prior_log_std_init,
            visual_log_std_init=kalman_visual_log_std_init,
            innovation_gate_init=kalman_innovation_gate_init,
        )
