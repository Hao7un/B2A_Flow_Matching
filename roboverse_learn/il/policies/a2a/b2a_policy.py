"""
Belief-to-Action Flow Matching Policy (B2A).

A flow matching policy that replaces A2A's deterministic history-latent source
with a learned conditional belief source over future action latents.
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from roboverse_learn.il.policies.a2a.action_ae import CNNActionEncoder, SimpleActionDecoder
from roboverse_learn.il.policies.a2a.belief_source import DiagGaussianBeliefSource
from roboverse_learn.il.policies.base_image_policy import BaseImagePolicy
from roboverse_learn.il.utils.flow.flow_matchers import TorchFlowMatcher
from roboverse_learn.il.utils.models.flow_net import SimpleFlowNet
from roboverse_learn.il.utils.normalizer import LinearNormalizer
from roboverse_learn.il.utils.pytorch_util import dict_apply
from roboverse_learn.il.utils.vision.multi_image_obs_encoder import MultiImageObsEncoder


class B2AImagePolicy(BaseImagePolicy):
    """
    Belief-to-Action Flow Matching Policy.

    - Flow START: sampled belief source q_phi(z | history_latents, obs_latents)
    - Flow TARGET: future action latents
    - CONDITION: visual observation latents
    """

    def __init__(
        self,
        shape_meta: dict,
        obs_encoder: MultiImageObsEncoder,
        horizon,
        n_action_steps,
        n_obs_steps,
        flow_net,
        flow_matcher: TorchFlowMatcher,
        decode_flow_latents=True,
        consistency_weight=1.0,
        enc_contrastive_weight=1e-4,
        flow_contrastive_weight=0.0,
        latent_dim=512,
        action_ae=None,
        belief_hidden_dim=512,
        belief_use_prev_latent=False,
        belief_min_log_std=-4.0,
        belief_max_log_std=0.0,
        belief_entropy_floor=0.02,
        belief_nll_weight=0.02,
        belief_w2_weight=0.05,
        belief_entropy_weight=0.01,
        source_dropout_prob=0.1,
        source_noise_std=0.0,
        detach_source_for_flow=True,
        deterministic_eval=True,
        belief_history_dominance_weight=0.0,
        belief_sample_dominance_weight=0.0,
        flow_descent_weight=0.0,
        flow_no_worse_weight=0.0,
        flow_start_velocity_weight=0.0,
        source_dominance_margin=0.0,
        flow_no_worse_margin=0.0,
        flow_descent_cos_margin=0.05,
        prev_latent_dropout_prob=0.0,
        prev_latent_noise_std=0.0,
        **kwargs,
    ):
        super().__init__()

        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_feature_dim = obs_encoder.output_shape()[0]

        self.decode_flow_latents = decode_flow_latents
        self.consistency_weight = consistency_weight
        self.enc_contrastive_weight = enc_contrastive_weight
        self.flow_contrastive_weight = flow_contrastive_weight
        self.latent_dim = latent_dim
        self.num_sampling_steps = flow_matcher.num_sampling_steps

        self.flow_matcher = flow_matcher
        self.action_ae = action_ae

        self.obs_encoder = obs_encoder
        self.obs_projector = nn.Linear(obs_feature_dim * n_obs_steps, latent_dim)

        self.flow_net = SimpleFlowNet(
            input_dim=latent_dim,
            hidden_dim=flow_net.hidden_dim,
            output_dim=latent_dim,
            num_layers=flow_net.num_layers,
            mlp_ratio=flow_net.mlp_ratio,
            dropout=flow_net.dropout,
            condition_dim=latent_dim,
        )

        self.history_action_encoder = CNNActionEncoder(
            pred_horizon=n_obs_steps,
            action_dim=action_dim,
            latent_dim=latent_dim,
            hidden_dim=action_ae.net.enc_hidden_dim,
        )

        future_horizon = n_action_steps
        self.future_horizon = future_horizon

        self.action_encoder = CNNActionEncoder(
            pred_horizon=future_horizon,
            action_dim=action_dim,
            latent_dim=latent_dim,
            hidden_dim=action_ae.net.enc_hidden_dim,
        )
        self.action_decoder = SimpleActionDecoder(
            dec_hidden_dim=action_ae.net.dec_hidden_dim,
            latent_dim=latent_dim,
            pred_horizon=future_horizon,
            action_dim=action_dim,
            num_layers=action_ae.net.num_layers,
            dropout=action_ae.net.dropout,
        )

        self.belief_source = DiagGaussianBeliefSource(
            latent_dim=latent_dim,
            hidden_dim=belief_hidden_dim,
            use_prev_latent=belief_use_prev_latent,
            residual_mean=True,
            min_log_std=belief_min_log_std,
            max_log_std=belief_max_log_std,
        )

        self.belief_entropy_floor = belief_entropy_floor
        self.belief_nll_weight = belief_nll_weight
        self.belief_w2_weight = belief_w2_weight
        self.belief_entropy_weight = belief_entropy_weight
        self.source_dropout_prob = source_dropout_prob
        self.source_noise_std = source_noise_std
        self.detach_source_for_flow = detach_source_for_flow
        self.deterministic_eval = deterministic_eval
        self.belief_history_dominance_weight = belief_history_dominance_weight
        self.belief_sample_dominance_weight = belief_sample_dominance_weight
        self.flow_descent_weight = flow_descent_weight
        self.flow_no_worse_weight = flow_no_worse_weight
        self.flow_start_velocity_weight = flow_start_velocity_weight
        self.source_dominance_margin = source_dominance_margin
        self.flow_no_worse_margin = flow_no_worse_margin
        self.flow_descent_cos_margin = flow_descent_cos_margin
        self.prev_latent_dropout_prob = prev_latent_dropout_prob
        self.prev_latent_noise_std = prev_latent_noise_std
        self.prev_action_latents = None
        self.last_metrics = {}

        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.kwargs = kwargs

    def _encode_obs_latents(self, nobs, batch_size):
        this_nobs = dict_apply(
            nobs,
            lambda x: x[:, : self.n_obs_steps, ...].reshape(-1, *x.shape[2:]),
        )
        nobs_features = self.obs_encoder(this_nobs)
        nobs_features = nobs_features.reshape(batch_size, -1)
        return self.obs_projector(nobs_features)

    def _build_belief_start(
        self,
        history_latents,
        obs_latents,
        nactions=None,
        future_start=None,
        future_end=None,
        deterministic=False,
    ):
        prev_latents = None

        if self.belief_source.use_prev_latent:
            if (
                nactions is not None
                and future_start is not None
                and future_end is not None
                and future_start > 0
            ):
                prev_future_actions = nactions[:, future_start - 1 : future_end - 1, :]
                if prev_future_actions.shape[1] == self.future_horizon:
                    prev_latents = self.action_encoder(prev_future_actions).detach()
                    if self.training:
                        if self.prev_latent_dropout_prob > 0:
                            keep_mask = (
                                torch.rand(prev_latents.shape[0], 1, device=prev_latents.device)
                                >= self.prev_latent_dropout_prob
                            )
                            prev_latents = torch.where(keep_mask, prev_latents, torch.zeros_like(prev_latents))
                        if self.prev_latent_noise_std > 0:
                            prev_latents = prev_latents + self.prev_latent_noise_std * torch.randn_like(prev_latents)
            else:
                prev_latents = self.prev_action_latents
                if prev_latents is not None:
                    if prev_latents.shape[0] != history_latents.shape[0]:
                        prev_latents = None
                    else:
                        prev_latents = prev_latents.to(
                            device=history_latents.device,
                            dtype=history_latents.dtype,
                        )

        return self.belief_source(
            history_latents=history_latents,
            obs_latents=obs_latents,
            prev_latents=prev_latents,
            deterministic=deterministic,
        )

    @staticmethod
    def _metrics_to_float(metrics):
        result = {}
        for key, value in metrics.items():
            if torch.is_tensor(value):
                result[key] = value.detach().item()
            else:
                result[key] = value
        return result

    def compute_loss(self, batch):
        assert "valid_mask" not in batch
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        batch_size = nactions.shape[0]

        obs_latents = self._encode_obs_latents(nobs, batch_size)

        history_states = nobs["agent_pos"][:, : self.n_obs_steps, :]
        history_latents = self.history_action_encoder(history_states)

        future_start = self.n_obs_steps - 1
        future_end = future_start + self.n_action_steps
        future_actions = nactions[:, future_start:future_end, :]
        future_action_latents = self.action_encoder(future_actions)

        belief_start, belief_stats = self._build_belief_start(
            history_latents=history_latents,
            obs_latents=obs_latents,
            nactions=nactions,
            future_start=future_start,
            future_end=future_end,
            deterministic=False,
        )

        raw_belief_start = belief_start

        if self.training and self.source_dropout_prob > 0:
            drop_mask = (
                torch.rand(belief_start.shape[0], 1, device=belief_start.device)
                < self.source_dropout_prob
            )
            belief_start = torch.where(drop_mask, history_latents, belief_start)

        start_for_flow = belief_start.detach() if self.detach_source_for_flow else belief_start

        # Decoupled flow-source noise. Must stay identical to the noise applied in
        # predict_action: the velocity field is only valid on the source
        # distribution it was trained on. Scale is set wider than the belief's
        # learned std so the flow learns corrections at the magnitude of
        # eval-time belief errors, keeping the source-target coupling
        # non-degenerate even when the belief mean fits the target.
        if self.source_noise_std > 0:
            start_for_flow = start_for_flow + self.source_noise_std * torch.randn_like(start_for_flow)

        flow_loss, _ = self.flow_matcher.compute_loss(
            self.flow_net,
            target=future_action_latents,
            start=start_for_flow,
            global_cond=obs_latents,
        )

        loss = flow_loss
        metrics = {"flow_loss": flow_loss.detach()}

        belief_losses = self.belief_source.belief_losses(
            target_latents=future_action_latents,
            stats=belief_stats,
            entropy_floor=self.belief_entropy_floor,
        )
        loss = loss + self.belief_nll_weight * belief_losses["belief_nll"]
        loss = loss + self.belief_w2_weight * belief_losses["belief_w2_proxy"]
        loss = loss + self.belief_entropy_weight * belief_losses["belief_entropy_floor_loss"]
        metrics.update(belief_losses)

        target_detached = future_action_latents.detach()
        history_detached = history_latents.detach()
        history_target_dist = ((history_detached - target_detached) ** 2).mean(dim=-1)
        belief_mu_target_dist = ((belief_stats["mu"] - target_detached) ** 2).mean(dim=-1)
        belief_sample_target_dist = ((raw_belief_start - target_detached) ** 2).mean(dim=-1)

        belief_history_dominance_loss = F.relu(
            belief_mu_target_dist - history_target_dist + self.source_dominance_margin
        ).mean()
        belief_sample_dominance_loss = F.relu(
            belief_sample_target_dist - history_target_dist + self.source_dominance_margin
        ).mean()
        if self.belief_history_dominance_weight > 0:
            loss = loss + self.belief_history_dominance_weight * belief_history_dominance_loss
        if self.belief_sample_dominance_weight > 0:
            loss = loss + self.belief_sample_dominance_weight * belief_sample_dominance_loss
        metrics["belief_history_dominance_loss"] = belief_history_dominance_loss.detach()
        metrics["belief_sample_dominance_loss"] = belief_sample_dominance_loss.detach()
        metrics["belief_history_dominance_violation_rate"] = (
            belief_mu_target_dist > history_target_dist
        ).float().mean().detach()
        metrics["belief_sample_dominance_violation_rate"] = (
            belief_sample_target_dist > history_target_dist
        ).float().mean().detach()

        if self.flow_descent_weight > 0:
            tau = torch.rand(batch_size, 1, device=obs_latents.device)
            flow_start_detached = start_for_flow.detach()
            x_tau = (1.0 - tau) * flow_start_detached + tau * target_detached
            vt_tau = self.flow_net(x_tau, tau.squeeze(-1), global_cond=obs_latents)
            descent_cos = F.cosine_similarity(
                x_tau - target_detached, vt_tau, dim=-1, eps=1e-6
            )
            flow_descent_loss = F.relu(descent_cos + self.flow_descent_cos_margin).pow(2).mean()
            loss = loss + self.flow_descent_weight * flow_descent_loss
            metrics["flow_descent_loss"] = flow_descent_loss.detach()
            metrics["flow_descent_cosine"] = descent_cos.mean().detach()
            metrics["flow_descent_violation_rate"] = (
                descent_cos > -self.flow_descent_cos_margin
            ).float().mean().detach()

        if self.flow_start_velocity_weight > 0:
            flow_start_detached = start_for_flow.detach()
            t0 = torch.zeros(batch_size, device=obs_latents.device)
            v0 = self.flow_net(flow_start_detached, t0, global_cond=obs_latents)
            flow_start_velocity_loss = ((flow_start_detached + v0 - target_detached) ** 2).mean()
            loss = loss + self.flow_start_velocity_weight * flow_start_velocity_loss
            metrics["flow_start_velocity_loss"] = flow_start_velocity_loss.detach()

        with torch.no_grad():
            target = future_action_latents.detach()
            metrics["source_target_mse"] = ((start_for_flow - target) ** 2).mean()
            metrics["history_target_mse"] = ((history_latents - target) ** 2).mean()
            metrics["belief_mu_target_mse"] = ((belief_stats["mu"] - target) ** 2).mean()
            metrics["belief_vs_history_mse"] = (
                (belief_stats["mu"] - history_latents.detach()) ** 2
            ).mean()

        if self.enc_contrastive_weight > 0:
            image_features = obs_latents.view(batch_size, -1)
            action_features = future_action_latents.view(batch_size, -1)
            contrastive_loss = self._compute_contrastive_loss(image_features, action_features)
            loss += self.enc_contrastive_weight * contrastive_loss
            metrics["enc_contrastive_loss"] = contrastive_loss.detach()

        if self.decode_flow_latents:
            action_latents_pred = self.flow_matcher.sample(
                self.flow_net,
                shape=(batch_size, self.latent_dim),
                device=obs_latents.device,
                start=start_for_flow,
                num_steps=self.num_sampling_steps,
                global_cond=obs_latents,
            )

            start_target_dist = ((start_for_flow.detach() - target_detached) ** 2).mean(dim=-1)
            end_target_dist = ((action_latents_pred - target_detached) ** 2).mean(dim=-1)
            flow_no_worse_loss = F.relu(
                end_target_dist - start_target_dist + self.flow_no_worse_margin
            ).mean()
            if self.flow_no_worse_weight > 0:
                loss = loss + self.flow_no_worse_weight * flow_no_worse_loss
            metrics["flow_no_worse_loss"] = flow_no_worse_loss.detach()
            metrics["flow_no_worse_violation_rate"] = (
                end_target_dist > start_target_dist
            ).float().mean().detach()
            metrics["flow_improvement_train"] = (start_target_dist - end_target_dist).mean().detach()

            if self.consistency_weight > 0:
                consistency_loss = F.mse_loss(action_latents_pred, future_action_latents)
                loss += self.consistency_weight * consistency_loss
                metrics["consistency_loss"] = consistency_loss.detach()

            if self.flow_contrastive_weight > 0:
                image_features = obs_latents.view(batch_size, -1)
                action_features = action_latents_pred.view(batch_size, -1)
                contrastive_loss = self._compute_contrastive_loss(image_features, action_features)
                loss += self.flow_contrastive_weight * contrastive_loss
                metrics["flow_contrastive_loss"] = contrastive_loss.detach()

            if self.action_ae["flow_recon_weight"] > 0:
                actions_recon = self.action_decoder(action_latents_pred)
                action_recon_loss = F.l1_loss(actions_recon, future_actions)
                metrics["flow_action_recon_loss"] = action_recon_loss.detach()
                loss += self.action_ae["flow_recon_weight"] * action_recon_loss
        else:
            action_latents_pred = future_action_latents

        if self.action_ae["enc_recon_weight"] > 0:
            actions_recon = self.action_decoder(future_action_latents)
            action_recon_loss = F.l1_loss(actions_recon, future_actions)
            metrics["enc_action_recon_loss"] = action_recon_loss.detach()
            loss += self.action_ae["enc_recon_weight"] * action_recon_loss

        self.last_metrics = self._metrics_to_float(metrics)
        return loss

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        batch_size = value.shape[0]

        obs_latents = self._encode_obs_latents(nobs, batch_size)

        history_states = nobs["agent_pos"][:, : self.n_obs_steps, :]
        history_latents = self.history_action_encoder(history_states)

        belief_start, _ = self._build_belief_start(
            history_latents=history_latents,
            obs_latents=obs_latents,
            deterministic=self.deterministic_eval,
        )

        if self.source_noise_std > 0:
            belief_start = belief_start + self.source_noise_std * torch.randn_like(belief_start)

        action_latents_pred = self.flow_matcher.sample(
            self.flow_net,
            shape=(batch_size, self.latent_dim),
            device=obs_latents.device,
            num_steps=self.num_sampling_steps,
            start=belief_start,
            global_cond=obs_latents,
            return_traces=False,
        )

        if self.belief_source.use_prev_latent:
            self.prev_action_latents = action_latents_pred.detach()

        with torch.no_grad():
            action_pred = self.action_decoder(action_latents_pred)

        action_pred = self.normalizer["action"].unnormalize(action_pred)
        action = action_pred[:, : self.n_action_steps]

        return {"action": action, "action_pred": action_pred}

    @torch.no_grad()
    def eval_belief_diagnostics(self, batch):
        """Open-loop error decomposition against ground-truth actions.

        Mirrors predict_action's inference path (deterministic_eval +
        source_noise) but uses the batch's GT future actions to split the
        policy's action error into three additive parts:
            decoder floor      = decode(true target latent) vs GT action
            belief contribution= decode(belief mu) - decoder floor
            flow contribution  = decode(flow output) - decode(belief mu)
        Run on train vs held-out splits to read each component's
        generalization gap. All errors are means over batch x horizon x dim.
        """
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        batch_size = nactions.shape[0]

        obs_latents = self._encode_obs_latents(nobs, batch_size)
        history_states = nobs["agent_pos"][:, : self.n_obs_steps, :]
        history_latents = self.history_action_encoder(history_states)

        future_start = self.n_obs_steps - 1
        future_end = future_start + self.n_action_steps
        future_actions = nactions[:, future_start:future_end, :]
        target_latents = self.action_encoder(future_actions)

        # Belief mean (the point estimate) and the actual eval-time source,
        # both built without seeing the future, exactly as predict_action does.
        belief_mu, _ = self._build_belief_start(
            history_latents=history_latents,
            obs_latents=obs_latents,
            deterministic=True,
        )
        eval_source, belief_stats = self._build_belief_start(
            history_latents=history_latents,
            obs_latents=obs_latents,
            deterministic=self.deterministic_eval,
        )
        if self.source_noise_std > 0:
            eval_source = eval_source + self.source_noise_std * torch.randn_like(eval_source)

        flow_out = self.flow_matcher.sample(
            self.flow_net,
            shape=(batch_size, self.latent_dim),
            device=obs_latents.device,
            num_steps=self.num_sampling_steps,
            start=eval_source,
            global_cond=obs_latents,
            return_traces=False,
        )

        action_unnorm = self.normalizer["action"].unnormalize
        future_actions_raw = action_unnorm(future_actions)

        def action_l1(latent):
            dec = self.action_decoder(latent)
            l1_norm = (dec - future_actions).abs().mean()
            l1_raw = (action_unnorm(dec) - future_actions_raw).abs().mean()
            return l1_norm, l1_raw

        mu_l1n, mu_l1r = action_l1(belief_mu)
        tgt_l1n, tgt_l1r = action_l1(target_latents)
        flow_l1n, flow_l1r = action_l1(flow_out)

        def latent_mse(a):
            return ((a - target_latents) ** 2).mean()

        return {
            "n": float(batch_size),
            # latent-space squared distance to the true target latent
            "belief_mu_target_mse": latent_mse(belief_mu).item(),
            "source_target_mse": latent_mse(eval_source).item(),
            "flow_target_mse": latent_mse(flow_out).item(),
            # action-space L1, normalized units (comparable to training losses)
            "decode_target_l1": tgt_l1n.item(),
            "decode_mu_l1": mu_l1n.item(),
            "decode_flow_l1": flow_l1n.item(),
            # action-space L1, raw action units (real joint-space magnitude)
            "decode_target_l1_raw": tgt_l1r.item(),
            "decode_mu_l1_raw": mu_l1r.item(),
            "decode_flow_l1_raw": flow_l1r.item(),
            # belief internals at eval
            "belief_avg_std": belief_stats["std"].mean().item(),
            "belief_gate": belief_stats["gate"].mean().item(),
        }

    def reset_belief(self):
        self.prev_action_latents = None

    def reset(self):
        self.reset_belief()

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    @torch.no_grad()
    def get_latents_for_visualization(self, batch):
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        batch_size = nactions.shape[0]

        obs_latents = self._encode_obs_latents(nobs, batch_size)

        history_states = nobs["agent_pos"][:, : self.n_obs_steps, :]
        history_latents = self.history_action_encoder(history_states)

        future_start = self.n_obs_steps - 1
        future_end = future_start + self.n_action_steps
        future_actions = nactions[:, future_start:future_end, :]
        future_latents = self.action_encoder(future_actions)

        _, belief_stats = self._build_belief_start(
            history_latents=history_latents,
            obs_latents=obs_latents,
            nactions=nactions,
            future_start=future_start,
            future_end=future_end,
            deterministic=True,
        )

        return belief_stats["mu"], future_latents

    @torch.no_grad()
    def get_flow_trajectories(self, batch, num_steps=None, n_samples=5):
        if num_steps is None:
            num_steps = self.num_sampling_steps

        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        batch_size = nactions.shape[0]
        n_samples = min(n_samples, batch_size)

        nobs_sample = {k: v[:n_samples] for k, v in nobs.items()}
        obs_latents = self._encode_obs_latents(nobs_sample, n_samples)

        history_states = nobs["agent_pos"][:n_samples, : self.n_obs_steps, :]
        history_latents = self.history_action_encoder(history_states)

        future_start = self.n_obs_steps - 1
        future_end = future_start + self.n_action_steps
        future_actions = nactions[:n_samples, future_start:future_end, :]
        future_latents = self.action_encoder(future_actions)

        belief_start, _ = self._build_belief_start(
            history_latents=history_latents,
            obs_latents=obs_latents,
            nactions=nactions[:n_samples],
            future_start=future_start,
            future_end=future_end,
            deterministic=True,
        )

        _, (traj_history, _) = self.flow_matcher.sample(
            self.flow_net,
            shape=(n_samples, self.latent_dim),
            device=obs_latents.device,
            num_steps=num_steps,
            start=belief_start,
            global_cond=obs_latents,
            return_traces=True,
        )

        traj_history_cpu = []
        for traj in traj_history:
            if hasattr(traj, "cpu"):
                traj_history_cpu.append(traj.cpu())
            else:
                traj_history_cpu.append(torch.tensor(traj))

        traj_stacked = torch.stack(traj_history_cpu, dim=0)
        trajectories = [traj_stacked[:, i, :].numpy() for i in range(n_samples)]
        future_latents_np = future_latents.cpu().numpy()

        return trajectories, future_latents_np

    @staticmethod
    def _compute_contrastive_loss(image_features, action_features, temperature=0.07):
        """Contrastive loss between image and action features (InfoNCE)."""
        batch_size = image_features.size(0)
        image_features = F.normalize(image_features, dim=1)
        action_features = F.normalize(action_features, dim=1)

        logits = torch.matmul(image_features, action_features.T) / temperature

        labels = torch.arange(batch_size, device=logits.device)
        loss_i2a = F.cross_entropy(logits, labels)
        loss_a2i = F.cross_entropy(logits.T, labels)

        return (loss_i2a + loss_a2i) / 2
