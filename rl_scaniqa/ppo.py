from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.optim as optim

# Minimal policy interface

@dataclass
class PPOConfig:
    clip_eps_start: float = 0.20
    clip_eps_end: float   = 0.10
    entropy_coef_start: float = 0.02
    entropy_coef_end: float   = 0.005
    value_coef: float = 0.5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    normalize_adv: bool = True
    lr: float = 3e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    max_grad_norm: float = 1.0
    update_epochs: int = 4
    minibatch_size: int = 4096
    target_kl: Optional[float] = None
    total_updates: int = 1000
    rank_mse_max_lambda: float = 2.0
    rank_mse_warmup_epochs: int = 30

class LinearScheduler:
    def __init__(self, start: float, end: float, total_steps: int) -> None:
        self.start = float(start); self.end = float(end); self.total = max(1, int(total_steps))
    def at(self, step: int) -> float:
        if self.total == 1:
            return self.start if step <= 0 else self.end
        p = min(max(step, 0), self.total - 1) / (self.total - 1)
        return self.start + (self.end - self.start) * p

def rank_mse_lambda(epoch_idx: int, warmup_epochs: int, max_lambda: float) -> float:
    if warmup_epochs <= 0:
        return max_lambda
    p = float(min(epoch_idx, warmup_epochs)) / float(warmup_epochs)
    return max_lambda * p

class PPO:
    """
    Proximal Policy Optimization for time-major rollouts.
    """
    def __init__(self, actor_critic: nn.Module, cfg: PPOConfig) -> None:
        self.ac = actor_critic
        self.cfg = cfg
        self.device = next(actor_critic.parameters()).device
        self.optimizer = optim.Adam(self.ac.parameters(), lr=cfg.lr, betas=cfg.betas)

        # set up linear schedulers for clip epsilon and entropy coef
        self._clip_sched = LinearScheduler(cfg.clip_eps_start, cfg.clip_eps_end, cfg.total_updates)
        self._ent_sched  = LinearScheduler(cfg.entropy_coef_start, cfg.entropy_coef_end, cfg.total_updates)

        # house-keeping
        self.update_idx = 0  # how many PPO updates we have performed

    @torch.no_grad()
    def compute_gae_inplace(self, buffer, next_value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        adv, ret = buffer.compute_gae(
            next_value=next_value,
            gamma=self.cfg.gamma,
            gae_lambda=self.cfg.gae_lambda,
            normalize_advantages=self.cfg.normalize_adv,
        )
        return adv, ret

    def update(self, buffer) -> Dict[str, float]:
        """
        One PPO update over the filled rollout buffer.
        """
        # current scheduled coefficients
        clip_eps = self._clip_sched.at(self.update_idx)
        ent_coef = self._ent_sched.at(self.update_idx)
        val_coef = self.cfg.value_coef

        # Flatten data once (no copies), then draw minibatches.
        data = buffer.get_flat()
        B = data["logprobs"].shape[0]

        # For numeric stability: detach & ensure contiguous
        old_logprob = data["logprobs"].detach()
        old_value   = data["values"].detach()
        adv         = data["advantages"].detach()
        ret         = data["returns"].detach()

        # Stats accumulators
        epoch_policy_loss = 0.0
        epoch_value_loss  = 0.0
        epoch_entropy     = 0.0
        epoch_kl          = 0.0
        epoch_clipfrac    = 0.0
        num_mb            = 0

        # multiple epochs over the same rollout
        for _ in range(self.cfg.update_epochs):
            for mb in buffer.iter_minibatches(
                batch_size=min(self.cfg.minibatch_size, B),
                shuffle=True,
                drop_last=False,
                yield_dict=False,
            ):
                obs_mb, act_mb, log_mb, adv_mb, ret_mb, val_mb = mb
                # Re-evaluate actions under current policy:
                logprob_new, entropy, value_new = self.ac.evaluate_actions(obs_mb, act_mb)

                # ρ = exp(logπ_new - logπ_old);  Lπ = E [min(ρ*A, clip(ρ,1-ε,1+ε)*A)]
                ratio = torch.exp(logprob_new - log_mb)
                unclipped = ratio * adv_mb
                clipped   = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_mb
                policy_loss = -torch.mean(torch.min(unclipped, clipped))  # negative for descent

                # Value loss: (V(s) - R_total)^2
                value_loss = torch.mean((value_new - ret_mb) ** 2)

                # Entropy bonus: -c_H * H(π)
                ent_loss = -torch.mean(entropy)

                loss = policy_loss + val_coef * value_loss + ent_coef * ent_loss

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

                # Diagnostics
                with torch.no_grad():
                    # approx KL = E[logπ_old - logπ_new]
                    approx_kl = torch.mean(log_mb - logprob_new).abs()
                    clipfrac  = torch.mean((torch.abs(ratio - 1.0) > clip_eps).float())

                epoch_policy_loss += float(policy_loss.item())
                epoch_value_loss  += float(value_loss.item())
                epoch_entropy     += float(entropy.mean().item())
                epoch_kl          += float(approx_kl.item())
                epoch_clipfrac    += float(clipfrac.item())
                num_mb += 1


                if (self.cfg.target_kl is not None) and (approx_kl > self.cfg.target_kl):
                    break

            if (self.cfg.target_kl is not None) and (epoch_kl / max(1, num_mb) > self.cfg.target_kl):
                # stop further epochs on this rollout
                break

        self.update_idx += 1

        # average logs
        denom = max(1, num_mb)
        return {
            "loss/policy": epoch_policy_loss / denom,
            "loss/value":  epoch_value_loss  / denom,
            "entropy":     epoch_entropy     / denom,
            "stats/kl":    epoch_kl          / denom,
            "stats/clipfrac": epoch_clipfrac / denom,
            "sched/clip_eps": clip_eps,
            "sched/entropy_coef": ent_coef,
        }



def inject_group_rewards_into_buffer(
    buffer,
    *,
    group_diversity: Optional[torch.Tensor],  # R_div per image (Eq. 6)
    group_mse: Optional[torch.Tensor],        # R_mse per image (Eq. 7)
    group_rank: Optional[torch.Tensor],       # R_rank per image (Eq. 8)
    epoch_idx: int,
    cfg: PPOConfig,
    average_step_rewards: bool = True,
) -> None:
    """
    RolloutBuffer must carry 'group_index' (env->image id) and know K.
    """
    lam = rank_mse_lambda(epoch_idx, cfg.rank_mse_warmup_epochs, cfg.rank_mse_max_lambda)
    buffer.apply_multilevel_group_rewards(
        group_diversity=group_diversity,
        group_mse=group_mse,
        group_rank=group_rank,
        lambda_mse=lam,
        lambda_rank=lam,
        average_step_rewards=average_step_rewards,
        distribute_set_rewards_evenly=True,
    )
