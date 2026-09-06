
from __future__ import annotations
from typing import Tuple, Optional
import torch


__all__ = [
    "generalized_advantage_estimation",
    "normalize_advantages_",
]


@torch.no_grad()
def generalized_advantage_estimation(
    rewards: torch.Tensor,     # step-wise rewards r_t for each parallel env
    values: torch.Tensor,      # critic predictions V(s_t)
    dones: torch.Tensor,       # bool terminal flags d_t (True if episode ends at step t)
    next_value: torch.Tensor,  # bootstrap value V(s_T) after the last collected step
    gamma: float,              # discount γ (=0.99)
    lam: float,                # GAE λ (=0.95)
    *,
    out_adv: Optional[torch.Tensor] = None,
    out_ret: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:

    if rewards.dim() != 2 or values.dim() != 2 or dones.dim() != 2:
        raise ValueError("rewards, values, dones must all be 2-D tensors of shape [T, N].")
    if rewards.shape != values.shape or rewards.shape != dones.shape:
        raise ValueError(f"Shape mismatch: rewards{tuple(rewards.shape)}, "
                         f"values{tuple(values.shape)}, dones{tuple(dones.shape)} must match.")
    T, N = rewards.shape

    next_value = torch.as_tensor(next_value, device=rewards.device, dtype=values.dtype).contiguous()
    if tuple(next_value.shape) != (N,):
        raise ValueError(f"next_value must be shape [N], got {tuple(next_value.shape)}")

    # Allocate outputs (keep time-major [T, N])
    advantages = out_adv if out_adv is not None else torch.empty_like(values)
    returns    = out_ret if out_ret is not None else torch.empty_like(values)

    # Internal running variables are env-major [N]
    last_gae = torch.zeros(N, dtype=values.dtype, device=values.device)
    next_val = next_value

    # Main backward recursion over time: t = T-1, ..., 0
    for t in reversed(range(T)):
        nonterminal = (~dones[t]).to(values.dtype)
        # δ_t = r_t + γ (1 - d_t) V_{t+1} - V_t
        delta = rewards[t] + gamma * nonterminal * next_val - values[t]
        # A_t = δ_t + γ λ (1 - d_t) A_{t+1}
        last_gae = delta + gamma * lam * nonterminal * last_gae
        advantages[t].copy_(last_gae)
        # Prepare V_{t} as V_{t+1} for next iteration
        next_val = values[t]

    # Returns: R_t = A_t + V_t
    returns.copy_((advantages + values))
    return advantages, returns


@torch.no_grad()
def normalize_advantages_(
    advantages: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:

    a = advantages.view(-1)
    a.sub_(a.mean()).div_(a.std(unbiased=False).clamp_min(eps))
    return advantages
