from __future__ import annotations
from typing import Optional, Dict, List, Tuple, Sequence

import torch
import torch.nn.functional as F


__all__ = [
    "rgb_to_gray",
    "grayscale_entropy",
    "ssim_simple",
    # step-wise exploration (Eq. 5)
    "novelty_delta",
    "equator_bias",
    "stepwise_exploration_reward",
    # set-level diversity (Eq. 6)
    "compute_r_div_from_actions",
    # task-aligned rewards (Eq. 7, 8)
    "pairwise_mse_rank_rewards",
]

@torch.no_grad()
def rgb_to_gray(x_bchw: torch.Tensor) -> torch.Tensor:
    """
    Convert RGB to grayscale.
    """
    if x_bchw.ndim != 4:
        raise ValueError("rgb_to_gray: expect [N, C, H, W]")
    if x_bchw.size(1) == 1:
        return x_bchw
    w = torch.tensor([0.2989, 0.5870, 0.1140],
                     device=x_bchw.device, dtype=x_bchw.dtype).view(1, 3, 1, 1)
    g = (x_bchw * w).sum(dim=1, keepdim=True)
    return g.clamp_(0.0, 1.0)


@torch.no_grad()
def grayscale_entropy(x_b1hw: torch.Tensor, bins: int = 256) -> torch.Tensor:
    """
    Shannon entropy H(x) of grayscale viewports (per-sample).
    """
    if x_b1hw.ndim != 4 or x_b1hw.size(1) != 1:
        raise ValueError("grayscale_entropy expects [N,1,H,W]")
    N = x_b1hw.size(0)
    x = x_b1hw.clamp(0.0, 1.0)

    # Convert to integer levels in [0, bins-1]
    xb = (x * (bins - 1)).to(torch.long).view(N, -1)
    Hs: List[torch.Tensor] = []
    for i in range(N):
        counts = torch.bincount(xb[i], minlength=bins).to(x.dtype)
        p = counts / counts.sum().clamp_min(1.0)
        H_i = -(p * (p.add(1e-12)).log()).sum()
        Hs.append(H_i)
    return torch.stack(Hs, dim=0).to(x.dtype)


@torch.no_grad()
def ssim_simple(x_b1hw: torch.Tensor,
                y_b1hw: torch.Tensor,
                window_size: int = 3,
                C1: float = 0.01 ** 2,
                C2: float = 0.03 ** 2) -> torch.Tensor:
    """
    per-sample SSIM and averages over spatial dimensions.
    """
    if x_b1hw.shape != y_b1hw.shape or x_b1hw.ndim != 4 or x_b1hw.size(1) != 1:
        raise ValueError("ssim_simple expects x,y both [N,1,H,W]")

    N, _, H, W = x_b1hw.shape
    pad = window_size // 2

    # Reflection padding to keep same HxW after avg pooling
    x = F.pad(x_b1hw, (pad, pad, pad, pad), mode="reflect")
    y = F.pad(y_b1hw, (pad, pad, pad, pad), mode="reflect")

    mu_x = F.avg_pool2d(x, kernel_size=window_size, stride=1)
    mu_y = F.avg_pool2d(y, kernel_size=window_size, stride=1)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.avg_pool2d(x * x, window_size, 1) - mu_x2
    sigma_y2 = F.avg_pool2d(y * y, window_size, 1) - mu_y2
    sigma_xy = F.avg_pool2d(x * y, window_size, 1) - mu_xy

    # SSIM map
    num = (2.0 * mu_xy + C1) * (2.0 * sigma_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    ssim_map = num / den.clamp_min(1e-12)
    ssim_map = ssim_map.clamp(-1.0, 1.0)

    # Mean over spatial dims -> [N]
    return ssim_map.view(N, -1).mean(dim=1)


@torch.no_grad()
def novelty_delta(
    idx_t: torch.Tensor,            # indices of the selected viewport at time t
    visited_mask: torch.Tensor,     # bool mask; True means "visited"
) -> torch.Tensor:
    """
    Novelty indicator δ_new(x_t):
      1.0 if the selected viewport is not visited before in the *current episode*
      0.0 otherwise.
    """
    if idx_t.ndim != 1:
        raise ValueError("novelty_delta: idx_t must be [N]")
    if visited_mask.ndim != 2 or visited_mask.size(0) != idx_t.size(0):
        raise ValueError("novelty_delta: visited_mask must be [N, X]")

    N = idx_t.size(0)
    arangeN = torch.arange(N, device=idx_t.device)
    first_time = ~visited_mask[arangeN, idx_t]
    visited_mask[arangeN, idx_t] = True
    return first_time.to(dtype=torch.float32)


@torch.no_grad()
def equator_bias(
    idx_t: torch.Tensor,
    yaw_pitch_table: torch.Tensor,
    gamma_eq: float = 0.5,
) -> torch.Tensor:
    """
    Equator bias B(x_t) = exp(-gamma_eq * |pitch(x_t)|) in (0,1].
    """
    if yaw_pitch_table.ndim != 2 or yaw_pitch_table.size(1) != 2:
        raise ValueError("equator_bias: yaw_pitch_table must be [X,2] (yaw, pitch)")
    pitch = yaw_pitch_table[idx_t, 1]
    return torch.exp(-float(gamma_eq) * pitch.abs()).to(dtype=torch.float32)


@torch.no_grad()
def stepwise_exploration_reward(
    x_t: torch.Tensor,
    x_prev: Optional[torch.Tensor],
    idx_t: torch.Tensor,
    visited_mask: torch.Tensor,
    yaw_pitch_table: torch.Tensor,
    lambdas: Optional[Dict[str, float]] = None,
    gamma_eq: float = 0.5,
) -> torch.Tensor:
    """
    Eq. (5) on p.4:
      r_t = λ_ent·H(x_t) + λ_ssim·1[t>1]·(1 - SSIM(x_{t-1}, x_t))
            + λ_nov·δ_new(x_t) + λ_eqb·B(x_t)
    """
    if lambdas is None:
        lambdas = dict(ent=0.5, ssim=1.0, nov=1.0, eqb=0.2)

    # Convert to grayscale for entropy/SSIM
    xg_t   = rgb_to_gray(x_t)
    H_t    = grayscale_entropy(xg_t)

    if x_prev is None:
        ssim_term = torch.zeros(x_t.size(0), device=x_t.device, dtype=x_t.dtype)
    else:
        xg_prev  = rgb_to_gray(x_prev)
        ssim_val = ssim_simple(xg_prev, xg_t)
        ssim_term = 1.0 - ssim_val

    delta_new = novelty_delta(idx_t, visited_mask)
    B_t       = equator_bias(idx_t, yaw_pitch_table, gamma_eq=gamma_eq)

    r_t = (float(lambdas["ent"])  * H_t
         + float(lambdas["ssim"]) * ssim_term
         + float(lambdas["nov"])  * delta_new
         + float(lambdas["eqb"])  * B_t).to(dtype=torch.float32)
    return r_t


# Scanpath diversity reward R_div (Eq. 6)

@torch.no_grad()
def compute_r_div_from_actions(
    actions_tn: torch.Tensor,
    group_index: torch.Tensor,
    X: int,
    beta_cov: float = 0.2,
    beta_jac: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor]:

    if actions_tn.ndim != 2 or actions_tn.size(1) != group_index.size(0):
        raise ValueError("compute_r_div_from_actions: actions [T,N], group_index [N] mismatch")
    if X <= 0:
        raise ValueError("compute_r_div_from_actions: X must be positive")

    device = actions_tn.device
    uniq, inv = torch.unique(group_index, sorted=True, return_inverse=True)  # uniq: [G_unique]
    G = uniq.numel()
    T, N = actions_tn.shape

    rdiv_vals: List[torch.Tensor] = []

    for g_idx in range(G):
        env_idx = (inv == g_idx).nonzero(as_tuple=False).squeeze(1)
        K = env_idx.numel()
        if K <= 0:
            rdiv_vals.append(torch.tensor(0.0, device=device, dtype=torch.float32))
            continue

        sets: List[set] = []
        for e in env_idx.tolist():
            path_idx: torch.Tensor = actions_tn[:, e]
            sets.append(set(path_idx.tolist()))

        union_size = len(set.union(*sets)) if K > 0 else 0
        cov_term = float(union_size) / float(X)

        # Pairwise Jaccard mean
        if K >= 2:
            jac_list: List[float] = []
            for i in range(K):
                for j in range(i + 1, K):
                    A, B = sets[i], sets[j]
                    u = len(A | B)
                    inter = len(A & B)
                    jac = (inter / u) if u > 0 else 0.0
                    jac_list.append(jac)
            jac_mean = sum(jac_list) / len(jac_list)
        else:
            jac_mean = 0.0

        rdiv = float(beta_cov) * cov_term - float(beta_jac) * jac_mean
        rdiv_vals.append(torch.tensor(rdiv, device=device, dtype=torch.float32))

    R_div = torch.stack(rdiv_vals, dim=0)
    return R_div, uniq

# Task-aligned rewards (Eq. 7 and Eq. 8)

@torch.no_grad()
def pairwise_mse_rank_rewards(
    q1: torch.Tensor,  # predicted score for image 1
    q2: torch.Tensor,
    Q1: torch.Tensor,  # ground-truth MOS for image 1
    Q2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute task-aligned rewards per Eq. (7) and Eq. (8) on p.4–5:
      R_mse  = - ((q1 - Q1)^2 + (q2 - Q2)^2)
      R_rank = - log(1 + exp( - s * (q1 - q2) )),  where s = sign(Q1 - Q2)
    """
    if not (q1.shape == q2.shape == Q1.shape == Q2.shape and q1.ndim == 1):
        raise ValueError("pairwise_mse_rank_rewards: all inputs must be [B]")

    device = q1.device
    target1 = Q1.to(device=device, dtype=q1.dtype)
    target2 = Q2.to(device=device, dtype=q2.dtype)
    R_mse = -(((q1 - target1) ** 2) + ((q2 - target2) ** 2)).to(torch.float32)

    sign = torch.sign(target1 - target2)
    ranked = -F.softplus(-sign * (q1 - q2))
    R_rank = torch.where(sign == 0, torch.zeros_like(ranked), ranked).to(torch.float32)

    return R_mse, R_rank
