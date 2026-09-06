from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import torch
import torch.nn.functional as F


def _to_1d(x: torch.Tensor) -> torch.Tensor:
    if x is None:
        return x
    if x.dim() == 0:
        x = x.unsqueeze(0)
    if x.dim() > 1:
        x = x.view(-1)
    return x.float()


@dataclass
class LossConfig:

    use_mse: bool = True
    use_rank: bool = True
    use_consistency: bool = True
    use_triplet: bool = True
    use_cross_rank: bool = True

    # Weights (betas) for Eq. (13)
    beta_mse: float = 1.0
    beta_rank: float = 0.2
    beta_cons: float = 0.2
    beta_triplet: float = 0.3
    beta_cross: float = 0.2

    # Triplet margins from Eq. (11)
    margin1: float = 0.02
    margin2: float = 0.10
    margin3: float = 0.12

    # Reduction mode
    reduction: str = "mean"


    tie_strategy: str = "zero"


def mse_loss_pair(
    pred1: torch.Tensor, gt1: torch.Tensor,
    pred2: torch.Tensor, gt2: torch.Tensor,
    reduction: str = "mean"
) -> torch.Tensor:
    """
    L_mse = (pred1 - gt1)^2 + (pred2 - gt2)^2  (batch-wise)
    """
    pred1, gt1 = _to_1d(pred1), _to_1d(gt1)
    pred2, gt2 = _to_1d(pred2), _to_1d(gt2)

    l1 = (pred1 - gt1) ** 2
    l2 = (pred2 - gt2) ** 2
    loss = l1 + l2
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def pairwise_rank_loss(
    pred1: torch.Tensor, pred2: torch.Tensor,
    gt1: torch.Tensor, gt2: torch.Tensor,
    reduction: str = "mean",
    tie_strategy: str = "zero"
) -> torch.Tensor:
    """
    L_rank = log(1 + exp( - s * (pred1 - pred2) )),
    """
    pred1, pred2 = _to_1d(pred1), _to_1d(pred2)
    gt1, gt2 = _to_1d(gt1), _to_1d(gt2)

    s = torch.sign(gt1 - gt2)  # [B] in {-1, 0, +1}

    if tie_strategy == "skip":
        mask = (s != 0)
        if mask.any():
            val = F.softplus(-s[mask] * (pred1[mask] - pred2[mask]))
            return val.mean() if reduction == "mean" else (val.sum() if reduction == "sum" else val)
        # No valid pairs; return zero
        return pred1.new_zeros(())

    # tie_strategy == "zero": keep s=0 terms (equals log2)
    val = F.softplus(-s * (pred1 - pred2))
    if reduction == "mean":
        return val.mean()
    if reduction == "sum":
        return val.sum()
    return val


def consistency_loss(
    pred_clean: torch.Tensor, pred_weak: torch.Tensor,
    reduction: str = "mean"
) -> torch.Tensor:
    """
    L_cons = ||pred_clean - pred_weak||^2
    """
    pred_clean, pred_weak = _to_1d(pred_clean), _to_1d(pred_weak)
    val = (pred_clean - pred_weak) ** 2
    if reduction == "mean":
        return val.mean()
    if reduction == "sum":
        return val.sum()
    return val


def triplet_loss(
    pred_clean: torch.Tensor,
    pred_mild: torch.Tensor,
    pred_strong: torch.Tensor,
    m1: float = 0.02,  # mild vs clean margin
    m2: float = 0.10,  # strong vs mild margin
    m3: float = 0.12,  # strong vs clean margin
    reduction: str = "mean"
) -> torch.Tensor:
    """
    L_triplet = max(0, pred_mild - pred_clean + m1)
             +  max(0, pred_strong - pred_mild + m2)
             +  max(0, pred_strong - pred_clean + m3)
    """
    pc = _to_1d(pred_clean)
    pm = _to_1d(pred_mild)
    ps = _to_1d(pred_strong)

    t1 = torch.clamp(pm - pc + m1, min=0.0)
    t2 = torch.clamp(ps - pm + m2, min=0.0)
    t3 = torch.clamp(ps - pc + m3, min=0.0)
    val = t1 + t2 + t3

    if reduction == "mean":
        return val.mean()
    if reduction == "sum":
        return val.sum()
    return val


def cross_rank_loss(
    pred_A_aug: torch.Tensor,
    pred_B_aug: torch.Tensor,
    gt_A: torch.Tensor,
    gt_B: torch.Tensor,
    reduction: str = "mean",
    tie_strategy: str = "zero"
) -> torch.Tensor:
    """
    L_cross = log(1 + exp( - s * (pred_A_aug - pred_B_aug) )),
    """
    pA = _to_1d(pred_A_aug)
    pB = _to_1d(pred_B_aug)
    gA = _to_1d(gt_A)
    gB = _to_1d(gt_B)

    s = torch.sign(gA - gB)  # [B]

    if tie_strategy == "skip":
        mask = (s != 0)
        if mask.any():
            val = F.softplus(-s[mask] * (pA[mask] - pB[mask]))
            return val.mean() if reduction == "mean" else (val.sum() if reduction == "sum" else val)
        return pA.new_zeros(())

    val = F.softplus(-s * (pA - pB))
    if reduction == "mean":
        return val.mean()
    if reduction == "sum":
        return val.sum()
    return val


def compute_total_loss(
    preds: Dict[str, torch.Tensor],
    gts: Dict[str, torch.Tensor],
    cfg: Optional[LossConfig] = None
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

    if cfg is None:
        cfg = LossConfig()

    # Initialize all component losses to zero
    zero = torch.tensor(0.0, device=next(iter(preds.values())).device)
    Lmse = zero
    Lrank = zero
    Lcons = zero
    Ltrip = zero
    Lcross = zero

    # L_mse
    if cfg.use_mse:
        Lmse = mse_loss_pair(
            preds.get("pair_pred1"), gts.get("pair_gt1"),
            preds.get("pair_pred2"), gts.get("pair_gt2"),
            reduction=cfg.reduction
        )

    # L_rank
    if cfg.use_rank:
        Lrank = pairwise_rank_loss(
            preds.get("pair_pred1"), preds.get("pair_pred2"),
            gts.get("pair_gt1"), gts.get("pair_gt2"),
            reduction=cfg.reduction,
            tie_strategy=cfg.tie_strategy
        )

    # L_cons
    if cfg.use_consistency:
        Lcons = consistency_loss(
            preds.get("clean"), preds.get("weak"),
            reduction=cfg.reduction
        )

    # L_triplet
    if cfg.use_triplet:
        Ltrip = triplet_loss(
            preds.get("clean"), preds.get("mild"), preds.get("strong"),
            m1=cfg.margin1, m2=cfg.margin2, m3=cfg.margin3,
            reduction=cfg.reduction
        )

    # L_cross
    if cfg.use_cross_rank:
        Lcross = cross_rank_loss(
            preds.get("A_aug"), preds.get("B_aug"),
            gts.get("gt_A"), gts.get("gt_B"),
            reduction=cfg.reduction,
            tie_strategy=cfg.tie_strategy
        )

    # Weighted sum
    total = (
        cfg.beta_mse * Lmse +
        cfg.beta_rank * Lrank +
        cfg.beta_cons * Lcons +
        cfg.beta_triplet * Ltrip +
        cfg.beta_cross * Lcross
    )

    details = {
        "L_mse": Lmse.detach(),
        "L_rank": Lrank.detach(),
        "L_cons": Lcons.detach(),
        "L_triplet": Ltrip.detach(),
        "L_cross": Lcross.detach(),
        "L_total": total.detach(),
    }
    return total, details
