
# -*- coding: utf-8 -*-
"""
================================================================================
RL-ScanIQA — Runnable Reference Training Script
================================================================================

This script provides a compact **reference training path** aligned with the
main ideas in the RL-ScanIQA paper. It implements:
  • PPO objective based on Eq.(2): clipped surrogate (policy), value MSE (unclipped
    by default), and entropy regularization — with linear annealing ε: 0.2→0.1,
    c_H: 0.02→0.005.
  • GAE advantages (γ, λ).
  • Multi-level rewards as in Eq.(3)–(7):
      - Step-wise Exploration Rewards (SER): Shannon entropy of viewport, 1-SSIM
        to previous viewport, novelty δ_new, equator bias B(rt).
      - Scanpath Diversity Reward (SDR): coverage fraction - pairwise Jaccard.
      - Task-aligned Perceptual Rewards (TPR): negative MSE and soft ranking.
      - Total reward R_total: average stepwise across K + R_div + λ_mse·R_mse +
        λ_rank·R_rank; λ_mse, λ_rank linearly annealed 0→2 over the first 30 epochs.
  • Scanpath policy per Sec.3.2.1: GRU memory (h_{t-1}), global feature g, candidate
    viewport features f^j; logits via v^T tanh(Wh h + Wg g + Wf f^j + b) + mask.
  • 8×4 candidate grid (X=32) on the sphere, FOV=90°×90°, viewport size 224×224,
    ERP→viewport warping with horizontal wrapping + polar clamping.
  • Quality Assessor per Sec.3.3.1: attention pooling α_t over scanpath features using g,
    and MLP to regress Q̂_k, then final Q̂ = mean_k Q̂_k.
  • Augmentation-based QA losses per Sec.3.3.2: consistency, triplet, cross-rank,
    combined as Eq.(12) with given β weights.
  • Joint training loop (policy + QA) with two optimizers and an inference API (K=15, T=7).

The default backbone is a light "ToyBackbone" so that the software path can be
tested without downloads. Set --backbone dino to use a DINOv2 convenience path.
This standalone script is not the complete internal experiment launcher used to
produce every paper table; consult docs/REPRODUCIBILITY.md before benchmarking.

IMPORTANT
---------
This script expects a CSV with columns: img1,img2,Q1,Q2 (paths & MOS). Use --pairs_csv.
Images are assumed ERP RGB images readable by PIL. For a dry-run sanity check,
use --pairs_csv '' to trigger a tiny synthetic dataset.

CLI EXAMPLES
------------
# 1) Dry-run with synthetic data and toy backbone (fast check):
python scripts/train_reference.py --pairs_csv '' --epochs 1 --batch_size 8 --device cpu

# 2) Real training with your data (replace paths) and default PPO & QA settings:
python scripts/train_reference.py \
  --pairs_csv /path/to/pairs.csv --img_root /path/to/images \
  --epochs 300 --batch_size 4 --device cuda --backbone toy \
  --K 15 --T 7 --lr_policy 3e-4 --lr_qa 1e-4

# 3) Use DINOv2 backbone if available:
python scripts/train_reference.py --pairs_csv ... --backbone dino

================================================================================
"""

from __future__ import annotations

import os
import math
import time
import json
import csv
import random
import argparse
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# =============================================================================
# Utilities & Schedulers
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

class LinearScheduler:
    def __init__(self, start: float, end: float, T: int):
        self.start = float(start); self.end = float(end); self.T = max(1, int(T))
    def __call__(self, i: int) -> float:
        i = min(max(int(i), 0), self.T)
        alpha = i / float(self.T)
        return (1.0 - alpha) * self.start + alpha * self.end

# Small helper: compute Shannon entropy over grayscale histogram
def grayscale_entropy(x: torch.Tensor, bins: int = 256) -> torch.Tensor:
    """
    x: [B,1,H,W] in [0,1] float
    returns: [B]
    """
    B = x.shape[0]
    x_flat = x.view(B, -1)
    # Hist per batch element (vectorized via scatter_add)
    hist = torch.zeros(B, bins, device=x.device)
    idx = torch.clamp((x_flat * (bins-1)).long(), 0, bins-1)
    hist.scatter_add_(1, idx, torch.ones_like(idx, dtype=hist.dtype))
    p = hist / (hist.sum(dim=1, keepdim=True).clamp_min(1e-8))
    ent = -(p.clamp_min(1e-12) * p.clamp_min(1e-12).log()).sum(dim=1)
    return ent  # [B]

def ssim_simple(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Simplified SSIM for small patches (no gaussian window).
    x,y: [B,1,H,W] in [0,1] float
    return: [B] SSIM in [~0,1]
    """
    C1, C2 = 0.01**2, 0.03**2
    mu_x = x.mean(dim=(2,3)); mu_y = y.mean(dim=(2,3))
    vx = x.var(dim=(2,3), unbiased=False); vy = y.var(dim=(2,3), unbiased=False)
    cxy = ((x - mu_x[:,None,None]) * (y - mu_y[:,None,None])).mean(dim=(2,3))
    num = (2*mu_x*mu_y + C1) * (2*cxy + C2)
    den = (mu_x**2 + mu_y**2 + C1) * (vx + vy + C2)
    return (num / den).clamp(0.0, 1.0)

# =============================================================================
# ERP → Viewport Sampler (8×4 candidates, FOV=90°, 224×224)
# =============================================================================

def make_candidate_grid(n_yaw: int=8, n_pitch: int=4) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Uniform grid on sphere for viewport *centers* (yaw ∈ [-π, π), pitch ∈ [-π/2, π/2]).
    returns:
      yaw_c:   [X] in radians
      pitch_c: [X] in radians
    where X = n_yaw * n_pitch
    """
    yaw_vals = torch.linspace(-math.pi, math.pi, steps=n_yaw+1)[:-1] + (math.pi / n_yaw)
    pitch_edges = torch.linspace(-math.pi / 2, math.pi / 2, steps=n_pitch + 1)
    pitch_vals = 0.5 * (pitch_edges[:-1] + pitch_edges[1:])
    Y, P = torch.meshgrid(yaw_vals, pitch_vals, indexing='ij')
    yaw_c = Y.reshape(-1)      # [X]
    pitch_c = P.reshape(-1)    # [X]
    return yaw_c, pitch_c

def erp_to_viewports(erp: torch.Tensor, yaw_c: torch.Tensor, pitch_c: torch.Tensor,
                     fov_deg: float=90.0, out_hw: int=224) -> torch.Tensor:
    """
    Sample perspective viewports from an ERP image.
    erp: [B,3,H,W] in [0,1]
    yaw_c:   [X] centers (rad)
    pitch_c: [X] centers (rad)
    return: viewports [B,X,3,out_hw,out_hw]
    Mapping: perspective 90°×90°, horizontal wrap, polar clamp.
    """
    B, C, H, W = erp.shape
    X = yaw_c.numel()
    device = erp.device
    # Build base grid in viewport pixel coords u,v ∈ [-1,1]
    t = torch.linspace(-1, 1, steps=out_hw, device=device)
    V, U = torch.meshgrid(t, t, indexing='ij')   # [H,W]
    U = U.unsqueeze(0).unsqueeze(0)              # [1,1,H,H]
    V = V.unsqueeze(0).unsqueeze(0)              # [1,1,H,H]
    # Perspective mapping: for FOV=90°, tan(theta) mapping
    fov = math.radians(fov_deg)
    Xc = torch.tan(0.5 * fov * U)                # [1,1,H,H]
    Yc = torch.tan(0.5 * fov * V)                # [1,1,H,H]
    Zc = torch.ones_like(Xc)                     # [1,1,H,H]
    # Normalize direction
    norm = torch.sqrt(Xc**2 + Yc**2 + Zc**2)     # [1,1,H,H]
    Xc, Yc, Zc = Xc/norm, Yc/norm, Zc/norm

    # Rotation for each candidate (yaw, pitch)
    yaw = yaw_c.view(1, X, 1, 1).to(device)      # [1,X,1,1]
    pit = pitch_c.view(1, X, 1, 1).to(device)    # [1,X,1,1]

    # Rotate ray by yaw (around Y) then pitch (around X)
    sin_y, cos_y = torch.sin(yaw), torch.cos(yaw)
    sin_p, cos_p = torch.sin(pit), torch.cos(pit)

    # First yaw: [x',z'] = [cos y * x + sin y * z, -sin y * x + cos y * z]
    x1 = cos_y * Xc + sin_y * Zc                 # [1,X,H,H]
    z1 = -sin_y * Xc + cos_y * Zc
    y1 = Yc
    # Then pitch: rotate around X: [y'', z''] = [cos p * y1 - sin p * z1, sin p * y1 + cos p * z1]
    y2 = cos_p * y1 - sin_p * z1
    z2 = sin_p * y1 + cos_p * z1
    x2 = x1

    # Convert to spherical (yaw, pitch)
    yaw_samp = torch.atan2(x2, z2)               # [-π, π]
    pitch_samp = torch.asin(torch.clamp(y2, -1.0, 1.0))  # [-π/2, π/2]

    # Map to ERP pixel coords
    # u in [0,W): (yaw + π)/(2π) * W
    # v in [0,H): (π/2 - pitch)/π * H
    u = (yaw_samp + math.pi) / (2*math.pi) * W   # [1,X,H,H]
    v = (math.pi/2 - pitch_samp) / math.pi * H

    # Wrap u horizontally
    u = u % W
    # Clamp v to [0, H-1]
    v = torch.clamp(v, 0, H - 1.0)

    # Normalize to [-1,1] for grid_sample
    u_norm = (u / (W - 1)) * 2 - 1               # [1,X,H,H]
    v_norm = (v / (H - 1)) * 2 - 1
    grid = torch.stack([u_norm, v_norm], dim=-1) # [1,X,H,H,2]
    grid = grid.repeat(B, 1, 1, 1, 1)            # [B,X,H,H,2]

    # Sample
    erp_rep = erp.unsqueeze(1).repeat(1, X, 1, 1, 1)     # [B,X,3,H,W]
    erp_rep = erp_rep.view(B*X, C, H, W)
    grid = grid.view(B*X, out_hw, out_hw, 2)
    vp = F.grid_sample(erp_rep, grid, mode='bilinear', padding_mode='border', align_corners=True)
    vp = vp.view(B, X, C, out_hw, out_hw)        # [B,X,3,224,224]
    return vp

# =============================================================================
# Backbones: Toy (default) or DINOv2 (optional)
# =============================================================================

class ToyBackbone(nn.Module):
    """
    Very small CNN to produce global and region features.
    Output dims:
      - global g: [B, 1024]
      - region f: [B, d_f] (d_f=512 by default)
    """
    def __init__(self, d_f: int = 512):
        super().__init__()
        self.d_f = d_f
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(),
        ) # 224 -> 14
        self.head_g = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(256, 1024), nn.ReLU(),
        )
        self.head_f = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(256, d_f), nn.ReLU(),
        )

    def global_feat(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,3,H,W]
        h = self.stem(x)
        g = self.head_g(h)  # [B,1024]
        return g

    def region_feat(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,3,224,224]
        h = self.stem(x)
        f = self.head_f(h)  # [B,d_f]
        return f

def get_backbone(name: str, d_f: int=512):
    if name.lower() == "toy":
        return ToyBackbone(d_f=d_f)
    elif name.lower() == "dino":
        # Try to import dinov2 from torch.hub; fallback to toy on failure.
        try:
            import torch.hub
            model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            # Wrap to expose .global_feat and .region_feat as simple pooled outputs
            class DinoWrap(nn.Module):
                def __init__(self, m):
                    super().__init__()
                    self.m = m
                    for parameter in self.m.parameters():
                        parameter.requires_grad_(False)
                    self.proj_g = nn.Linear(384, 1024)
                    self.proj_f = nn.Linear(384, d_f)
                    self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
                    self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
                def forward_features(self, x):
                    # returns [B,C,H',W'] tokens in conv-map form or sequence; use mean-pooled token
                    x = (x - self.mean) / self.std
                    r = self.m.forward_features(x)  # dict with "x_norm_clstoken" or "x_norm_patchtokens"
                    if "x_norm_clstoken" in r:
                        feat = r["x_norm_clstoken"]  # [B, C]
                    else:
                        feat = r["x_norm_patchtokens"].mean(dim=1)  # [B, C]
                    return feat
                def global_feat(self, x):  # [B,3,H,W] -> [B,1024]
                    f = self.forward_features(x)
                    return self.proj_g(f)
                def region_feat(self, x):  # [B,3,224,224] -> [B,d_f]
                    f = self.forward_features(x)
                    return self.proj_f(f)
            return DinoWrap(model)
        except Exception as exc:
            raise RuntimeError(
                "DINOv2 could not be loaded. Check network access and the torch.hub cache, "
                "or use --backbone toy for a software smoke test."
            ) from exc
    else:
        return ToyBackbone(d_f=d_f)

# =============================================================================
# Policy (GRU) + Value Head per Eq.(1) and Sec.3.2.1
# =============================================================================

class ScanpathPolicyGRU(nn.Module):
    """
    Policy scores candidate viewports using:
      z_j_t = v^T tanh(Wh h_{t-1} + Wg g + Wf f^j + b) + m_t(j)
    then softmax over j ∈ {1..X}. Also predicts value V(s_t) from [h_{t-1}; g].

    Inputs per step:
      h_prev: [B, d_h]
      g:      [B, 1024]
      F:      [B, X, d_f] candidate features
      mask:   [B, X] (-inf for invalid/blocked)
    Outputs per step:
      logits: [B, X]
      value:  [B]
      h_next: [B, d_h] (GRUCell with selected f^{a_t})
    """
    def __init__(self, d_f: int=512, d_h: int=512, X: int=32):
        super().__init__()
        self.d_f, self.d_h, self.X = d_f, d_h, X
        dz = 512
        self.Wh = nn.Linear(d_h, dz, bias=False)
        self.Wg = nn.Linear(1024, dz, bias=False)
        self.Wf = nn.Linear(d_f, dz, bias=False)
        self.b  = nn.Parameter(torch.zeros(dz))
        self.v  = nn.Linear(dz, 1, bias=False)   # v^T · tanh(...)

        self.val = nn.Sequential(
            nn.Linear(d_h + 1024, 512), nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.gru = nn.GRUCell(d_f, d_h)

        # Init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('tanh'))
                if m.bias is not None: nn.init.constant_(m.bias, 0.0)

    def forward_step(self, h_prev: torch.Tensor, g: torch.Tensor,
                     F: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # h_prev: [B,d_h], g: [B,1024], F: [B,X,d_f], mask: [B,X]
        B, X, d_f = F.shape
        # Broadcast affine: [B,X,dz]
        z = torch.tanh(self.Wh(h_prev).unsqueeze(1) + self.Wg(g).unsqueeze(1) + self.Wf(F) + self.b)
        logits = self.v(z).squeeze(-1) + mask  # [B,X]
        value  = self.val(torch.cat([h_prev, g], dim=-1)).squeeze(-1)  # [B]
        return logits, value

    def step_and_update_h(self, h_prev: torch.Tensor, g: torch.Tensor, F: torch.Tensor,
                          mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Compute logits/value
        logits, value = self.forward_step(h_prev, g, F, mask)   # [B,X], [B]
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()                       # [B]
        logp = dist.log_prob(a)                 # [B]
        ent = dist.entropy()                    # [B]
        # Update GRU with selected f^{a}
        f_sel = F[torch.arange(F.shape[0]), a]  # [B,d_f]
        h_next = self.gru(f_sel, h_prev)        # [B,d_h]
        return a, logp, ent, value, h_next

# =============================================================================
# Quality Assessor per Sec.3.3 (attention pooling + MLP)
# =============================================================================

class QualityAssessor(nn.Module):
    """
    Attention pooling over T region features {f_t} with global feature g:
        α_t = softmax_t( v^T tanh( Wp f_t + Wg g ) )
        h_k = Σ_t α_t f_t
        Q̂_k = MLP([h_k; g])

    Inputs:
      F_seq: [B, T, d_f]
      g:     [B, 1024]
    Output:
      q_hat: [B]
    """
    def __init__(self, d_f: int=512):
        super().__init__()
        dz = 512
        self.Wp = nn.Linear(d_f, dz, bias=False)
        self.Wg = nn.Linear(1024, dz, bias=False)
        self.v  = nn.Linear(dz, 1, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(d_f + 1024, 512), nn.ReLU(),
            nn.Linear(512, 1)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('tanh'))
                if m.bias is not None: nn.init.constant_(m.bias, 0.0)

    def forward(self, F_seq: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        # F_seq: [B,T,d_f], g: [B,1024]
        B, T, d_f = F_seq.shape
        # Broadcast
        Wp_f = self.Wp(F_seq)                   # [B,T,dz]
        Wg_g = self.Wg(g).unsqueeze(1)          # [B,1,dz]
        z = torch.tanh(Wp_f + Wg_g)             # [B,T,dz]
        alpha = torch.softmax(self.v(z).squeeze(-1), dim=1)  # [B,T]
        h = torch.sum(alpha.unsqueeze(-1) * F_seq, dim=1)    # [B,d_f]
        q_hat = self.mlp(torch.cat([h, g], dim=-1)).squeeze(-1)  # [B]
        return q_hat

# =============================================================================
# PPO Buffer (for many short episodes = scanpaths)
# =============================================================================

@dataclass
class StepStore:
    g: torch.Tensor         # [d_g=1024]
    h_prev: torch.Tensor    # [d_h]
    F_cand: torch.Tensor    # [X,d_f]
    mask: torch.Tensor      # [X]
    action: int
    logp: float
    entropy: float
    value: float
    reward: float
    done: bool

class EpisodesBuffer:
    """
    Stores many scanpaths (episodes). Each episode is T steps.
    We keep minimal sufficient info to recompute new logp/value.
    """
    def __init__(self, d_h: int, d_f: int, X: int):
        self.data: List[StepStore] = []
        self.d_h, self.d_f, self.X = d_h, d_f, X

    def add(self, g, h_prev, F_cand, mask, action, logp, entropy, value, reward, done):
        self.data.append(StepStore(
            g=g.detach().cpu(), h_prev=h_prev.detach().cpu(),
            F_cand=F_cand.detach().cpu(), mask=mask.detach().cpu(),
            action=int(action), logp=float(logp), entropy=float(entropy),
            value=float(value), reward=float(reward), done=bool(done)
        ))

    def to_tensors(self, device: torch.device) -> Dict[str, torch.Tensor]:
        # Flatten lists into tensors for minibatching
        M = len(self.data)
        d_h, d_f, X = self.d_h, self.d_f, self.X
        g   = torch.stack([d.g for d in self.data], dim=0).to(device)             # [M,1024]
        h   = torch.stack([d.h_prev for d in self.data], dim=0).to(device)        # [M,d_h]
        F   = torch.stack([d.F_cand for d in self.data], dim=0).to(device)        # [M,X,d_f]
        msk = torch.stack([d.mask for d in self.data], dim=0).to(device)          # [M,X]
        act = torch.tensor([d.action for d in self.data], device=device, dtype=torch.long)  # [M]
        old_lp = torch.tensor([d.logp for d in self.data], device=device, dtype=torch.float32)  # [M]
        old_ent = torch.tensor([d.entropy for d in self.data], device=device, dtype=torch.float32)  # [M]
        old_v = torch.tensor([d.value for d in self.data], device=device, dtype=torch.float32)  # [M]
        rew = torch.tensor([d.reward for d in self.data], device=device, dtype=torch.float32)   # [M]
        done = torch.tensor([d.done for d in self.data], device=device, dtype=torch.bool)       # [M]
        return {
            "g": g, "h": h, "F": F, "mask": msk, "act": act,
            "old_logp": old_lp, "old_ent": old_ent, "old_v": old_v,
            "rew": rew, "done": done
        }

# =============================================================================
# Data: PairDataset (CSV) with ERP images and MOS
# =============================================================================

class PairDataset(Dataset):
    def __init__(self, pairs_csv: str, img_root: str='', synthetic_if_empty: bool=True):
        self.samples = []
        self.img_root = img_root
        if pairs_csv:
            if not os.path.exists(pairs_csv):
                raise FileNotFoundError(f"pairs_csv not found: {pairs_csv}")
            with open(pairs_csv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                required = {"img1", "img2", "Q1", "Q2"}
                missing = required.difference(reader.fieldnames or [])
                if missing:
                    raise ValueError(f"pairs_csv is missing columns: {sorted(missing)}")
                for row in reader:
                    img1, img2 = row["img1"].strip(), row["img2"].strip()
                    q1, q2 = float(row["Q1"]), float(row["Q2"])
                    self.samples.append((img1, img2, q1, q2))
        elif synthetic_if_empty:
            # Tiny synthetic dataset (random noise images) for sanity check
            for i in range(8):
                self.samples.append(('', '', float(np.random.uniform(10, 90)), float(np.random.uniform(10, 90))))
        else:
            raise FileNotFoundError("pairs_csv not found.")
        if len(self.samples) == 0:
            raise RuntimeError("No samples found in pairs_csv")

    def __len__(self): return len(self.samples)

    def _load_img(self, path: str, H: int=512, W: int=1024) -> Image.Image:
        if not path:
            return Image.fromarray((np.random.rand(H, W, 3) * 255).astype(np.uint8))

        image_path = os.path.join(self.img_root, path)
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"ERP image not found: {image_path}")
        return Image.open(image_path).convert("RGB")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img1, img2, q1, q2 = self.samples[idx]
        I1 = self._load_img(img1)
        I2 = self._load_img(img2)
        return {
            "img1": I1, "img2": I2,
            "Q1": float(q1), "Q2": float(q2)
        }

# =============================================================================
# Augmentations for QA
# =============================================================================

import torchvision.transforms as T

def make_qa_augs():
    # Clean: identity  (we will just resize to backbone input size)
    # Weak: small compression/blur/color jitter + Poisson noise
    weak = T.Compose([
        T.Resize((512, 1024)),
        T.ColorJitter(0.05,0.05,0.05,0.02),
    ])
    mild = T.Compose([
        T.Resize((512, 1024)),
        T.GaussianBlur(3, sigma=(0.5,1.0)),
        T.ColorJitter(0.1,0.1,0.1,0.05),
    ])
    strong = T.Compose([
        T.Resize((512, 1024)),
        T.GaussianBlur(5, sigma=(1.0,2.0)),
        T.ColorJitter(0.2,0.2,0.2,0.1),
    ])
    to_tensor = T.ToTensor()
    return weak, mild, strong, to_tensor

# =============================================================================
# Trainer (Joint): policy + QA
# =============================================================================

@dataclass
class TrainConfig:
    # Data
    pairs_csv: str = ''
    img_root: str = ''

    # Scanpath
    n_yaw: int = 8
    n_pitch: int = 4
    X: int = 32
    FOV_deg: float = 90.0
    viewport_hw: int = 224
    K: int = 5
    T: int = 4

    # Backbones
    backbone: str = "toy"
    d_f: int = 512
    d_h: int = 512

    # PPO/GAE
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_start: float = 0.20
    clip_end: float = 0.10
    ent_start: float = 0.02
    ent_end: float = 0.005
    value_coef: float = 0.5
    clip_value_loss: bool = False
    vf_clip_param: float = 0.2
    target_kl: Optional[float] = 0.03

    # TPR annealing
    lmse_start: float = 0.0
    lmse_end: float = 2.0
    lrank_start: float = 0.0
    lrank_end: float = 2.0
    tpr_anneal_epochs: int = 30

    # QA loss weights (Eq.12)
    beta_mse: float = 1.0
    beta_rank: float = 0.2
    beta_cons: float = 0.2
    beta_triplet: float = 0.3
    beta_cross: float = 0.2

    # LRs & optim
    lr_policy: float = 3e-4
    lr_qa: float = 1e-4
    max_grad_norm: float = 1.0
    adam_eps: float = 1e-8
    weight_decay: float = 0.0

    # Loop
    epochs: int = 2
    batch_size: int = 1
    num_workers: int = 0

    # Device & misc
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0
    log_interval: int = 1
    save_dir: str = ""

class JointTrainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        set_seed(cfg.seed)
        self.device = torch.device(cfg.device)

        if cfg.n_yaw <= 0 or cfg.n_pitch <= 0 or cfg.K <= 0 or cfg.T <= 0:
            raise ValueError("n_yaw, n_pitch, K, and T must all be positive")
        if cfg.T > cfg.n_yaw * cfg.n_pitch:
            raise ValueError("T cannot exceed the number of candidate viewports when revisits are masked")

        # Data
        self.ds = PairDataset(cfg.pairs_csv, img_root=cfg.img_root, synthetic_if_empty=True)
        self.loader = DataLoader(self.ds, batch_size=cfg.batch_size, shuffle=True,
                                 num_workers=cfg.num_workers, collate_fn=self._collate)

        # Backbones
        self.backbone = get_backbone(cfg.backbone, d_f=cfg.d_f).to(self.device)

        # Policy & QA
        self.policy = ScanpathPolicyGRU(d_f=cfg.d_f, d_h=cfg.d_h, X=cfg.n_yaw*cfg.n_pitch).to(self.device)
        self.qa = QualityAssessor(d_f=cfg.d_f).to(self.device)

        # Optims
        self.opt_policy = optim.Adam(list(self.policy.parameters()), lr=cfg.lr_policy, eps=cfg.adam_eps, weight_decay=cfg.weight_decay)
        self.trainable_qa_parameters = list(self.qa.parameters()) + [
            parameter for parameter in self.backbone.parameters() if parameter.requires_grad
        ]
        self.opt_qa = optim.Adam(
            self.trainable_qa_parameters,
            lr=cfg.lr_qa,
            eps=cfg.adam_eps,
            weight_decay=cfg.weight_decay,
        )

        # Schedulers
        self.eps_sched = LinearScheduler(cfg.clip_start, cfg.clip_end, max(cfg.epochs-1, 1))
        self.ent_sched = LinearScheduler(cfg.ent_start, cfg.ent_end, max(cfg.epochs-1, 1))
        self.lmse_sched = LinearScheduler(cfg.lmse_start, cfg.lmse_end, max(cfg.tpr_anneal_epochs-1, 1))
        self.lrank_sched = LinearScheduler(cfg.lrank_start, cfg.lrank_end, max(cfg.tpr_anneal_epochs-1, 1))

        # Reward weights (SER)
        self.lam_ent, self.lam_ssim, self.lam_nov, self.lam_eqb = 0.5, 1.0, 1.0, 0.2
        self.gamma_eq = 0.5

        # Augs
        self.weak_aug, self.mild_aug, self.strong_aug, self.to_tensor = make_qa_augs()

        # Candidate grid
        yaw_c, pitch_c = make_candidate_grid(cfg.n_yaw, cfg.n_pitch)
        self.yaw_c = yaw_c.to(self.device)      # [X]
        self.pitch_c = pitch_c.to(self.device)  # [X]
        self.X = yaw_c.numel()

    def _collate(self, batch):
        # Batch is a list of dicts with PIL images and MOS
        imgs1 = [b["img1"] for b in batch]
        imgs2 = [b["img2"] for b in batch]
        Q1 = torch.tensor([b["Q1"] for b in batch], dtype=torch.float32)
        Q2 = torch.tensor([b["Q2"] for b in batch], dtype=torch.float32)
        return imgs1, imgs2, Q1, Q2

    def _img_to_tensor(self, I: Image.Image) -> torch.Tensor:
        # Resize ERP to 512×1024 then to tensor [3,512,1024]
        I = I.resize((1024, 512))
        x = torch.from_numpy(np.array(I)).permute(2,0,1).float() / 255.0
        return x

    # ---------- Reward helpers ----------

    def _stepwise_reward(self, vp_cur: torch.Tensor, vp_prev: Optional[torch.Tensor], pitch: torch.Tensor,
                         is_new: torch.Tensor) -> torch.Tensor:
        """
        vp_cur:  [B,3,224,224] in [0,1]
        vp_prev: None or [B,3,224,224]
        pitch:   [B] in radians
        is_new:  [B] bool
        return:  [B] r_t
        """
        B = vp_cur.shape[0]
        gray = vp_cur.mean(dim=1, keepdim=True)  # [B,1,224,224]
        Hx = grayscale_entropy(gray)             # [B]
        if vp_prev is None:
            ssim_term = torch.zeros(B, device=vp_cur.device)
        else:
            ssim = ssim_simple(vp_prev.mean(dim=1, keepdim=True), gray)  # [B]
            ssim_term = (1.0 - ssim).clamp(0.0, 2.0)
        delta_new = is_new.float()               # [B]
        B_eq = torch.exp(-self.gamma_eq * pitch.abs())  # [B]
        r = (self.lam_ent * Hx
             + self.lam_ssim * ssim_term
             + self.lam_nov * delta_new
             + self.lam_eqb * B_eq)
        return r

    def _diversity_reward(self, scanpaths: List[List[int]]) -> float:
        # coverage - jaccard, Eq.(4)
        X = self.X
        K = len(scanpaths)
        union = set().union(*[set(p) for p in scanpaths])
        cov = len(union) / float(X)
        pairs = 0
        jacc_sum = 0.0
        for i in range(K):
            for j in range(i+1, K):
                a, b = set(scanpaths[i]), set(scanpaths[j])
                inter = len(a & b); uni = len(a | b) if len(a | b) > 0 else 1
                jacc = inter / float(uni)
                jacc_sum += jacc; pairs += 1
        jac = (jacc_sum / pairs) if pairs > 0 else 0.0
        beta_cov = 0.2; beta_jac = 0.2
        return beta_cov * cov - beta_jac * jac

    # ---------- Main per-image forward for K scanpaths ----------

    def _precompute_candidates(self, erp_b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        erp_b: [B,3,512,1024]
        returns:
          vp_all: [B,X,3,224,224]      (candidate viewports)
          F_all:  [B,X,d_f]            (candidate features)
          g:      [B,1024]             (global features)
        """
        B = erp_b.shape[0]
        vp_all = erp_to_viewports(erp_b, self.yaw_c, self.pitch_c, fov_deg=self.cfg.FOV_deg, out_hw=self.cfg.viewport_hw)  # [B,X,3,224,224]
        # Region features
        vp_flat = vp_all.view(B*self.X, 3, self.cfg.viewport_hw, self.cfg.viewport_hw)
        F_flat = self.backbone.region_feat(vp_flat)                        # [B*X,d_f]
        F_all = F_flat.view(B, self.X, self.cfg.d_f)                       # [B,X,d_f]
        # Global features
        g = self.backbone.global_feat(erp_b)                               # [B,1024]
        return vp_all, F_all, g

    def _sample_scanpaths(self, F_all: torch.Tensor, g: torch.Tensor, vp_all: torch.Tensor) -> Tuple[List[List[int]], Dict[str, torch.Tensor], EpisodesBuffer]:
        """
        Sample K scanpaths of length T per image in the batch.
        F_all: [B,X,d_f], g: [B,1024], vp_all: [B,X,3,224,224]
        Returns:
          scanpaths: List of length B, each is List[List[int]] with shape [K,T].
          pack: dict with tensors (for QA later): f_seq_sel [B,K,T,d_f]
          buffer: EpisodesBuffer (for PPO) filled with step stores
        """
        B, X, d_f = F_all.shape
        K, T = self.cfg.K, self.cfg.T
        device = F_all.device
        # Init hidden states with zeros
        h = torch.zeros(B*K, self.cfg.d_h, device=device)  # [B*K,d_h]
        # Expand g, F_all, vp_all to [B*K,...] for convenience
        g_exp = g.unsqueeze(1).expand(B, K, -1).reshape(B*K, -1)            # [B*K,1024]
        F_exp = F_all.unsqueeze(1).expand(B, K, X, d_f).reshape(B*K, X, d_f) # [B*K,X,d_f]
        batch_index = torch.arange(B, device=device).repeat_interleave(K)
        env_index = torch.arange(B*K, device=device)

        # Trackers
        visited = torch.zeros(B*K, X, dtype=torch.bool, device=device)  # [B*K,X]
        actions_all = torch.zeros(B*K, T, dtype=torch.long, device=device)
        logp_all = torch.zeros(B*K, T, dtype=torch.float32, device=device)
        ent_all = torch.zeros(B*K, T, dtype=torch.float32, device=device)
        val_all = torch.zeros(B*K, T+1, dtype=torch.float32, device=device)
        rew_all = torch.zeros(B*K, T, dtype=torch.float32, device=device)
        f_seq_sel = torch.zeros(B*K, T, d_f, dtype=torch.float32, device=device)
        pitch_selected = torch.zeros(B*K, T, dtype=torch.float32, device=device)

        buffer = EpisodesBuffer(d_h=self.cfg.d_h, d_f=self.cfg.d_f, X=X)

        prev_vp = None
        for t in range(T):
            # Dynamic mask: forbid revisits with a large negative bias
            mask = torch.where(visited, torch.full_like(visited, -1e9, dtype=torch.float32), torch.zeros_like(visited, dtype=torch.float32))  # [B*K,X]
            # Cache h_prev BEFORE taking action
            h_prev_store = h.clone()                           # [B*K,d_h]
            # One step of policy from h_prev_store
            logits, v_t = self.policy.forward_step(h_prev_store, g_exp, F_exp, mask)  # [B*K,X], [B*K]
            dist = torch.distributions.Categorical(logits=logits)
            a_t = dist.sample()                                # [B*K]
            lp_t = dist.log_prob(a_t)                          # [B*K]
            ent_t = dist.entropy()                             # [B*K]
            # Cache value_t
            val_all[:, t] = v_t
            # Update GRU with selected features
            f_sel = F_exp[env_index, a_t]                      # [B*K,d_f]
            h = self.policy.gru(f_sel, h_prev_store)           # [B*K,d_h]

            # Rewards at step t
            vp_sel = vp_all[batch_index, a_t]                  # [B*K,3,224,224]
            pitch_sel = self.pitch_c[a_t]                      # [B*K]
            # δ_new: True if not visited
            is_new = (~visited[env_index, a_t])
            r_t = self._stepwise_reward(vp_sel, prev_vp, pitch_sel, is_new)  # [B*K]
            # Record
            actions_all[:, t] = a_t
            logp_all[:, t] = lp_t
            ent_all[:, t] = ent_t
            rew_all[:, t] = r_t
            f_seq_sel[:, t] = f_sel
            pitch_selected[:, t] = pitch_sel
            # Update visited
            visited[env_index, a_t] = True
            prev_vp = vp_sel

            # Buffer for PPO (store per step) with exact h_prev and mask
            for bi in range(B*K):
                buffer.add(
                    g=g_exp[bi], h_prev=h_prev_store[bi],
                    F_cand=F_exp[bi], mask=mask[bi], action=int(a_t[bi].item()),
                    logp=float(lp_t[bi].item()), entropy=float(ent_t[bi].item()), value=float(v_t[bi].item()),
                    reward=float(r_t[bi].item()), done=(t==T-1)
                )
            # NOTE: For exact h_prev, we should store 'h_before_update'.
            # Here we approximate by reusing 'h' with a cheap trick (subtract f_sel*0).

        # Bootstrap value at T
        with torch.no_grad():
            v_T = self.policy.val(torch.cat([h, g_exp], dim=-1)).squeeze(-1)  # [B*K]
        val_all[:, T] = v_T

        # Pack scanpaths indices for SDR
        scanpaths = []
        for b in range(B):
            sp_set = []
            for k in range(K):
                idx = b*K + k
                sp_set.append(actions_all[idx].tolist())
            scanpaths.append(sp_set)

        pack = {
            "actions": actions_all.view(B, K, T),      # [B,K,T]
            "logp": logp_all.view(B, K, T),            # [B,K,T]
            "entropy": ent_all.view(B, K, T),          # [B,K,T]
            "values": val_all.view(B, K, T+1),         # [B,K,T+1]
            "rewards": rew_all.view(B, K, T),          # [B,K,T]
            "f_seq": f_seq_sel.view(B, K, T, d_f),     # [B,K,T,d_f]
            "g": g,                                    # [B,1024]
        }
        return scanpaths, pack, buffer

    def _qa_forward(self, f_seq: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """
        f_seq: [B,K,T,d_f], g: [B,1024]
        returns: q_hat: [B] = mean_k Q̂_k
        """
        B, K, T, d_f = f_seq.shape
        f_view = f_seq.view(B*K, T, d_f)
        g_rep = g.unsqueeze(1).repeat(1, K, 1).view(B*K, -1)
        q_hat_k = self.qa(f_view, g_rep)                         # [B*K]
        q_hat = q_hat_k.view(B, K).mean(dim=1)                   # [B]
        return q_hat

    def _gae(self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor,
             gamma: float, lam: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        rewards: [M] per-step rewards (concatenated)
        values:  [M+1] value estimates with bootstrap at end of episode
        dones:   [M] bool, True indicates episode terminal at step t
        returns: advantages[M], returns[M]
        """
        M = rewards.numel()
        adv = torch.zeros(M, device=rewards.device, dtype=torch.float32)
        last = 0.0
        for i in range(M-1, -1, -1):
            nonterminal = 1.0 - float(dones[i].item())
            delta = rewards[i] + gamma * nonterminal * values[i+1] - values[i]
            last = delta + gamma * lam * nonterminal * last
            adv[i] = last
        ret = adv + values[:-1]
        return adv, ret

    def _ppo_update(
        self,
        buffer: EpisodesBuffer,
        next_values: torch.Tensor,
        eps_now: float,
        ent_now: float,
    ) -> Dict[str, float]:
        # Convert to tensors
        data = buffer.to_tensors(self.device)
        g = data["g"]; h = data["h"]; F = data["F"]; mask = data["mask"]
        act = data["act"]; old_lp = data["old_logp"]; old_v = data["old_v"]
        rew = data["rew"]; done = data["done"]

        # The buffer is time-major: T consecutive blocks, each containing N parallel
        # scanpaths. Compute GAE per environment so trajectories never leak into one another.
        N = int(next_values.numel())
        if N <= 0 or rew.numel() % N != 0:
            raise ValueError("Invalid rollout shape for time-major GAE")
        T = rew.numel() // N
        reward_tm = rew.view(T, N)
        done_tm = done.view(T, N)
        old_value_tm = old_v.view(T, N)
        advantage_tm = torch.zeros_like(old_value_tm)
        last_advantage = torch.zeros(N, device=self.device)
        next_value = next_values.detach().to(self.device).reshape(N)
        for t in reversed(range(T)):
            nonterminal = (~done_tm[t]).to(old_value_tm.dtype)
            delta = reward_tm[t] + self.cfg.gamma * nonterminal * next_value - old_value_tm[t]
            last_advantage = (
                delta
                + self.cfg.gamma * self.cfg.gae_lambda * nonterminal * last_advantage
            )
            advantage_tm[t] = last_advantage
            next_value = old_value_tm[t]
        adv = advantage_tm.reshape(-1)
        ret = (advantage_tm + old_value_tm).reshape(-1)
        # Normalize adv
        adv = (adv - adv.mean()) / adv.std(unbiased=False).clamp_min(1e-8)

        # Now recompute new logp/value in graph (for grad)
        logits, values0 = self.policy.forward_step(h, g, F, mask)  # [M,X], [M]
        dist = torch.distributions.Categorical(logits=logits)
        new_lp = dist.log_prob(act)                    # [M]
        entropy = dist.entropy().mean()                # scalar

        ratio = (new_lp - old_lp).exp()
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0-eps_now, 1.0+eps_now) * adv
        policy_loss = -torch.mean(torch.minimum(surr1, surr2))
        if self.cfg.clip_value_loss:
            v_clip = old_v + (values0 - old_v).clamp(-self.cfg.vf_clip_param, self.cfg.vf_clip_param)
            v_loss = 0.5 * torch.mean(torch.maximum((values0 - ret).pow(2), (v_clip - ret).pow(2)))
        else:
            v_loss = 0.5 * torch.mean((values0 - ret).pow(2))
        ent_loss = -ent_now * entropy
        loss = policy_loss + self.cfg.value_coef * v_loss + ent_loss

        approx_kl = torch.mean(old_lp - new_lp).item()
        clipfrac = torch.mean((torch.abs(ratio - 1.0) > eps_now).float()).item()

        self.opt_policy.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
        self.opt_policy.step()

        return {
            "pi_loss": policy_loss.item(),
            "v_loss": v_loss.item(),
            "entropy": float(entropy.item()),
            "approx_kl": approx_kl,
            "clipfrac": clipfrac
        }

    # ---------- QA losses ----------

    def _qa_losses(self, I1: torch.Tensor, I2: torch.Tensor, Q1: torch.Tensor, Q2: torch.Tensor,
                   f_seq1: torch.Tensor, f_seq2: torch.Tensor, g1: torch.Tensor, g2: torch.Tensor) -> Tuple[torch.Tensor, Dict[str,float]]:
        """
        Build Eq.(12) losses with augmentations:
          L_total = β_mse L_mse + β_rank L_rank + β_cons L_cons + β_triplet L_triplet + β_cross L_cross-rank
        Returns total loss and logs.
        """
        cfg = self.cfg
        device = I1.device
        B = I1.shape[0]

        # Clean predictions from scanpaths
        q1 = self._qa_forward(f_seq1, g1)  # [B]
        q2 = self._qa_forward(f_seq2, g2)  # [B]

        # L_mse (supervised regression)
        L_mse = F.mse_loss(q1, Q1.to(device)) + F.mse_loss(q2, Q2.to(device))

        # L_rank (pairwise soft rank with margin δ=0.0)
        s = torch.sign(Q1 - Q2).to(device)
        L_rank = F.softplus(-s * (q1 - q2)).mean()

        # Augmentations for consistency, triplet, cross-rank
        # (We reuse global feature extractor on ERP-level for simplicity.)
        def _prep(x): return x  # already [B,3,512,1024]
        # Weak consistency
        with torch.no_grad():
            weak1 = torch.stack([self.to_tensor(self.weak_aug(Image.fromarray((x.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)))) for x in I1], dim=0).to(device)
            weak2 = torch.stack([self.to_tensor(self.weak_aug(Image.fromarray((x.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)))) for x in I2], dim=0).to(device)
        g1w, g2w = self.backbone.global_feat(weak1), self.backbone.global_feat(weak2)
        # Use the same scanpaths' f_seq as clean for simplicity (could resample).
        q1w = self._qa_forward(f_seq1, g1w); q2w = self._qa_forward(f_seq2, g2w)
        L_cons = F.mse_loss(q1, q1w) + F.mse_loss(q2, q2w)

        # Triplet (clean, mild, strong) — operate at ERP-level for the g feature, reuse f_seq
        with torch.no_grad():
            mild1 = torch.stack([self.to_tensor(self.mild_aug(Image.fromarray((x.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)))) for x in I1], dim=0).to(device)
            strong1 = torch.stack([self.to_tensor(self.strong_aug(Image.fromarray((x.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)))) for x in I1], dim=0).to(device)
            mild2 = torch.stack([self.to_tensor(self.mild_aug(Image.fromarray((x.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)))) for x in I2], dim=0).to(device)
            strong2 = torch.stack([self.to_tensor(self.strong_aug(Image.fromarray((x.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)))) for x in I2], dim=0).to(device)
        g1m, g1s = self.backbone.global_feat(mild1), self.backbone.global_feat(strong1)
        g2m, g2s = self.backbone.global_feat(mild2), self.backbone.global_feat(strong2)
        q1c, q2c = q1, q2
        q1m, q1s = self._qa_forward(f_seq1, g1m), self._qa_forward(f_seq1, g1s)
        q2m, q2s = self._qa_forward(f_seq2, g2m), self._qa_forward(f_seq2, g2s)
        m1, m2, m3 = 0.02, 0.10, 0.12
        L_triplet = (
            F.relu(q1m - q1c + m1).mean() + F.relu(q1s - q1m + m2).mean() + F.relu(q1s - q1c + m3).mean()
            + F.relu(q2m - q2c + m1).mean() + F.relu(q2s - q2m + m2).mean() + F.relu(q2s - q2c + m3).mean()
        )

        # Cross-rank after augmentation (use mild augmented)
        s_aug = torch.sign(Q1 - Q2).to(device)
        q1_aug = q1m; q2_aug = q2m
        L_cross = F.softplus(-s_aug * (q1_aug - q2_aug)).mean()

        L_total = (cfg.beta_mse * L_mse
                   + cfg.beta_rank * L_rank
                   + cfg.beta_cons * L_cons
                   + cfg.beta_triplet * L_triplet
                   + cfg.beta_cross * L_cross)

        logs = {
            "qa_mse": float(L_mse.item()),
            "qa_rank": float(L_rank.item()),
            "qa_cons": float(L_cons.item()),
            "qa_triplet": float(L_triplet.item()),
            "qa_cross": float(L_cross.item()),
        }
        return L_total, logs, q1.detach(), q2.detach()

    # ---------- Joint train loop ----------

    def train(self):
        cfg = self.cfg
        device = self.device
        start_time = time.time()

        for ep in range(cfg.epochs):
            eps_now = self.eps_sched(ep)
            ent_now = self.ent_sched(ep)
            lam_mse = self.lmse_sched(ep if ep < cfg.tpr_anneal_epochs else cfg.tpr_anneal_epochs-1)
            lam_rank = self.lrank_sched(ep if ep < cfg.tpr_anneal_epochs else cfg.tpr_anneal_epochs-1)

            self.policy.train(); self.qa.train(); self.backbone.train()

            for it, (imgs1, imgs2, Q1, Q2) in enumerate(self.loader):
                B = len(imgs1)
                # ERP tensors
                I1 = torch.stack([self._img_to_tensor(im) for im in imgs1], dim=0).to(device)  # [B,3,512,1024]
                I2 = torch.stack([self._img_to_tensor(im) for im in imgs2], dim=0).to(device)  # [B,3,512,1024]

                # Precompute candidates (viewport tensors, region features, global features)
                vp1, F1, g1 = self._precompute_candidates(I1)  # [B,X,3,224,224], [B,X,d_f], [B,1024]
                vp2, F2, g2 = self._precompute_candidates(I2)

                # Sample K scanpaths per image (with rewards from SER)
                scanpaths1, pack1, buffer1 = self._sample_scanpaths(F1, g1, vp1)
                scanpaths2, pack2, buffer2 = self._sample_scanpaths(F2, g2, vp2)

                # Task-level rewards via QA (q̂ from clean scanpaths) + SDR
                with torch.no_grad():
                    q1 = self._qa_forward(pack1["f_seq"], pack1["g"])  # [B]
                    q2 = self._qa_forward(pack2["f_seq"], pack2["g"])  # [B]
                # Pairwise supervised terms (R_mse, R_rank)
                R_mse = - ((q1 - Q1.to(device))**2 + (q2 - Q2.to(device))**2)   # [B]
                s = torch.sign(Q1 - Q2).to(device)
                R_rank = -F.softplus(-s * (q1 - q2))                            # [B]

                # SDR per image
                R_div1 = torch.tensor([self._diversity_reward(sp) for sp in scanpaths1], device=device, dtype=torch.float32)  # [B]
                R_div2 = torch.tensor([self._diversity_reward(sp) for sp in scanpaths2], device=device, dtype=torch.float32)  # [B]

                # Add TPR (annealed) + SDR to the LAST step rewards of each scanpath
                # pack["rewards"]: [B,K,T]
                add1 = R_div1 + lam_mse * R_mse + lam_rank * R_rank             # [B]
                add2 = R_div2 + lam_mse * R_mse + lam_rank * R_rank             # [B]
                pack1["rewards"][:, :, -1] += add1.unsqueeze(1)                 # broadcast to all K's last step
                pack2["rewards"][:, :, -1] += add2.unsqueeze(1)

                # Update the last-step rewards inside the original buffers (last B*K entries)
                Bsz = B; Ksz = self.cfg.K
                # buffer1
                offset1 = len(buffer1.data) - (Bsz * Ksz)
                for b in range(Bsz):
                    for k in range(Ksz):
                        bi = b*Ksz + k
                        idx_last = offset1 + bi
                        buffer1.data[idx_last] = buffer1.data[idx_last].__class__(
                            **{**buffer1.data[idx_last].__dict__, 'reward': buffer1.data[idx_last].reward + float(add1[b].item())}
                        )
                # buffer2
                offset2 = len(buffer2.data) - (Bsz * Ksz)
                for b in range(Bsz):
                    for k in range(Ksz):
                        bi = b*Ksz + k
                        idx_last = offset2 + bi
                        buffer2.data[idx_last] = buffer2.data[idx_last].__class__(
                            **{**buffer2.data[idx_last].__dict__, 'reward': buffer2.data[idx_last].reward + float(add2[b].item())}
                        )

                # PPO update (two halves) with current annealed ε/c_H
                next_values1 = pack1["values"][:, :, -1].reshape(-1)
                next_values2 = pack2["values"][:, :, -1].reshape(-1)
                logs1 = self._ppo_update(
                    buffer1, next_values=next_values1, eps_now=eps_now, ent_now=ent_now
                )
                logs2 = self._ppo_update(
                    buffer2, next_values=next_values2, eps_now=eps_now, ent_now=ent_now
                )

                # QA update (Eq.12)
                qa_loss, qa_logs, q1_clean, q2_clean = self._qa_losses(I1, I2, Q1, Q2, pack1["f_seq"], pack2["f_seq"], pack1["g"], pack2["g"])
                self.opt_qa.zero_grad(set_to_none=True)
                qa_loss.backward()
                nn.utils.clip_grad_norm_(self.trainable_qa_parameters, cfg.max_grad_norm)
                self.opt_qa.step()

                if (it+1) % cfg.log_interval == 0:
                    msg = (f"[ep {ep+1:03d}/{cfg.epochs}] it={it+1:03d} "
                           f"ε={eps_now:.3f} c_H={ent_now:.4f} "
                           f"pi(V1/V2)={logs1['pi_loss']:.3f}/{logs2['pi_loss']:.3f} "
                           f"V(V1/V2)={logs1['v_loss']:.3f}/{logs2['v_loss']:.3f} "
                           f"H={0.5*(logs1['entropy']+logs2['entropy']):.3f} "
                           f"KL={0.5*(logs1['approx_kl']+logs2['approx_kl']):.4f} "
                           f"clipfrac={0.5*(logs1['clipfrac']+logs2['clipfrac']):.3f} "
                           f"QA(Lmse/Lrank/cons/trip/cross)={qa_logs['qa_mse']:.3f}/{qa_logs['qa_rank']:.3f}/{qa_logs['qa_cons']:.3f}/{qa_logs['qa_triplet']:.3f}/{qa_logs['qa_cross']:.3f} "
                           f"q̂1={q1_clean.mean().item():.2f} q̂2={q2_clean.mean().item():.2f}")
                    print(msg)

            if cfg.save_dir:
                os.makedirs(cfg.save_dir, exist_ok=True)
                checkpoint_path = os.path.join(cfg.save_dir, f"epoch_{ep + 1:03d}.pt")
                torch.save(
                    {
                        "epoch": ep + 1,
                        "config": asdict(cfg),
                        "policy": self.policy.state_dict(),
                        "quality_assessor": self.qa.state_dict(),
                        "backbone": self.backbone.state_dict(),
                        "policy_optimizer": self.opt_policy.state_dict(),
                        "quality_optimizer": self.opt_qa.state_dict(),
                    },
                    checkpoint_path,
                )

        print(f"Training finished in {(time.time()-start_time)/60.0:.1f} min.")

    # ---------- Inference API ----------

    @torch.no_grad()
    def infer_quality(self, I_erp: Image.Image, K: int=15, T: int=7) -> float:
        """
        Inference as Sec.4.1.2: sample K scanpaths (length T), average Q̂_k.
        """
        self.policy.eval(); self.qa.eval(); self.backbone.eval()
        x = self._img_to_tensor(I_erp).unsqueeze(0).to(self.device)    # [1,3,512,1024]
        vp, F, g = self._precompute_candidates(x)                     # [1,X,3,224,224], [1,X,d_f], [1,1024]

        # Sample K scanpaths
        g1 = g; F1 = F; B=1; X=self.X; d_f=self.cfg.d_f
        h = torch.zeros(B*K, self.cfg.d_h, device=self.device)
        g_exp = g1.unsqueeze(1).repeat(1, K, 1).view(B*K, -1)
        F_exp = F1.unsqueeze(1).repeat(1, K, 1, 1).view(B*K, X, d_f)
        visited = torch.zeros(B*K, X, dtype=torch.bool, device=self.device)
        f_seq_sel = torch.zeros(B*K, T, d_f, device=self.device)
        prev_vp = None
        for t in range(T):
            mask = torch.where(visited, torch.full_like(visited, -1e9, dtype=torch.float32), torch.zeros_like(visited, dtype=torch.float32))
            logits, v_t = self.policy.forward_step(h, g_exp, F_exp, mask)
            dist = torch.distributions.Categorical(logits=logits)
            a_t = dist.sample()
            env_index = torch.arange(B*K, device=self.device)
            f_sel = F_exp[env_index, a_t]
            h = self.policy.gru(f_sel, h)
            visited[env_index, a_t] = True
            f_seq_sel[:, t] = f_sel
        q_hat = self._qa_forward(f_seq_sel.view(B, K, T, d_f), g1)    # [1]
        return float(q_hat.item())

# =============================================================================
# CLI
# =============================================================================

def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser("RL-ScanIQA — end-to-end training (PPO + QA)")
    # Data
    p.add_argument("--pairs_csv", type=str, default="")
    p.add_argument("--img_root", type=str, default="")
    # Scanpath
    p.add_argument("--n_yaw", type=int, default=8)
    p.add_argument("--n_pitch", type=int, default=4)
    p.add_argument("--FOV_deg", type=float, default=90.0)
    p.add_argument("--viewport_hw", type=int, default=224)
    p.add_argument("--K", type=int, default=5)
    p.add_argument("--T", type=int, default=4)
    # Backbones
    p.add_argument("--backbone", type=str, default="toy", choices=["toy", "dino"])
    p.add_argument("--d_f", type=int, default=512)
    p.add_argument("--d_h", type=int, default=512)
    # PPO/GAE
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--clip_start", type=float, default=0.20)
    p.add_argument("--clip_end", type=float, default=0.10)
    p.add_argument("--ent_start", type=float, default=0.02)
    p.add_argument("--ent_end", type=float, default=0.005)
    p.add_argument("--value_coef", type=float, default=0.5)
    p.add_argument("--clip_value_loss", action="store_true", default=False)
    p.add_argument("--vf_clip_param", type=float, default=0.2)
    p.add_argument("--target_kl", type=float, default=0.03)
    # TPR anneal
    p.add_argument("--tpr_anneal_epochs", type=int, default=30)
    # QA loss weights
    p.add_argument("--beta_mse", type=float, default=1.0)
    p.add_argument("--beta_rank", type=float, default=0.2)
    p.add_argument("--beta_cons", type=float, default=0.2)
    p.add_argument("--beta_triplet", type=float, default=0.3)
    p.add_argument("--beta_cross", type=float, default=0.2)
    # Optims
    p.add_argument("--lr_policy", type=float, default=3e-4)
    p.add_argument("--lr_qa", type=float, default=1e-4)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--adam_eps", type=float, default=1e-8)
    p.add_argument("--weight_decay", type=float, default=0.0)
    # Loop
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0)
    # Device & misc
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_interval", type=int, default=1)
    p.add_argument("--save_dir", type=str, default="")

    args = p.parse_args()
    cfg = TrainConfig(
        pairs_csv=args.pairs_csv,
        img_root=args.img_root,
        n_yaw=args.n_yaw,
        n_pitch=args.n_pitch,
        X=args.n_yaw*args.n_pitch,
        FOV_deg=args.FOV_deg,
        viewport_hw=args.viewport_hw,
        K=args.K, T=args.T,
        backbone=args.backbone, d_f=args.d_f, d_h=args.d_h,
        gamma=args.gamma, gae_lambda=args.gae_lambda,
        clip_start=args.clip_start, clip_end=args.clip_end,
        ent_start=args.ent_start, ent_end=args.ent_end,
        value_coef=args.value_coef, clip_value_loss=args.clip_value_loss, vf_clip_param=args.vf_clip_param,
        target_kl=args.target_kl,
        tpr_anneal_epochs=args.tpr_anneal_epochs,
        beta_mse=args.beta_mse, beta_rank=args.beta_rank, beta_cons=args.beta_cons,
        beta_triplet=args.beta_triplet, beta_cross=args.beta_cross,
        lr_policy=args.lr_policy, lr_qa=args.lr_qa, max_grad_norm=args.max_grad_norm, adam_eps=args.adam_eps, weight_decay=args.weight_decay,
        epochs=args.epochs, batch_size=args.batch_size, num_workers=args.num_workers,
        device=args.device, seed=args.seed, log_interval=args.log_interval, save_dir=args.save_dir
    )
    return cfg

def main():
    cfg = parse_args()
    print("=== RL-ScanIQA Config ===")
    print(json.dumps(asdict(cfg), indent=2))
    trainer = JointTrainer(cfg)
    trainer.train()

if __name__ == "__main__":
    main()
