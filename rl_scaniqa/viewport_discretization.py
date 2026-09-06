from __future__ import annotations
from typing import Tuple, List, Optional
import math
import torch
import torch.nn.functional as F

DEFAULT_NUM_YAW = 8
DEFAULT_NUM_PITCH = 4
DEFAULT_FOV_DEG = (90.0, 90.0)
DEFAULT_OUT_HW = (224, 224)


def deg2rad(deg: float) -> float:
    return deg * math.pi / 180.0


def rad2deg(rad: float) -> float:
    return rad * 180.0 / math.pi


def make_uniform_viewport_centers(
    num_yaw: int = DEFAULT_NUM_YAW,
    num_pitch: int = DEFAULT_NUM_PITCH,
) -> torch.Tensor:
    """
    Generate uniformly spaced viewport centers (yaw, pitch) covering the viewing sphere.

    Conventions :
    - yaw   ∈ [-π, π) is longitude (wrap horizontally)
    - pitch ∈ [-π/2, π/2] is latitude (clamp at poles)
    - centers lie at the midpoint of each partition interval

    Returns
    centers : torch.Tensor of shape (X, 2), where X = num_yaw * num_pitch.
              Order per row: [yaw(rad), pitch(rad)].
    """
    yaw_step = 2 * math.pi / num_yaw
    pitch_step = math.pi / num_pitch

    # Yaw centers: split [-π, π) into num_yaw bins and take the midpoints
    yaw_centers = torch.linspace(-math.pi, math.pi, steps=num_yaw + 1)[:-1] + yaw_step / 2

    # Pitch centers: split [-π/2, π/2] into num_pitch bins, use midpoints (avoid the exact poles)
    pitch_low = -math.pi / 2
    pitch_high = math.pi / 2
    pitch_edges = torch.linspace(pitch_low, pitch_high, steps=num_pitch + 1)
    pitch_centers = 0.5 * (pitch_edges[:-1] + pitch_edges[1:])

    # Meshgrid -> reshape to (X, 2) where X = num_yaw * num_pitch
    pitch_grid, yaw_grid = torch.meshgrid(pitch_centers, yaw_centers, indexing="ij")
    centers = torch.stack([yaw_grid.reshape(-1), pitch_grid.reshape(-1)], dim=-1)
    return centers


# Rotations and coordinate transforms
def _rotation_yaw_pitch(yaw: torch.Tensor, pitch: torch.Tensor) -> torch.Tensor:
    """
    Build rotation matrix R = R_yaw @ R_pitch for yaw (around Y axis) and pitch (around X axis).
    """
    yaw = yaw.reshape(-1)
    pitch = pitch.reshape(-1)
    assert yaw.shape == pitch.shape
    n = yaw.shape[0]

    cy, sy = torch.cos(yaw), torch.sin(yaw)
    cx, sx = torch.cos(pitch), torch.sin(pitch)

    # R_yaw (around Y)
    R_y = torch.zeros(n, 3, 3, dtype=yaw.dtype, device=yaw.device)
    R_y[:, 0, 0] = cy
    R_y[:, 0, 2] = sy
    R_y[:, 1, 1] = 1.0
    R_y[:, 2, 0] = -sy
    R_y[:, 2, 2] = cy

    # R_pitch (around X)
    R_x = torch.zeros(n, 3, 3, dtype=yaw.dtype, device=yaw.device)
    R_x[:, 0, 0] = 1.0
    R_x[:, 1, 1] = cx
    R_x[:, 1, 2] = -sx
    R_x[:, 2, 1] = sx
    R_x[:, 2, 2] = cx

    # Final rotation
    R = torch.bmm(R_y, R_x)
    return R


def _perspective_rays(
    out_h: int, out_w: int, fov_deg: Tuple[float, float], device=None, dtype=None
) -> torch.Tensor:
    """
    Generate unit ray directions in the camera frame for a perspective view.
    """
    dtype = dtype if dtype is not None else torch.float32
    device = device if device is not None else "cpu"

    fov_x = deg2rad(float(fov_deg[0]))
    fov_y = deg2rad(float(fov_deg[1]))

    # Normalized pixel coordinates in [-1, 1]
    nx = torch.linspace(-1.0, 1.0, out_w, dtype=dtype, device=device)
    ny = torch.linspace(-1.0, 1.0, out_h, dtype=dtype, device=device)
    yy, xx = torch.meshgrid(ny, nx, indexing="ij")  # (out_h, out_w)

    # Camera-space rays (before normalization)
    x_cam = torch.tan(xx * (fov_x * 0.5))
    y_cam = torch.tan(yy * (fov_y * 0.5))
    z_cam = torch.ones_like(x_cam)

    # Flip y so that "image v increasing (down)" maps to "3D y decreasing"
    rays = torch.stack([x_cam, -y_cam, z_cam], dim=-1)  # (H, W, 3)

    # Normalize to unit directions
    rays = rays / torch.linalg.norm(rays, dim=-1, keepdim=True).clamp_min(1e-8)
    return rays


def _world_directions(
    rays_cam: torch.Tensor, R: torch.Tensor
) -> torch.Tensor:
    """
    Rotate camera-frame rays into the world frame.

    Inputs
    ------
    rays_cam : (H, W, 3) camera-frame unit directions
    R        : (3, 3) or (N, 3, 3) world-from-camera rotation(s)

    Returns
    -------
    dirs_world : (H, W, 3) if R is (3, 3), or (N, H, W, 3) if R is (N, 3, 3).
    """
    H, W, _ = rays_cam.shape
    if R.ndim == 2:
        dirs = torch.einsum("ij,hwj->hwi", R, rays_cam)
    else:
        n = R.shape[0]
        rc = rays_cam.reshape(1, H, W, 3).repeat(n, 1, 1, 1)
        dirs = torch.einsum("nij,nhwj->nhwi", R, rc)
    return dirs


def _world_to_spherical(dirs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert world-frame directions to spherical angles (yaw=longitude, pitch=latitude).
    """
    x = dirs[..., 0]
    y = dirs[..., 1]
    z = dirs[..., 2]

    yaw = torch.atan2(x, z)  # [-π, π)
    pitch = torch.atan2(y, torch.sqrt(x * x + z * z + 1e-8)).clamp(-math.pi/2, math.pi/2)
    return yaw, pitch


def _spherical_to_erp_grid(
    yaw: torch.Tensor,
    pitch: torch.Tensor,
    erp_h: int,
    erp_w: int,
) -> torch.Tensor:
    """
    Convert spherical angles to ERP sampling grid for torch.grid_sample.
    """
    two_pi = 2 * math.pi

    # Wrap yaw into [-π, π)
    yaw_wrapped = (yaw + math.pi) - torch.floor((yaw + math.pi) / two_pi) * two_pi - math.pi

    # Clamp pitch into [-π/2, π/2]
    pitch_clamped = pitch.clamp(-math.pi / 2, math.pi / 2)

    # Map to [0,1]
    u = (yaw_wrapped + math.pi) / (2 * math.pi)
    v = (0.5 - pitch_clamped / math.pi)

    # Map to [-1,1] as expected by grid_sample
    gx = u * 2.0 - 1.0
    gy = v * 2.0 - 1.0
    grid = torch.stack([gx, gy], dim=-1)
    return grid


# ERP -> perspective viewport(s)
@torch.no_grad()
def erp_to_viewport(
    erp: torch.Tensor,
    center_yaw: float,
    center_pitch: float,
    fov_deg: Tuple[float, float] = DEFAULT_FOV_DEG,
    out_hw: Tuple[int, int] = DEFAULT_OUT_HW,
    align_corners: bool = True,
) -> torch.Tensor:
    """
    Sample a single perspective viewport from an ERP image, given center (yaw,pitch) and FOV.
    """
    assert erp.ndim == 4, "erp must be (B, C, H, W)"
    B, C, H, W = erp.shape
    out_h, out_w = out_hw

    device = erp.device
    dtype = erp.dtype

    # Build camera-frame rays for the output resolution
    rays_cam = _perspective_rays(out_h, out_w, fov_deg, device=device, dtype=dtype)

    # Compute rotation from camera -> world for the given center (yaw,pitch)
    R = _rotation_yaw_pitch(
        yaw=torch.tensor(center_yaw, dtype=dtype, device=device),
        pitch=torch.tensor(center_pitch, dtype=dtype, device=device),
    )[0]

    # Rotate rays into world frame, then convert to spherical
    dirs_world = _world_directions(rays_cam, R)
    yaw, pitch = _world_to_spherical(dirs_world)

    # Build ERP sampling grid with yaw-wrapping & pitch-clamping
    grid = _spherical_to_erp_grid(yaw, pitch, erp_h=H, erp_w=W)

    # Sample from ERP (bilinear)
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)
    vp = F.grid_sample(
        erp, grid, mode="bilinear", padding_mode="border", align_corners=align_corners
    )
    return vp


@torch.no_grad()
def erp_to_all_viewports(
    erp: torch.Tensor,
    centers: Optional[torch.Tensor] = None,
    num_yaw: int = DEFAULT_NUM_YAW,
    num_pitch: int = DEFAULT_NUM_PITCH,
    fov_deg: Tuple[float, float] = DEFAULT_FOV_DEG,
    out_hw: Tuple[int, int] = DEFAULT_OUT_HW,
    align_corners: bool = True,
) -> torch.Tensor:
    """
    Sample all candidate viewports in one pass (looping over X centers).
    """
    assert erp.ndim == 4, "erp must be (B, C, H, W)"
    device = erp.device
    dtype = erp.dtype

    if centers is None:
        centers = make_uniform_viewport_centers(num_yaw=num_yaw, num_pitch=num_pitch).to(device=device, dtype=dtype)
    else:
        assert centers.ndim == 2 and centers.shape[-1] == 2, "centers must be (X, 2)"
        centers = centers.to(device=device, dtype=dtype)

    X = centers.shape[0]
    out_h, out_w = out_hw

    vps: List[torch.Tensor] = []
    for i in range(X):
        yaw_i = float(centers[i, 0].item())
        pitch_i = float(centers[i, 1].item())
        vp_i = erp_to_viewport(
            erp=erp, center_yaw=yaw_i, center_pitch=pitch_i,
            fov_deg=fov_deg, out_hw=out_hw, align_corners=align_corners
        )
        vps.append(vp_i)

    vps = torch.stack(vps, dim=1)
    return vps


# Convenience helpers
def viewport_grid_shape(num_yaw: int = DEFAULT_NUM_YAW, num_pitch: int = DEFAULT_NUM_PITCH) -> Tuple[int, int]:
    """Return (num_pitch, num_yaw), useful when reshaping X back to a 2D grid."""
    return (num_pitch, num_yaw)


def index_to_yaw_pitch(index: int, num_yaw: int = DEFAULT_NUM_YAW, num_pitch: int = DEFAULT_NUM_PITCH) -> Tuple[float, float]:
    """
    Map a linear index in [0..X-1] back to the center (yaw, pitch) in radians.
    """
    centers = make_uniform_viewport_centers(num_yaw=num_yaw, num_pitch=num_pitch)
    yaw, pitch = centers[index, 0].item(), centers[index, 1].item()
    return yaw, pitch
