from __future__ import annotations
import io, random, math
from dataclasses import dataclass
from typing import List, Tuple, Callable

import numpy as np
from PIL import Image
import cv2
import torch
import torchvision.transforms as T


class JPEGCompression:

    def __init__(self, qmin: int, qmax: int):
        self.qmin, self.qmax = int(qmin), int(qmax)
    def __call__(self, img: Image.Image) -> Image.Image:
        q = random.randint(self.qmin, self.qmax)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q, optimize=True)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

class PoissonNoise:

    def __init__(self, lam_min: float, lam_max: float):
        self.lam_min, self.lam_max = float(lam_min), float(lam_max)
    def __call__(self, img: Image.Image) -> Image.Image:
        lam = random.uniform(self.lam_min, self.lam_max)
        x = np.asarray(img).astype(np.float32) / 255.0              # [H,W,3], 0~1
        y = np.random.poisson(x * lam) / max(lam, 1e-8)
        y = np.clip(y, 0.0, 1.0)
        return Image.fromarray((y * 255.0).astype(np.uint8))

class MotionBlur:

    def __init__(self, k_min: int, k_max: int, angle_min: float = 0.0, angle_max: float = 180.0):
        self.k_min, self.k_max = int(k_min), int(k_max)
        self.angle_min, self.angle_max = float(angle_min), float(angle_max)
    def __call__(self, img: Image.Image) -> Image.Image:
        k = (random.randint(self.k_min, self.k_max) | 1)
        angle = random.uniform(self.angle_min, self.angle_max)
        kernel = np.zeros((k, k), dtype=np.float32)
        kernel[k // 2, :] = 1.0
        M = cv2.getRotationMatrix2D((k/2 - 0.5, k/2 - 0.5), angle, 1.0)
        kernel = cv2.warpAffine(kernel, M, (k, k))
        kernel = kernel / max(kernel.sum(), 1e-8)
        x = np.asarray(img)
        y = cv2.filter2D(x, -1, kernel, borderType=cv2.BORDER_REPLICATE)
        return Image.fromarray(np.clip(y, 0, 255).astype(np.uint8))

class DefocusBlur:

    def __init__(self, r_min: int, r_max: int):
        self.r_min, self.r_max = int(r_min), int(r_max)
    def __call__(self, img: Image.Image) -> Image.Image:
        r = random.randint(self.r_min, self.r_max)
        k = 2 * r + 1
        yy, xx = np.ogrid[-r:r+1, -r:r+1]
        mask = (xx*xx + yy*yy) <= (r*r)
        kernel = np.zeros((k, k), dtype=np.float32)
        kernel[mask] = 1.0
        kernel /= max(kernel.sum(), 1e-8)
        x = np.asarray(img)
        y = cv2.filter2D(x, -1, kernel, borderType=cv2.BORDER_REPLICATE)
        return Image.fromarray(np.clip(y, 0, 255).astype(np.uint8))

class RandomOneOf:

    def __init__(self, transforms: List[Callable[[Image.Image], Image.Image]]):
        self.transforms = list(transforms)
    def __call__(self, img: Image.Image) -> Image.Image:
        t = random.choice(self.transforms)
        return t(img)




@dataclass
class AugPipelines:
    weak: T.Compose
    mild: T.Compose
    strong: T.Compose
    to_tensor: T.ToTensor

def make_qa_augs(resize_hw: Tuple[int, int] = (512, 1024)) -> AugPipelines:

    resize = T.Resize(resize_hw)


    weak_dist = RandomOneOf([
        JPEGCompression(85, 95),
        T.GaussianBlur(kernel_size=3, sigma=(0.3, 0.8)),
        MotionBlur(3, 7),
        DefocusBlur(1, 2),
        T.ColorJitter(0.06, 0.06, 0.06, 0.03),
        PoissonNoise(50.0, 80.0),       #
    ])


    mild_dist = RandomOneOf([
        JPEGCompression(60, 75),
        T.GaussianBlur(kernel_size=5, sigma=(0.8, 1.5)),
        MotionBlur(7, 11),
        DefocusBlur(2, 3),
        T.ColorJitter(0.15, 0.15, 0.15, 0.07),
        PoissonNoise(18.0, 30.0),
    ])


    strong_dist = RandomOneOf([
        JPEGCompression(20, 40),
        T.GaussianBlur(kernel_size=7, sigma=(1.8, 3.0)),
        MotionBlur(11, 19),
        DefocusBlur(4, 6),
        T.ColorJitter(0.28, 0.28, 0.28, 0.12),
        PoissonNoise(6.0, 12.0),
    ])

    return AugPipelines(
        weak=T.Compose([resize, weak_dist]),
        mild=T.Compose([resize, mild_dist]),
        strong=T.Compose([resize, strong_dist]),
        to_tensor=T.ToTensor()
    )




def apply_aug_batch(x_bchw: torch.Tensor, aug: Callable[[Image.Image], Image.Image],
                    to_tensor: Callable[[Image.Image], torch.Tensor]) -> torch.Tensor:

    assert x_bchw.ndim == 4 and x_bchw.shape[1] == 3, "expect [B,3,H,W]"
    b, _, h, w = x_bchw.shape
    imgs = []
    for i in range(b):
        # [3,H,W] → [H,W,3] → PIL
        pil = T.ToPILImage()(x_bchw[i].cpu())
        pil_aug = aug(pil)
        imgs.append(to_tensor(pil_aug))

    return torch.stack(imgs, dim=0)
