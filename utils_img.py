from __future__ import annotations
import numpy as np
import torch

def z0(img_u8: np.ndarray) -> torch.Tensor:
    assert img_u8.ndim == 3 and img_u8.shape[2] == 3
    return torch.from_numpy(img_u8).permute(2, 0, 1).float() / 255.0

def z1(t: torch.Tensor) -> np.ndarray:
    if t.ndim != 3:
        raise ValueError('Expected CHW tensor')
    z3 = (t.clamp(0, 1) * 255.0 + 0.5).to(torch.uint8)
    return z3.permute(1, 2, 0).cpu().numpy()

def z2(sigma_255: float, H: int, W: int) -> torch.Tensor:
    return torch.full((1, H, W), float(sigma_255) / 255.0, dtype=torch.float32)
u8_hwc_to_t01_chw=z0
t01_chw_to_u8_hwc=z1
make_sigma_map=z2

