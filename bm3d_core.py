from __future__ import annotations
import os
import sys
import time
import numpy as np
import cv2
from scipy.fft import dct as sp_dct, idct as sp_idct
from numpy.lib.stride_tricks import sliding_window_view
import tkinter as tk
from tkinter import filedialog

def z0(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)

def z1(img_u8: np.ndarray) -> np.ndarray:
    return img_u8.astype(np.float32) / 255.0

def z2(img01: np.ndarray) -> np.ndarray:
    return (z0(img01) * 255.0 + 0.5).astype(np.uint8)

def z3(img01: np.ndarray, sigma_255: float, seed: int=0) -> np.ndarray:
    sigma01 = float(sigma_255) / 255.0
    z25 = np.random.default_rng(seed)
    return img01 + z25.normal(0.0, sigma01, size=img01.shape).astype(np.float32)

def z4(a01: np.ndarray, b01: np.ndarray) -> float:
    z26 = a01.astype(np.float64)
    z27 = b01.astype(np.float64)
    z28 = np.mean((z26 - z27) ** 2)
    if z28 < 1e-15:
        return 99.0
    return 10.0 * np.log10(1.0 / z28)

def z5(x: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(x.astype(np.float32), (0, 0), sigma)

def z6(img_bgr_u8: np.ndarray) -> bool:
    z27, z7c, z63 = cv2.split(img_bgr_u8)
    return np.array_equal(z27, z7c) and np.array_equal(z7c, z63)
C20 = np.array([[1 / 3, 1 / 3, 1 / 3], [0.5, 0.0, -0.5], [0.25, -0.5, 0.25]], dtype=np.float32)
C21 = np.array([[0.299, 0.587, 0.114], [-0.16873660714285, -0.33126339285715, 0.5], [0.5, -0.4186875, -0.0813125]], dtype=np.float32)

def z7(rgb01: np.ndarray, mode: str):
    mode = mode.lower()
    if mode == 'opp':
        A = C20
    elif mode in ['ycbcr', 'ycrcb']:
        A = C21
    else:
        raise ValueError("colorspace must be 'opp' or 'ycbcr'")
    maxV = np.sum(A * (A > 0), axis=1)
    minV = np.sum(A * (A < 0), axis=1)
    z29 = (maxV - minV).astype(np.float32)
    z2a = rgb01.reshape(-1, 3).astype(np.float32)
    y = z2a @ A.T
    y = (y - minV[None, :]) / z29[None, :]
    lc01 = y.reshape(rgb01.shape).astype(np.float32)
    z2b = np.sqrt(np.sum(A * A, axis=1)).astype(np.float32)
    z2c = (z2b / z29).astype(np.float32)
    return (lc01, A.astype(np.float32), maxV.astype(np.float32), minV.astype(np.float32), z2c)

def z8(lc01: np.ndarray, A: np.ndarray, maxV: np.ndarray, minV: np.ndarray):
    z29 = (maxV - minV).astype(np.float32)
    z2a = lc01.reshape(-1, 3).astype(np.float32)
    x = z2a * z29[None, :] + minV[None, :]
    C2d = np.linalg.inv(A).astype(np.float32)
    z2e = x @ C2d.T
    return z2e.reshape(lc01.shape).astype(np.float32)

def z9(img_bgr_u8: np.ndarray, colorspace: str):
    rgb01 = cv2.cvtColor(img_bgr_u8, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return z7(rgb01, colorspace)

def za(lc01: np.ndarray, A: np.ndarray, maxV: np.ndarray, minV: np.ndarray):
    rgb01 = z8(lc01, A, maxV, minV)
    rgb01 = z0(rgb01)
    z2f = cv2.cvtColor((rgb01 * 255.0 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR)
    return z2f

def zb(x: np.ndarray) -> np.ndarray:
    return sp_dct(x, axis=0, norm='ortho')

def zc(x: np.ndarray) -> np.ndarray:
    return sp_idct(x, axis=0, norm='ortho')

def zd(x: np.ndarray) -> np.ndarray:
    y = sp_dct(x, axis=-1, norm='ortho')
    y = sp_dct(y, axis=-2, norm='ortho')
    return y.astype(np.float32)

def ze(x: np.ndarray) -> np.ndarray:
    y = sp_idct(x, axis=-1, norm='ortho')
    y = sp_idct(y, axis=-2, norm='ortho')
    return y.astype(np.float32)
C22 = np.array([[0.353553390593274, 0.353553390593274, 0.353553390593274, 0.353553390593274, 0.353553390593274, 0.353553390593274, 0.353553390593274, 0.353553390593274], [0.219417649252501, 0.449283757993216, 0.449283757993216, 0.219417649252501, -0.219417649252501, -0.449283757993216, -0.449283757993216, -0.219417649252501], [0.569359398342846, 0.402347308162278, -0.402347308162278, -0.569359398342846, -0.083506045090284, 0.083506045090284, -0.083506045090284, 0.083506045090284], [-0.083506045090284, 0.083506045090284, -0.083506045090284, 0.083506045090284, 0.569359398342846, 0.402347308162278, -0.402347308162278, -0.569359398342846], [0.707106781186547, -0.707106781186547, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.707106781186547, -0.707106781186547, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.707106781186547, -0.707106781186547, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.707106781186547, -0.707106781186547]], dtype=np.float32)
C23 = np.linalg.inv(C22).astype(np.float32)

def zf(blocks: np.ndarray) -> np.ndarray:
    C30 = C22
    z31 = blocks @ C30.T
    z32 = C30 @ z31
    return z32.astype(np.float32)

def z10(coeffs: np.ndarray) -> np.ndarray:
    C33 = C23
    z31 = coeffs @ C33.T
    z32 = C33 @ z31
    return z32.astype(np.float32)

class C11:

    def __init__(self, profile: str, sigma_255: float):
        self.profile = profile.lower()
        self.sigma_255 = float(sigma_255)
        self.sigma01 = self.sigma_255 / 255.0
        self.N1 = 8
        self.Nstep = 2
        self.N2 = 16
        self.Ns = 39
        self.tau_match = 3000.0
        self.lambda_thr2D = 0.0
        self.lambda_thr3D = 2.6
        self.beta = 2.0
        self.N1_w = 8
        self.Nstep_w = 2
        self.N2_w = 32
        self.Ns_w = 39
        self.tau_match_w = 400.0
        self.beta_w = 2.0
        self.thrToIncStep = 8.0
        self.eps = 1e-12
        if self.profile == 'lc':
            self.Nstep = 6
            self.Ns = 25
            self.Nstep_w = 5
            self.N2_w = 16
            self.Ns_w = 25
            self.thrToIncStep = 3.0
        if self.profile in ['vn', 'vn_old'] or self.sigma_255 > 40:
            self.N2 = 32
            self.Nstep = 4
            self.lambda_thr3D = 2.8
            self.thrToIncStep = 3.0
            self.tau_match_w = 3500.0
            self.tau_match = 25000.0
            self.Ns_w = 39
        if self.profile == 'high':
            self.Nstep = 2
            self.Nstep_w = 2
            self.lambda_thr3D = 2.5
            self.beta = 2.5
            self.beta_w = 1.5
        self.match_lf = 6
        self.use_hybrid_select = True
        self.kaiser_ht = self._kaiser2d(self.N1, self.beta)
        self.kaiser_w = self._kaiser2d(self.N1_w, self.beta_w)
        self.tau_match_norm = self.tau_match * (self.N1 * self.N1) / (255.0 * 255.0)
        self.thrToIncStep01 = self.thrToIncStep / 255.0
        self.tau_match_w_norm = self.tau_match_w * (self.N1_w * self.N1_w) / (255.0 * 255.0)

    @staticmethod
    def _kaiser2d(n: int, beta: float) -> np.ndarray:
        z51 = np.kaiser(n, beta).astype(np.float32)
        return (z51[:, None] * z51[None, :]).astype(np.float32)

def z12(channel01: np.ndarray, N: int, tile_rows: int=0) -> np.ndarray:
    if N != 8:
        raise ValueError('Step1 bior1.5 hardcoded only for N=8. Use N1=8.')
    H, W = channel01.shape
    z34 = H - N + 1
    z35 = W - N + 1
    z32 = np.empty((z34, z35, N, N), dtype=np.float32)
    if tile_rows <= 0:
        blocks = sliding_window_view(channel01, (N, N))
        z32[:] = zf(blocks)
        return z32
    y = 0
    while y < z34:
        z7d = min(z34, y + tile_rows)
        blocks = sliding_window_view(channel01[y:z7d + N - 1, :], (N, N))
        z32[y:z7d] = zf(blocks)
        y = z7d
    return z32

def z13(channel01: np.ndarray, N: int, tile_rows: int=0) -> np.ndarray:
    H, W = channel01.shape
    z34 = H - N + 1
    z35 = W - N + 1
    z32 = np.empty((z34, z35, N, N), dtype=np.float32)
    if tile_rows <= 0:
        blocks = sliding_window_view(channel01, (N, N))
        z32[:] = zd(blocks)
        return z32
    y = 0
    while y < z34:
        z7d = min(z34, y + tile_rows)
        blocks = sliding_window_view(channel01[y:z7d + N - 1, :], (N, N))
        z32[y:z7d] = zd(blocks)
        y = z7d
    return z32

def z14(H: int, W: int, ry: int, rx: int, N: int, Ns: int):
    z36 = (Ns - 1) // 2
    z37 = max(0, ry - z36)
    z38 = max(0, rx - z36)
    z39 = min(H - N, ry + z36)
    z3a = min(W - N, rx + z36)
    return (z37, z39, z38, z3a)

def z15(luma_coeff: np.ndarray, ry: int, rx: int, N: int, Ns: int, tau: float, max_match: int, lf: int, sigma_luma01: float, lambda_thr2D: float, use_hybrid: bool=True):
    C7e, C7f = luma_coeff.shape[:2]
    z37, z39, z38, z3a = z14(C7e + N - 1, C7f + N - 1, ry, rx, N, Ns)
    z37 = max(0, min(z37, C7e - 1))
    z38 = max(0, min(z38, C7f - 1))
    z39 = max(0, min(z39, C7e - 1))
    z3a = max(0, min(z3a, C7f - 1))
    z3b = luma_coeff[ry, rx]
    lf = int(np.clip(lf, 2, N))
    z3c = z3b[:lf, :lf]
    z3d = luma_coeff[z37:z39 + 1, z38:z3a + 1, :lf, :lf]
    if lambda_thr2D and lambda_thr2D > 0:
        z47 = float(lambda_thr2D) * float(sigma_luma01)
        z3c = np.where(np.abs(z3c) < z47, 0.0, z3c)
        z3d = np.where(np.abs(z3d) < z47, 0.0, z3d)
    z3e = z3d - z3c
    z3f = np.mean(z3e * z3e, axis=(-1, -2))
    z80, z81 = z3f.shape
    z40 = z3f.reshape(-1)
    z41 = ry - z37
    z42 = rx - z38
    z43 = z41 * z81 + z42
    z44 = min(max_match, z40.size)
    if use_hybrid:
        z82 = np.flatnonzero(z40 < tau)
        if z82.size > 0:
            if z82.size > z44:
                z83 = z82[np.argpartition(z40[z82], z44 - 1)[:z44]]
            else:
                z83 = z82
        else:
            z83 = np.argpartition(z40, z44 - 1)[:z44]
    else:
        z83 = np.argpartition(z40, z44 - 1)[:z44]
    if z43 not in z83:
        if z83.size < z44:
            z83 = np.append(z83, z43)
        else:
            z91 = np.argmax(z40[z83])
            z83[z91] = z43
    z45 = z83 // z81
    z46 = z83 % z81
    return [(int(z37 + za1), int(z38 + za2)) for za1, za2 in zip(z45, z46)]

def z16(group_coeff: np.ndarray, sigma01: float, lam3d: float):
    C84, N, _ = group_coeff.shape
    z47 = lam3d * sigma01
    C48 = group_coeff.reshape(C84, N * N)
    C49 = zb(C48)
    C49[np.abs(C49) < z47] = 0.0
    z4a = int(np.count_nonzero(C49))
    F = zc(C49).astype(np.float32)
    return (F.reshape(C84, N, N), z4a)

def z17(group_basic: np.ndarray, group_noisy: np.ndarray, sigma01: float):
    C84, N, _ = group_basic.shape
    C4b = N * N
    C2d = group_basic.reshape(C84, C4b)
    C4c = group_noisy.reshape(C84, C4b)
    C4d = zb(C2d)
    C4e = zb(C4c)
    z4f = C4d * C4d
    z50 = z4f + sigma01 * sigma01
    z51 = np.divide(z4f, z50, out=np.zeros_like(z4f), where=z50 > 0)
    C52 = z51 * C4e
    z53 = float(np.sum(z51 * z51))
    F = zc(C52).astype(np.float32)
    return (F.reshape(C84, N, N), z53)

def z18(acc: np.ndarray, wacc: np.ndarray, group_coeff: np.ndarray, positions: list, kaiser2d: np.ndarray, wscalar: float):
    blocks = z10(group_coeff)
    W = (wscalar * kaiser2d).astype(np.float32)
    N = blocks.shape[-1]
    for z44, (y, x) in enumerate(positions):
        acc[y:y + N, x:x + N] += W * blocks[z44]
        wacc[y:y + N, x:x + N] += W

def z19(acc: np.ndarray, wacc: np.ndarray, group_dct: np.ndarray, positions: list, kaiser2d: np.ndarray, wscalar: float):
    blocks = ze(group_dct)
    W = (wscalar * kaiser2d).astype(np.float32)
    N = blocks.shape[-1]
    for z44, (y, x) in enumerate(positions):
        acc[y:y + N, x:x + N] += W * blocks[z44]
        wacc[y:y + N, x:x + N] += W

def z1a(luma01: np.ndarray, y: int, x: int, N: int) -> float:
    return float(np.std(luma01[y:y + N, x:x + N]))

def z1b(z_lc01: np.ndarray, p: BM3DParams, sigma_lc01: np.ndarray, tile_rows: int=0) -> np.ndarray:
    H, W, _ = z_lc01.shape
    N = p.N1
    if N != 8:
        raise ValueError('Step1 bior1.5 only supports N1=8 in this code.')
    z54 = z_lc01[..., 0].astype(np.float32)
    z55 = z12(z_lc01[..., 0].astype(np.float32), N, tile_rows)
    z56 = z12(z_lc01[..., 1].astype(np.float32), N, tile_rows)
    z57 = z12(z_lc01[..., 2].astype(np.float32), N, tile_rows)
    acc = np.zeros((H, W, 3), dtype=np.float32)
    wacc = np.zeros((H, W, 3), dtype=np.float32)
    z85, z86, z87 = map(float, sigma_lc01)
    y = 0
    while y <= H - N:
        x = 0
        while x <= W - N:
            z92 = z1a(z54, y, x, N)
            z93 = p.Nstep * 2 if z92 < p.thrToIncStep01 else p.Nstep
            z94 = z15(z55, y, x, N, p.Ns, p.tau_match_norm, p.N2, lf=p.match_lf, sigma_luma01=z85, lambda_thr2D=p.lambda_thr2D, use_hybrid=p.use_hybrid_select)
            z95 = np.stack([z55[za1, za2] for za1, za2 in z94], axis=0)
            z96 = np.stack([z56[za1, za2] for za1, za2 in z94], axis=0)
            z97 = np.stack([z57[za1, za2] for za1, za2 in z94], axis=0)
            za3, za4 = z16(z95, z85, p.lambda_thr3D)
            za5, za6 = z16(z96, z86, p.lambda_thr3D)
            za7, za8 = z16(z97, z87, p.lambda_thr3D)
            z98 = max(1, za4 + za6 + za8)
            wscalar = 1.0 / ((z85 * z85 + z86 * z86 + z87 * z87) / 3.0 * z98 + p.eps)
            z18(acc[..., 0], wacc[..., 0], za3, z94, p.kaiser_ht, wscalar)
            z18(acc[..., 1], wacc[..., 1], za5, z94, p.kaiser_ht, wscalar)
            z18(acc[..., 2], wacc[..., 2], za7, z94, p.kaiser_ht, wscalar)
            x += z93
        y += p.Nstep
    z32 = np.empty_like(z_lc01)
    for z58 in range(3):
        z88 = wacc[..., z58]
        z89 = acc[..., z58]
        z8a = z88 > 0
        z8b = np.copy(z89)
        z8b[z8a] = z89[z8a] / z88[z8a]
        z8b[~z8a] = z_lc01[..., z58][~z8a]
        z32[..., z58] = z8b
    return z32

def z1c(z_lc01: np.ndarray, basic_lc01: np.ndarray, p: BM3DParams, sigma_lc01: np.ndarray, tile_rows: int=0) -> np.ndarray:
    H, W, _ = z_lc01.shape
    N = p.N1_w
    z59 = basic_lc01[..., 0].astype(np.float32)
    z5a = z13(z59, N, tile_rows)
    z5b = [z13(basic_lc01[..., z58].astype(np.float32), N, tile_rows) for z58 in range(3)]
    z5c = [z13(z_lc01[..., z58].astype(np.float32), N, tile_rows) for z58 in range(3)]
    acc = np.zeros((H, W, 3), dtype=np.float32)
    wacc = np.zeros((H, W, 3), dtype=np.float32)
    z85, z86, z87 = map(float, sigma_lc01)
    y = 0
    while y <= H - N:
        x = 0
        while x <= W - N:
            z92 = z1a(z59, y, x, N)
            z93 = p.Nstep_w * 2 if z92 < p.thrToIncStep01 else p.Nstep_w
            z94 = z15(z5a, y, x, N, p.Ns_w, p.tau_match_w_norm, p.N2_w, lf=p.match_lf, sigma_luma01=z85, lambda_thr2D=p.lambda_thr2D, use_hybrid=p.use_hybrid_select)
            z99 = np.stack([z5b[0][za1, za2] for za1, za2 in z94], axis=0)
            z9a = np.stack([z5b[1][za1, za2] for za1, za2 in z94], axis=0)
            z9b = np.stack([z5b[2][za1, za2] for za1, za2 in z94], axis=0)
            z9c = np.stack([z5c[0][za1, za2] for za1, za2 in z94], axis=0)
            z9d = np.stack([z5c[1][za1, za2] for za1, za2 in z94], axis=0)
            z9e = np.stack([z5c[2][za1, za2] for za1, za2 in z94], axis=0)
            za9, zaa = z17(z99, z9c, z85)
            zab, zac = z17(z9a, z9d, z86)
            zad, zae = z17(z9b, z9e, z87)
            z53 = max(p.eps, zaa + zac + zae)
            wscalar = 1.0 / ((z85 * z85 + z86 * z86 + z87 * z87) / 3.0 * z53 + p.eps)
            z19(acc[..., 0], wacc[..., 0], za9, z94, p.kaiser_w, wscalar)
            z19(acc[..., 1], wacc[..., 1], zab, z94, p.kaiser_w, wscalar)
            z19(acc[..., 2], wacc[..., 2], zad, z94, p.kaiser_w, wscalar)
            x += z93
        y += p.Nstep_w
    z32 = np.empty_like(z_lc01)
    for z58 in range(3):
        z88 = wacc[..., z58]
        z89 = acc[..., z58]
        z8a = z88 > 0
        z8b = np.copy(z89)
        z8b[z8a] = z89[z8a] / z88[z8a]
        z8b[~z8a] = z_lc01[..., z58][~z8a]
        z32[..., z58] = z8b
    return z32

def z1d(noisy_bgr_u8: np.ndarray, denoised_bgr_u8: np.ndarray, sigma_255: float):
    z5d = z1(noisy_bgr_u8)
    z5e = z1(denoised_bgr_u8)
    sigma01 = sigma_255 / 255.0
    z5f = 1e-12
    z60 = z5d - z5e
    z61 = float(np.sqrt(max(np.mean(z60 * z60), z5f)))
    z62 = float(np.clip(sigma01 / (z61 + z5f), 0.0, 1.0))
    z5e = z62 * z5e + (1.0 - z62) * z5d
    z63 = z5d - z5e
    z64 = np.mean(z63 * z63, axis=2)
    z65 = z5(z64, 1.5)
    z51 = np.sqrt(sigma01 * sigma01 / (z65 + z5f))
    z51 = np.clip(z51, 0.0, 1.0).astype(np.float32)
    z51 = z5(z51, 1.0)
    z51 = np.clip(z51, 0.0, 1.0).astype(np.float32)
    z66 = z51[..., None] * z5e + (1.0 - z51[..., None]) * z5d
    z67 = z2(z66)
    return (z67, {'alpha_global': z62, 'avg_noisy_percent': float((1.0 - np.mean(z51)) * 100.0)})

def z1e(gray_u8: np.ndarray, sigma_255: float, profile='np', tile_rows: int=0):
    p = C11(profile, sigma_255)
    z68 = z1(gray_u8).astype(np.float32)
    H, W = z68.shape
    lc01 = np.zeros((H, W, 3), dtype=np.float32)
    lc01[..., 0] = z68
    lc01[..., 1] = 0.5
    lc01[..., 2] = 0.5
    sigma_lc01 = np.array([p.sigma01, 0.0, 0.0], dtype=np.float32)
    z69 = z1b(lc01, p, sigma_lc01, tile_rows=tile_rows)
    z6a = z1c(lc01, z69, p, sigma_lc01, tile_rows=tile_rows)
    z6b = z2(z6a[..., 0])
    z6c = {'luma_only': True}
    return (z6b, z6c)

def z1f(img_bgr_u8: np.ndarray, sigma_255: float, profile='np', colorspace='opp', tile_rows: int=0, use_softblend: bool=True):
    p = C11(profile, sigma_255)
    lc01, A, maxV, minV, z2c = z9(img_bgr_u8, colorspace)
    sigma_lc01 = p.sigma01 * z2c
    z6d = time.time()
    z69 = z1b(lc01, p, sigma_lc01, tile_rows=tile_rows)
    z6e = time.time()
    z6a = z1c(lc01, z69, p, sigma_lc01, tile_rows=tile_rows)
    z6f = time.time()
    z70 = za(z6a, A, maxV, minV)
    z6c = {'t_step1': z6e - z6d, 't_step2': z6f - z6e, 't_total': z6f - z6d, 'l2normLumChrom': z2c.tolist()}
    if not use_softblend:
        return (z70, None, z6c)
    z67, z8c = z1d(img_bgr_u8, z70, sigma_255)
    z6c.update(z8c)
    return (z70, z67, z6c)
if __name__ == '__main__':
    C71 = 30.0
    C72 = 'np'
    C73 = 'opp'
    C74 = True
    C75 = 0
    C76 = True
    C77 = 0
    C78 = 'out_cbm3d_step1_bior15'
    os.makedirs(C78, exist_ok=True)

    def z24(file_path: str):
        z8d = cv2.imread(file_path, cv2.IMREAD_COLOR)
        if z8d is None:
            print('Görüntü okunamadı:', file_path)
            return
        z8e = z6(z8d)
        z8f = os.path.splitext(os.path.basename(file_path))[0]
        print('\n' + '=' * 80)
        print(f'Seçilen: {file_path}')
        print(f'Ayarlar: sigma={C71}, profile={C72}, colorspace={C73}, synthetic={C74}, TILE_ROWS={C77}, softblend={C76}')
        print('Gerçekten gri mi? (B==G==R):', z8e)
        z90 = C71
        if z8e:
            z9f = cv2.cvtColor(z8d, cv2.COLOR_BGR2GRAY)
            if C74:
                zaf = z1(z9f)
                zb0 = z3(zaf, C71, seed=C75)
                zb1 = z2(zb0)
                cv2.imwrite(os.path.join(C78, f'{z8f}_clean_gray.png'), z9f)
                cv2.imwrite(os.path.join(C78, f'{z8f}_noisy_gray.png'), zb1)
                print(f'Noisy PSNR (gray): {z4(zaf, zb0):.2f} dB')
            else:
                zb1 = z9f
            z6d = time.time()
            zb2, zb3 = z1e(zb1, sigma_255=z90, profile=C72, tile_rows=C77)
            z6e = time.time()
            z6c = {'t_total': float(z6e - z6d), **zb3}
            z67 = None
            if C76:
                noisy_bgr_u8 = cv2.cvtColor(zb1, cv2.COLOR_GRAY2BGR)
                z70 = cv2.cvtColor(zb2, cv2.COLOR_GRAY2BGR)
                z67, z8c = z1d(noisy_bgr_u8, z70, z90)
                z6c.update(z8c)
            za0 = os.path.join(C78, f'{z8f}_den_gray.png')
            cv2.imwrite(za0, zb2)
            print(f'Gri denoised kaydedildi: {za0}')
            if z67 is not None:
                zb4 = cv2.cvtColor(z67, cv2.COLOR_BGR2GRAY)
                zb5 = os.path.join(C78, f'{z8f}_den_softblend_gray.png')
                cv2.imwrite(zb5, zb4)
                print(f'Gri softblend kaydedildi: {zb5}')
            print(f"Süre (GRAY total): {z6c.get('t_total', 0.0):.2f}s")
            if z67 is not None:
                print(f"Global alpha: {z6c.get('alpha_global', 0.0):.4f} | Avg noisy contrib: %{z6c.get('avg_noisy_percent', 0.0):.2f}")
            if C74:
                zaf = z1(z9f)
                zb6 = z1(zb2)
                zb7 = z4(zaf, zb6)
                print(f'PSNR denoised (gray): {zb7:.2f} dB')
                if z67 is not None:
                    zbd = z1(cv2.cvtColor(z67, cv2.COLOR_BGR2GRAY))
                    zbe = z4(zaf, zbd)
                    print(f'PSNR softblend (gray): {zbe:.2f} dB')
            return
        else:
            if C74:
                zb8 = z1(z8d)
                zb9 = z3(zb8, C71, seed=C75)
                zba = z2(zb9)
                cv2.imwrite(os.path.join(C78, f'{z8f}_clean.png'), z2(zb8))
                cv2.imwrite(os.path.join(C78, f'{z8f}_noisy.png'), zba)
                print(f'Noisy PSNR (color): {z4(zb8, zb9):.2f} dB')
            else:
                zba = z8d
            zbb, z67, z6c = z1f(zba, sigma_255=z90, profile=C72, colorspace=C73, tile_rows=C77, use_softblend=C76)
            za0 = os.path.join(C78, f'{z8f}_den.png')
            cv2.imwrite(za0, zbb)
            print(f'Kaydedildi: {za0}')
            if z67 is not None:
                zb5 = os.path.join(C78, f'{z8f}_den_softblend.png')
                cv2.imwrite(zb5, z67)
                print(f'Kaydedildi: {zb5}')
            print(f"Süre: Step1={z6c.get('t_step1', 0.0):.2f}s | Step2={z6c.get('t_step2', 0.0):.2f}s | Total={z6c.get('t_total', 0.0):.2f}s")
            print(f"l2normLumChrom={z6c.get('l2normLumChrom')}")
            if z67 is not None:
                print(f"Global alpha: {z6c.get('alpha_global', 0.0):.4f} | Avg noisy contrib: %{z6c.get('avg_noisy_percent', 0.0):.2f}")
            if C74:
                zbc = z1(zbb)
                zb7 = z4(zb8, zbc)
                print(f'PSNR denoised (color): {zb7:.2f} dB')
                if z67 is not None:
                    zbe = z4(zb8, z1(z67))
                    print(f'PSNR softblend (color): {zbe:.2f} dB')
    z79 = tk.Tk()
    z79.withdraw()
    z7a = filedialog.askopenfilenames(title='Birden fazla resim seç', filetypes=[('Image files', '*.jpg *.jpeg *.png *.bmp *.tiff'), ('All files', '*.*')])
    if not z7a:
        print('Dosya seçilmedi.')
        sys.exit(0)
    for z7b in z7a:
        z24(z7b)
denoise_cbm3d_bior_step1=z1f
denoise_bm3d_gray_only=z1e
u8_to_f01=z1
f01_to_u8=z2
add_awgn_01=z3

