from __future__ import annotations
import os
import cv2
import numpy as np
import torch
from bm3d_core import denoise_cbm3d_bior_step1, denoise_bm3d_gray_only, u8_to_f01, f01_to_u8, add_awgn_01
from model_unet import UNetResidual
from utils_img import u8_hwc_to_t01_chw, t01_chw_to_u8_hwc, make_sigma_map
import tkinter as tk
from tkinter import filedialog
import torch.nn.functional as F

def z0(x: torch.Tensor, mult: int=16):
    _, _, C24, C25 = x.shape
    za = (mult - C24 % mult) % mult
    zb = (mult - C25 % mult) % mult
    zc = F.pad(x, (0, zb, 0, za), mode='reflect')
    return (zc, (C24, C25))

def z1(x: torch.Tensor, orig_hw):
    C24, C25 = orig_hw
    return x[:, :, :C24, :C25]

def z2(img_bgr_u8: np.ndarray) -> bool:
    z26, z27, z28 = cv2.split(img_bgr_u8)
    return np.array_equal(z26, z27) and np.array_equal(z27, z28)

def z3(noisy_u8: np.ndarray, sigma_255: float, profile: str='np', colorspace: str='opp', tile_rows: int=0, use_softblend: bool=False) -> np.ndarray:
    zd = z2(noisy_u8)
    if zd:
        z29 = cv2.cvtColor(noisy_u8, cv2.COLOR_BGR2GRAY)
        z2f, _info = denoise_bm3d_gray_only(z29, sigma_255=float(sigma_255), profile=profile, tile_rows=tile_rows)
        bm3d_u8 = cv2.cvtColor(z2f, cv2.COLOR_GRAY2BGR)
        return bm3d_u8
    else:
        bm3d_u8, _mix_u8, _info = denoise_cbm3d_bior_step1(noisy_u8, sigma_255=float(sigma_255), profile=profile, colorspace=colorspace, tile_rows=tile_rows, use_softblend=use_softblend)
        return bm3d_u8

def z4(ref_u8: np.ndarray, est_u8: np.ndarray) -> float:
    ze = ref_u8.astype(np.float32)
    zf = est_u8.astype(np.float32)
    z10 = np.mean((ze - zf) ** 2)
    if z10 == 0:
        return 99.0
    return 10.0 * np.log10(255.0 ** 2 / z10)

@torch.no_grad()
def z5(noisy_u8: np.ndarray, bm3d_u8: np.ndarray, sigma_255: float, ckpt_path: str, device: str=None) -> np.ndarray:
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    z11 = u8_hwc_to_t01_chw(noisy_u8)
    z12 = u8_hwc_to_t01_chw(bm3d_u8)
    C24, C25 = (z11.shape[1], z11.shape[2])
    z13 = make_sigma_map(float(sigma_255), C24, C25)
    z14 = torch.cat([z11, z12, z11 - z12, z13], dim=0).unsqueeze(0).to(device)
    z2a, orig_hw = z0(z14, mult=16)
    z15 = UNetResidual(in_ch=10, out_ch=3, base=32).to(device)
    z16 = torch.load(ckpt_path, map_location=device)
    if isinstance(z16, dict) and 'state_dict' in z16:
        z15.load_state_dict(z16['state_dict'], strict=True)
    else:
        z15.load_state_dict(z16, strict=True)
    z15.eval()
    z17 = z15(z2a)[0].detach().cpu()
    z18 = z12.unsqueeze(0)
    z19, _ = z0(z18, mult=16)
    z19 = z19[0]
    z1a = (z19 + z17).clamp(0.0, 1.0)
    z1b = z1(z1a.unsqueeze(0), orig_hw)[0]
    z1c = t01_chw_to_u8_hwc(z1b)
    return z1c

def z6() -> list[str]:
    z1d = tk.Tk()
    z1d.withdraw()
    z1d.attributes('-topmost', True)
    z1e = filedialog.askopenfilenames(title='Görüntü(leri) seç', filetypes=[('Image files', '*.png *.jpg *.jpeg *.bmp *.tif *.tiff'), ('All files', '*.*')])
    z1d.destroy()
    return list(z1e)

def z7(path: str):
    z1f = np.fromfile(path, dtype=np.uint8)
    z20 = cv2.imdecode(z1f, cv2.IMREAD_COLOR)
    return z20

def z8(img_path: str, ckpt_path: str, input_is_noisy: bool, sigma_255: float, seed: int, profile: str, colorspace: str, tile_rows: int, use_softblend: bool, out_root: str, device: str, demo_mode: bool=True):
    z20 = z7(img_path)
    if z20 is None:
        print('read fail:', img_path)
        return
    if input_is_noisy:
        noisy_u8 = z20
    else:
        zd = z2(z20)
        if zd:
            z30 = cv2.cvtColor(z20, cv2.COLOR_BGR2GRAY)
            z31 = z30.astype(np.float32) / 255.0
            z32 = add_awgn_01(z31, sigma_255=float(sigma_255), seed=seed)
            z29 = f01_to_u8(z32)
            noisy_u8 = cv2.cvtColor(z29, cv2.COLOR_GRAY2BGR)
        else:
            z33 = u8_to_f01(z20)
            z34 = add_awgn_01(z33, sigma_255=float(sigma_255), seed=seed)
            noisy_u8 = f01_to_u8(z34)
    bm3d_u8 = z3(noisy_u8, sigma_255=float(sigma_255), profile=profile, colorspace=colorspace, tile_rows=tile_rows, use_softblend=use_softblend)
    if demo_mode:
        print('DEMO MODE: UNet disabled; using BM3D output as final refine.png')
        z1c = bm3d_u8.copy()
    else:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f'Checkpoint bulunamadı: {ckpt_path}. Full mode için önce modeli train edip checkpoint dosyasını ekleyin veya demo_mode=True kullanın.')
        z1c = z5(noisy_u8=noisy_u8, bm3d_u8=bm3d_u8, sigma_255=float(sigma_255), ckpt_path=ckpt_path, device=device)
    z21 = os.path.splitext(os.path.basename(img_path))[0]
    z22 = os.path.join(out_root, z21)
    os.makedirs(z22, exist_ok=True)
    if not input_is_noisy:
        z2b = z4(z20, noisy_u8)
        z2c = z4(z20, bm3d_u8)
        z2d = z4(z20, z1c)
        z2e = 'BM3D demo' if demo_mode else 'BM3D+UNet'
        print(f'\n[{z21}] PSNR (dB) | Noisy: {z2b:.2f} | BM3D: {z2c:.2f} | {z2e}: {z2d:.2f}')
        with open(os.path.join(z22, 'metrics.txt'), 'w', encoding='utf-8') as z35:
            z35.write(f"Mode           : {('DEMO_BM3D_ONLY' if demo_mode else 'FULL_BM3D_UNET')}\n")
            z35.write(f'Noisy PSNR     : {z2b:.4f}\n')
            z35.write(f'BM3D PSNR      : {z2c:.4f}\n')
            z35.write(f'{z2e} PSNR : {z2d:.4f}\n')
    else:
        print(f'\n[{z21}] input_is_noisy=True -> PSNR yok (clean referans yok).')
    cv2.imwrite(os.path.join(z22, 'input.png'), z20)
    cv2.imwrite(os.path.join(z22, 'noisy.png'), noisy_u8)
    cv2.imwrite(os.path.join(z22, 'bm3d.png'), bm3d_u8)
    cv2.imwrite(os.path.join(z22, 'refine.png'), z1c)
    print('Saved to:', z22)

def z9():
    demo_mode = True
    ckpt_path = 'weights/bm3d_res_unet_best.pth'
    input_is_noisy = False
    sigma_255 = 50.0
    seed = 0
    profile = 'np'
    colorspace = 'opp'
    tile_rows = 0
    use_softblend = False
    out_root = 'infer_out_batch'
    os.makedirs(out_root, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device:', device)
    z23 = z6()
    if not z23:
        print('Hiç dosya seçilmedi.')
        return
    print(f'{len(z23)} görüntü seçildi.')
    for img_path in z23:
        z8(img_path=img_path, ckpt_path=ckpt_path, input_is_noisy=input_is_noisy, sigma_255=float(sigma_255), seed=int(seed), profile=profile, colorspace=colorspace, tile_rows=int(tile_rows), use_softblend=bool(use_softblend), out_root=out_root, device=device, demo_mode=bool(demo_mode))
    print('\nBitti. Çıktılar:', out_root)
if __name__ == '__main__':
    z9()
