import math
from typing import Dict

import numpy as np
import torch

from losses import ssim


def compute_psnr(pred: torch.Tensor, target: torch.Tensor, max_value: float = 1.0) -> torch.Tensor:
    mse = torch.mean((pred - target) ** 2, dim=(1, 2, 3)).clamp_min(1e-10)
    return 10.0 * torch.log10((max_value ** 2) / mse)


def rgb_to_lab_np(image: np.ndarray) -> np.ndarray:
    image = np.clip(image, 0.0, 1.0)
    mask = image > 0.04045
    linear = np.where(mask, ((image + 0.055) / 1.055) ** 2.4, image / 12.92)
    linear = linear * 100.0

    x = linear[..., 0] * 0.4124 + linear[..., 1] * 0.3576 + linear[..., 2] * 0.1805
    y = linear[..., 0] * 0.2126 + linear[..., 1] * 0.7152 + linear[..., 2] * 0.0722
    z = linear[..., 0] * 0.0193 + linear[..., 1] * 0.1192 + linear[..., 2] * 0.9505

    x = x / 95.047
    y = y / 100.0
    z = z / 108.883

    eps = 216.0 / 24389.0
    kappa = 24389.0 / 27.0

    def f(t: np.ndarray) -> np.ndarray:
        return np.where(t > eps, np.cbrt(t), (kappa * t + 16.0) / 116.0)

    fx, fy, fz = f(x), f(y), f(z)
    l = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return np.stack([l, a, b], axis=-1)


def compute_uciqe(pred: torch.Tensor) -> float:
    image = pred.detach().cpu().permute(1, 2, 0).numpy()
    lab = rgb_to_lab_np(image)
    chroma = np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2)
    sigma_c = float(np.std(chroma))
    con_l = float(np.percentile(lab[..., 0], 99) - np.percentile(lab[..., 0], 1))
    sat = float(np.mean(chroma / np.sqrt(chroma ** 2 + lab[..., 0] ** 2 + 1e-8)))
    return 0.4680 * sigma_c + 0.2745 * con_l + 0.2576 * sat


def compute_uiqm(pred: torch.Tensor) -> float:
    image = pred.detach().cpu().permute(1, 2, 0).numpy()
    r, g, b = image[..., 0], image[..., 1], image[..., 2]
    rg = r - g
    yb = 0.5 * (r + g) - b
    uicm = -0.0268 * np.mean(rg) + 0.1586 * np.std(rg) - 0.0780 * np.mean(yb) + 0.1528 * np.std(yb)

    gray = 0.299 * r + 0.587 * g + 0.114 * b
    contrast = np.percentile(gray, 99) - np.percentile(gray, 1)
    sharpness = np.mean(np.abs(np.diff(gray, axis=0))) + np.mean(np.abs(np.diff(gray, axis=1)))
    return float(0.0282 * uicm + 0.2953 * contrast + 3.5753 * sharpness)


def evaluate_image_pair(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    psnr = float(compute_psnr(pred.unsqueeze(0), target.unsqueeze(0)).mean().item())
    ssim_value = float(ssim(pred.unsqueeze(0), target.unsqueeze(0)).item())
    mae = float(torch.mean(torch.abs(pred - target)).item())
    return {
        'psnr': psnr,
        'ssim': ssim_value,
        'mae': mae,
        'uciqe': compute_uciqe(pred),
        'uiqm': compute_uiqm(pred),
    }


def compute_batch_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    batch_size = pred.shape[0]
    psnr = float(compute_psnr(pred, target).mean().item())
    ssim_value = float(ssim(pred, target).item())
    mae = float(torch.mean(torch.abs(pred - target)).item())
    uciqe = sum(compute_uciqe(pred[idx]) for idx in range(batch_size)) / batch_size
    uiqm = sum(compute_uiqm(pred[idx]) for idx in range(batch_size)) / batch_size
    return {
        'psnr': psnr,
        'ssim': ssim_value,
        'mae': mae,
        'uciqe': float(uciqe),
        'uiqm': float(uiqm),
    }
