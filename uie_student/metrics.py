from typing import Dict

import torch
import torch.nn.functional as F

from .losses import clamp01


def _gaussian_window(window_size: int, sigma: float, channels: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).view(1, 1, -1)
    window_2d = (g.transpose(2, 1) @ g).view(1, 1, window_size, window_size)
    return window_2d.repeat(channels, 1, 1, 1)


def calc_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    pred = clamp01(pred)
    target = clamp01(target)
    mse = F.mse_loss(pred, target, reduction="none")
    mse = mse.mean(dim=(1, 2, 3))
    psnr = 10.0 * torch.log10((max_val ** 2) / torch.clamp(mse, min=1e-12))
    return psnr


def calc_ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    pred = clamp01(pred)
    target = clamp01(target)
    channels = pred.shape[1]
    window = _gaussian_window(window_size, sigma, channels, pred.device, pred.dtype)

    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=channels)
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=channels)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu12 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size // 2, groups=channels) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size // 2, groups=channels) - mu12

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim_map = ((2 * mu12 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-12
    )
    return ssim_map.mean(dim=(1, 2, 3))


def calc_batch_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {
        "psnr": calc_psnr(pred, target),
        "ssim": calc_ssim(pred, target),
    }
