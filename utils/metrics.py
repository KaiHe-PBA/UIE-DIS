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


def _to_numpy_image(pred: torch.Tensor) -> np.ndarray:
    return pred.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy().astype(np.float32)


def _trimmed_mean_std(values: np.ndarray, trim_ratio: float = 0.1) -> tuple[float, float]:
    flat = np.sort(values.reshape(-1).astype(np.float64))
    count = flat.size
    if count == 0:
        return 0.0, 0.0
    trim = int(math.floor(trim_ratio * count))
    if 2 * trim >= count:
        trimmed = flat
    else:
        trimmed = flat[trim: count - trim]
    return float(np.mean(trimmed)), float(np.std(trimmed))


def _sobel_magnitude(channel: np.ndarray) -> np.ndarray:
    if min(channel.shape) <= 1:
        return np.zeros_like(channel, dtype=np.float32)
    padded = np.pad(channel, ((1, 1), (1, 1)), mode="reflect")
    gx = (
        padded[:-2, 2:] + 2.0 * padded[1:-1, 2:] + padded[2:, 2:]
        - padded[:-2, :-2] - 2.0 * padded[1:-1, :-2] - padded[2:, :-2]
    )
    gy = (
        padded[2:, :-2] + 2.0 * padded[2:, 1:-1] + padded[2:, 2:]
        - padded[:-2, :-2] - 2.0 * padded[:-2, 1:-1] - padded[:-2, 2:]
    )
    return np.sqrt(gx * gx + gy * gy + 1e-12).astype(np.float32)


def _eme(channel: np.ndarray, window_size: int = 8, eps: float = 1e-8) -> float:
    height, width = channel.shape
    if height == 0 or width == 0:
        return 0.0
    window_size = max(1, min(window_size, height, width))
    score = 0.0
    block_count = 0
    for top in range(0, height, window_size):
        for left in range(0, width, window_size):
            block = channel[top:min(top + window_size, height), left:min(left + window_size, width)]
            if block.size == 0:
                continue
            block_max = float(np.max(block))
            block_min = float(np.min(block))
            if block_max <= eps or block_min <= eps or block_max <= block_min:
                continue
            score += math.log(block_max / block_min)
            block_count += 1
    if block_count == 0:
        return 0.0
    return float(2.0 * score / block_count)


def _uiconm(gray: np.ndarray, window_size: int = 10, eps: float = 1e-8) -> float:
    height, width = gray.shape
    if height == 0 or width == 0:
        return 0.0
    window_size = max(1, min(window_size, height, width))
    score = 0.0
    block_count = 0
    for top in range(0, height, window_size):
        for left in range(0, width, window_size):
            block = gray[top:min(top + window_size, height), left:min(left + window_size, width)]
            if block.size == 0:
                continue
            block_max = float(np.max(block))
            block_min = float(np.min(block))
            contrast = (block_max - block_min) / (block_max + block_min + eps)
            if contrast <= eps:
                continue
            score += contrast * math.log(contrast + eps)
            block_count += 1
    if block_count == 0:
        return 0.0
    return float(-score / block_count)


def _uicm(image: np.ndarray) -> float:
    r = image[..., 0]
    g = image[..., 1]
    b = image[..., 2]
    rg = r - g
    yb = 0.5 * (r + g) - b
    mu_rg, sigma_rg = _trimmed_mean_std(rg)
    mu_yb, sigma_yb = _trimmed_mean_std(yb)
    return float(-0.0268 * mu_rg + 0.1586 * sigma_rg - 0.0780 * mu_yb + 0.1528 * sigma_yb)


def _uism(image: np.ndarray) -> float:
    weights = (0.299, 0.587, 0.114)
    score = 0.0
    for idx, weight in enumerate(weights):
        channel = image[..., idx]
        edge_map = _sobel_magnitude(channel) * channel
        score += weight * _eme(edge_map)
    return float(score)


def compute_uciqe(pred: torch.Tensor) -> float:
    image = _to_numpy_image(pred)
    lab = rgb_to_lab_np(image)
    luminance = np.clip(lab[..., 0] / 100.0, 0.0, 1.0)
    a = lab[..., 1] / 127.0
    b = lab[..., 2] / 127.0
    chroma = np.sqrt(a * a + b * b + 1e-12)
    mean_chroma = float(np.mean(chroma))
    sigma_c = float(np.sqrt(np.mean(np.abs(1.0 - np.square(mean_chroma / (chroma + 1e-12))))))
    saturation = chroma / np.sqrt(chroma * chroma + luminance * luminance + 1e-12)
    mean_saturation = float(np.mean(saturation))

    histogram, _ = np.histogram(luminance, bins=256, range=(0.0, 1.0))
    cdf = np.cumsum(histogram).astype(np.float64)
    cdf = cdf / max(cdf[-1], 1.0)
    low_idx = int(np.searchsorted(cdf, 0.01))
    high_idx = int(np.searchsorted(cdf, 0.99))
    contrast_luminance = max((high_idx - low_idx) / 255.0, 0.0)
    return float(0.4680 * sigma_c + 0.2745 * contrast_luminance + 0.2576 * mean_saturation)


def compute_uiqm(pred: torch.Tensor) -> float:
    image = _to_numpy_image(pred)
    gray = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
    uicm = _uicm(image)
    uism = _uism(image)
    uiconm = _uiconm(gray)
    return float(0.0282 * uicm + 0.2953 * uism + 3.5753 * uiconm)


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
