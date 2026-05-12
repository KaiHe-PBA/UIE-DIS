from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def clamp01(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(0.0, 1.0)


def charbonnier_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.mean(torch.sqrt((pred - target) ** 2 + eps ** 2))


def _gaussian_window(window_size: int, sigma: float, channels: int, device: torch.device) -> torch.Tensor:
    coords = torch.arange(window_size, device=device).float() - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).view(1, 1, -1)
    window_2d = (g.transpose(2, 1) @ g).view(1, 1, window_size, window_size)
    return window_2d.repeat(channels, 1, 1, 1)


def ssim_loss(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    b, c, _, _ = pred.shape
    window = _gaussian_window(window_size, sigma, c, pred.device)
    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=c)
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=c)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu12 = mu1 * mu2
    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=c) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size // 2, groups=c) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size // 2, groups=c) - mu12
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim = ((2 * mu12 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-12)
    return 1.0 - ssim.mean()


def sobel_edges(x: torch.Tensor) -> torch.Tensor:
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    c = x.shape[1]
    kx = kx.repeat(c, 1, 1, 1)
    ky = ky.repeat(c, 1, 1, 1)
    gx = F.conv2d(x, kx, padding=1, groups=c)
    gy = F.conv2d(x, ky, padding=1, groups=c)
    return torch.sqrt(gx * gx + gy * gy + 1e-12)


def edge_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(sobel_edges(pred), sobel_edges(target))


def fft_mag(x: torch.Tensor) -> torch.Tensor:
    return torch.abs(torch.fft.rfft2(x, norm='ortho'))


def freq_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(fft_mag(pred), fft_mag(target))


def color_stat_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_mean = pred.mean(dim=[2, 3], keepdim=True)
    target_mean = target.mean(dim=[2, 3], keepdim=True)
    pred_std = pred.std(dim=[2, 3], keepdim=True) + 1e-6
    target_std = target.std(dim=[2, 3], keepdim=True) + 1e-6
    return F.l1_loss(pred_mean, target_mean) + F.l1_loss(pred_std, target_std)


@dataclass
class StudentLossConfig:
    w_recon: float = 1.0
    w_ssim: float = 0.2
    w_color: float = 0.1
    w_edge: float = 0.05
    w_freq: float = 0.05
    use_charbonnier: bool = True


class StudentLoss(nn.Module):
    def __init__(self, cfg: StudentLossConfig):
        super().__init__()
        self.cfg = cfg

    def forward(self, pred: Dict[str, torch.Tensor], target_x0: torch.Tensor) -> Dict[str, torch.Tensor]:
        x0 = pred.get('x0_from_residual', pred['x0'])
        losses = {}
        total = x0.new_zeros([])

        if self.cfg.use_charbonnier:
            l_recon = charbonnier_loss(x0, target_x0)
        else:
            l_recon = F.l1_loss(x0, target_x0)
        losses['loss_recon'] = l_recon
        total = total + self.cfg.w_recon * l_recon

        l_ssim = ssim_loss(clamp01(x0), clamp01(target_x0))
        losses['loss_ssim'] = l_ssim
        total = total + self.cfg.w_ssim * l_ssim

        l_color = color_stat_loss(clamp01(x0), clamp01(target_x0))
        losses['loss_color'] = l_color
        total = total + self.cfg.w_color * l_color

        if self.cfg.w_edge > 0:
            l_edge = edge_loss(clamp01(x0), clamp01(target_x0))
            losses['loss_edge'] = l_edge
            total = total + self.cfg.w_edge * l_edge

        if self.cfg.w_freq > 0:
            l_freq = freq_loss(clamp01(x0), clamp01(target_x0))
            losses['loss_freq'] = l_freq
            total = total + self.cfg.w_freq * l_freq

        losses['loss_total'] = total
        return losses
