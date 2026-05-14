from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_kernel(window_size: int, sigma: float, channels: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel = kernel_2d.expand(channels, 1, window_size, window_size).contiguous()
    return kernel


def ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    channels = pred.shape[1]
    kernel = _gaussian_kernel(window_size, sigma, channels, pred.device, pred.dtype)
    padding = window_size // 2

    mu_pred = F.conv2d(pred, kernel, padding=padding, groups=channels)
    mu_target = F.conv2d(target, kernel, padding=padding, groups=channels)

    mu_pred_sq = mu_pred.pow(2)
    mu_target_sq = mu_target.pow(2)
    mu_pred_target = mu_pred * mu_target

    sigma_pred = F.conv2d(pred * pred, kernel, padding=padding, groups=channels) - mu_pred_sq
    sigma_target = F.conv2d(target * target, kernel, padding=padding, groups=channels) - mu_target_sq
    sigma_cross = F.conv2d(pred * target, kernel, padding=padding, groups=channels) - mu_pred_target

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    numerator = (2 * mu_pred_target + c1) * (2 * sigma_cross + c2)
    denominator = (mu_pred_sq + mu_target_sq + c1) * (sigma_pred + sigma_target + c2)
    score = numerator / (denominator + 1e-8)
    return score.mean()


class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-3) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sqrt((pred - target) ** 2 + self.eps ** 2).mean()


class ConsistencyLoss(nn.Module):
    def __init__(self, mode: str = 'charbonnier') -> None:
        super().__init__()
        self.mode = mode
        self.charbonnier = CharbonnierLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.mode == 'l2':
            return F.mse_loss(pred, target)
        if self.mode == 'l1':
            return F.l1_loss(pred, target)
        return self.charbonnier(pred, target)


class SSIMLoss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 1.0 - ssim(pred, target)


class ColorConsistencyLoss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_mean = pred.mean(dim=(2, 3))
        target_mean = target.mean(dim=(2, 3))
        pred_std = pred.std(dim=(2, 3), unbiased=False)
        target_std = target.std(dim=(2, 3), unbiased=False)
        mean_loss = F.l1_loss(pred_mean, target_mean)
        std_loss = F.l1_loss(pred_std, target_std)
        channel_diff = F.l1_loss(pred_mean[:, 0] - pred_mean[:, 1], target_mean[:, 0] - target_mean[:, 1])
        channel_diff += F.l1_loss(pred_mean[:, 1] - pred_mean[:, 2], target_mean[:, 1] - target_mean[:, 2])
        return mean_loss + 0.5 * std_loss + 0.25 * channel_diff


class EdgeLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        sobel_x = torch.tensor([
            [-1.0, 0.0, 1.0],
            [-2.0, 0.0, 2.0],
            [-1.0, 0.0, 1.0],
        ])
        sobel_y = torch.tensor([
            [-1.0, -2.0, -1.0],
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 1.0],
        ])
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))

    def _gradient_magnitude(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.mean(dim=1, keepdim=True)
        grad_x = F.conv2d(tensor, self.sobel_x, padding=1)
        grad_y = F.conv2d(tensor, self.sobel_y, padding=1)
        return torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(self._gradient_magnitude(pred), self._gradient_magnitude(target))


class FrequencyLoss(nn.Module):
    def __init__(self, use_phase: bool = False) -> None:
        super().__init__()
        self.use_phase = use_phase

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_fft = torch.fft.rfft2(pred, norm='ortho')
        target_fft = torch.fft.rfft2(target, norm='ortho')
        loss = F.l1_loss(torch.abs(pred_fft), torch.abs(target_fft))
        if self.use_phase:
            loss = loss + 0.1 * F.l1_loss(torch.angle(pred_fft), torch.angle(target_fft))
        return loss


class PerceptualLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature_extractor = None
        self.available = False
        try:
            from torchvision.models import VGG19_Weights, vgg19

            backbone = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:16].eval()
            for param in backbone.parameters():
                param.requires_grad = False
            self.feature_extractor = backbone
            self.available = True
        except Exception:
            self.feature_extractor = None
            self.available = False

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not self.available or self.feature_extractor is None:
            return pred.new_tensor(0.0)
        pred_norm = (pred - self.mean) / self.std
        target_norm = (target - self.mean) / self.std
        pred_feat = self.feature_extractor(pred_norm)
        target_feat = self.feature_extractor(target_norm)
        return F.l1_loss(pred_feat, target_feat)


class MultiTermLoss(nn.Module):
    def __init__(
        self,
        consistency_weight: float = 1.0,
        reconstruction_weight: float = 1.0,
        ssim_weight: float = 0.2,
        color_weight: float = 0.1,
        edge_weight: float = 0.1,
        frequency_weight: float = 0.05,
        perceptual_weight: float = 0.0,
        reconstruction_mode: str = 'charbonnier',
        consistency_mode: str = 'charbonnier',
    ) -> None:
        super().__init__()
        self.consistency_weight = consistency_weight
        self.reconstruction_weight = reconstruction_weight
        self.ssim_weight = ssim_weight
        self.color_weight = color_weight
        self.edge_weight = edge_weight
        self.frequency_weight = frequency_weight
        self.perceptual_weight = perceptual_weight

        self.consistency = ConsistencyLoss(mode=consistency_mode)
        self.reconstruction = CharbonnierLoss() if reconstruction_mode == 'charbonnier' else nn.L1Loss()
        self.ssim_loss = SSIMLoss()
        self.color = ColorConsistencyLoss()
        self.edge = EdgeLoss()
        self.frequency = FrequencyLoss()
        self.perceptual = PerceptualLoss() if perceptual_weight > 0 else None

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        consistency_target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        losses: Dict[str, torch.Tensor] = {}
        total = pred.new_tensor(0.0)

        if self.reconstruction_weight > 0:
            losses['reconstruction'] = self.reconstruction(pred, target)
            total = total + self.reconstruction_weight * losses['reconstruction']
        if self.ssim_weight > 0:
            losses['ssim'] = self.ssim_loss(pred, target)
            total = total + self.ssim_weight * losses['ssim']
        if self.color_weight > 0:
            losses['color'] = self.color(pred, target)
            total = total + self.color_weight * losses['color']
        if self.edge_weight > 0:
            losses['edge'] = self.edge(pred, target)
            total = total + self.edge_weight * losses['edge']
        if self.frequency_weight > 0:
            losses['frequency'] = self.frequency(pred, target)
            total = total + self.frequency_weight * losses['frequency']
        if self.perceptual_weight > 0:
            if self.perceptual is None:
                raise RuntimeError('perceptual_weight > 0 但感知损失模块未初始化')
            losses['perceptual'] = self.perceptual(pred, target)
            total = total + self.perceptual_weight * losses['perceptual']
        if self.consistency_weight > 0 and consistency_target is not None:
            losses['consistency'] = self.consistency(pred, consistency_target)
            total = total + self.consistency_weight * losses['consistency']

        detached = {name: value.detach() for name, value in losses.items()}
        detached['total'] = total.detach()
        return total, detached
