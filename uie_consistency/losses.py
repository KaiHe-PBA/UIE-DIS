from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


TensorOrDict = Mapping[str, Tensor] | Dict[str, Tensor]


def _reduce(loss: Tensor, reduction: str = "mean") -> Tensor:
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError(f"Unsupported reduction: {reduction}")


def _to_4d_time(t: Tensor | float | int, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    if not torch.is_tensor(t):
        t = torch.tensor([t], device=device, dtype=dtype)
    t = t.to(device=device, dtype=dtype)
    if t.ndim == 0:
        t = t.repeat(batch_size)
    if t.ndim == 1 and t.shape[0] == 1:
        t = t.repeat(batch_size)
    if t.ndim != 1 or t.shape[0] != batch_size:
        raise ValueError(f"t must be scalar or shape [B], got {tuple(t.shape)}")
    return t.view(batch_size, 1, 1, 1)


def resolve_x0_prediction(output: TensorOrDict, x_t: Optional[Tensor] = None) -> Tensor:
    if "x0_pred" in output:
        return output["x0_pred"]
    if "residual_pred" in output and x_t is not None:
        return x_t + output["residual_pred"]
    raise KeyError("Model output must contain `x0_pred`, or `residual_pred` together with `x_t`.")


def clamp_image_prediction(x: Tensor, clamp_range: tuple[float, float] = (0.0, 1.0)) -> Tensor:
    return x.clamp(*clamp_range)


def l1_loss(pred: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    return _reduce((pred - target).abs(), reduction=reduction)


def charbonnier_loss(pred: Tensor, target: Tensor, epsilon: float = 1e-3, reduction: str = "mean") -> Tensor:
    return _reduce(torch.sqrt((pred - target) ** 2 + epsilon**2), reduction=reduction)


def _gaussian_kernel(window_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2.0
    kernel_1d = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    return kernel_2d


def ssim(
    pred: Tensor,
    target: Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
    k1: float = 0.01,
    k2: float = 0.03,
) -> Tensor:
    if pred.shape != target.shape:
        raise ValueError("pred and target must have the same shape for SSIM.")
    if pred.ndim != 4:
        raise ValueError("SSIM expects 4D tensors shaped as [B, C, H, W].")
    channels = pred.shape[1]
    kernel_2d = _gaussian_kernel(window_size, sigma, pred.device, pred.dtype)
    window = kernel_2d.expand(channels, 1, window_size, window_size)
    padding = window_size // 2

    mu_pred = F.conv2d(pred, window, padding=padding, groups=channels)
    mu_target = F.conv2d(target, window, padding=padding, groups=channels)

    mu_pred_sq = mu_pred.pow(2)
    mu_target_sq = mu_target.pow(2)
    mu_pred_target = mu_pred * mu_target

    sigma_pred_sq = F.conv2d(pred * pred, window, padding=padding, groups=channels) - mu_pred_sq
    sigma_target_sq = F.conv2d(target * target, window, padding=padding, groups=channels) - mu_target_sq
    sigma_pred_target = F.conv2d(pred * target, window, padding=padding, groups=channels) - mu_pred_target

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    numerator = (2 * mu_pred_target + c1) * (2 * sigma_pred_target + c2)
    denominator = (mu_pred_sq + mu_target_sq + c1) * (sigma_pred_sq + sigma_target_sq + c2)
    return (numerator / (denominator + 1e-12)).mean(dim=(1, 2, 3))


def ssim_loss(pred: Tensor, target: Tensor, **kwargs: Any) -> Tensor:
    return 1.0 - ssim(pred, target, **kwargs).mean()


class ConsistencyLoss(nn.Module):
    def __init__(self, mode: str = "charbonnier", epsilon: float = 1e-3, residual_cycle_weight: float = 0.0):
        super().__init__()
        self.mode = mode
        self.epsilon = epsilon
        self.residual_cycle_weight = residual_cycle_weight

    def _base_loss(self, pred: Tensor, target: Tensor) -> Tensor:
        if self.mode == "charbonnier":
            return charbonnier_loss(pred, target, epsilon=self.epsilon)
        if self.mode == "l1":
            return l1_loss(pred, target)
        raise ValueError(f"Unsupported consistency mode: {self.mode}")

    def forward(
        self,
        x0_pred: Tensor,
        reference_x0: Tensor,
        output: Optional[TensorOrDict] = None,
        x_t: Optional[Tensor] = None,
    ) -> Tensor:
        loss = self._base_loss(x0_pred, reference_x0)
        if self.residual_cycle_weight > 0.0 and output is not None and x_t is not None and "residual_pred" in output:
            cycle_target = x_t + output["residual_pred"]
            loss = loss + self.residual_cycle_weight * self._base_loss(x0_pred, cycle_target)
        return loss


class LightweightPerceptualFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor(
            [[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]], dtype=torch.float32
        )
        sobel_y = torch.tensor(
            [[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]], dtype=torch.float32
        )
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3), persistent=False)
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3), persistent=False)

    def _gradient_magnitude(self, x: Tensor) -> Tensor:
        gray = x.mean(dim=1, keepdim=True)
        grad_x = F.conv2d(gray, self.sobel_x.to(dtype=x.dtype), padding=1)
        grad_y = F.conv2d(gray, self.sobel_y.to(dtype=x.dtype), padding=1)
        return torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-6)

    def forward(self, x: Tensor) -> list[Tensor]:
        feats = [x]
        for scale in (2, 4):
            pooled = F.avg_pool2d(x, kernel_size=scale, stride=scale, ceil_mode=False)
            feats.append(pooled)
            feats.append(self._gradient_magnitude(pooled))
        feats.append(self._gradient_magnitude(x))
        return feats


class TorchvisionPerceptualFeatures(nn.Module):
    def __init__(self, layer_weights: Optional[dict[str, float]] = None, use_input_norm: bool = True):
        super().__init__()
        self.layer_weights = layer_weights or {"3": 1.0, "8": 1.0, "15": 1.0}
        self.use_input_norm = use_input_norm

        from torchvision.models import VGG16_Weights, vgg16

        try:
            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        except Exception:
            vgg = vgg16(weights=None).features

        for parameter in vgg.parameters():
            parameter.requires_grad_(False)
        self.features = vgg.eval()

        if self.use_input_norm:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            self.register_buffer("mean", mean, persistent=False)
            self.register_buffer("std", std, persistent=False)

    def forward(self, x: Tensor) -> list[Tensor]:
        if x.shape[1] != 3:
            raise ValueError("TorchvisionPerceptualFeatures expects RGB tensors with 3 channels.")
        if self.use_input_norm:
            x = (x - self.mean.to(dtype=x.dtype)) / self.std.to(dtype=x.dtype)
        outputs = []
        for index, layer in enumerate(self.features):
            x = layer(x)
            key = str(index)
            if key in self.layer_weights:
                outputs.append(x)
        return outputs


class PerceptualLoss(nn.Module):
    def __init__(self, backend: str = "auto"):
        super().__init__()
        self.backend = backend
        self.features, self.active_backend = self._build_backend(backend)

    def _build_backend(self, backend: str) -> tuple[nn.Module, str]:
        if backend in {"auto", "torchvision_vgg16"}:
            try:
                return TorchvisionPerceptualFeatures(), "torchvision_vgg16"
            except Exception:
                if backend == "torchvision_vgg16":
                    raise
        return LightweightPerceptualFeatures(), "lightweight"

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        pred_features = self.features(pred)
        target_features = self.features(target)
        total = pred.new_zeros(())
        for pred_feat, target_feat in zip(pred_features, target_features):
            total = total + charbonnier_loss(pred_feat, target_feat)
        return total / max(len(pred_features), 1)


class ColorCorrectionLoss(nn.Module):
    def __init__(self, mean_weight: float = 1.0, std_weight: float = 0.5, color_map_weight: float = 0.05):
        super().__init__()
        self.mean_weight = mean_weight
        self.std_weight = std_weight
        self.color_map_weight = color_map_weight

    def forward(self, x0_pred: Tensor, target_x0: Tensor, color_map: Optional[Tensor] = None) -> Tensor:
        pred_mean = x0_pred.mean(dim=(2, 3))
        target_mean = target_x0.mean(dim=(2, 3))
        pred_std = x0_pred.std(dim=(2, 3), unbiased=False)
        target_std = target_x0.std(dim=(2, 3), unbiased=False)

        loss = self.mean_weight * charbonnier_loss(pred_mean, target_mean)
        loss = loss + self.std_weight * charbonnier_loss(pred_std, target_std)

        if color_map is not None:
            smooth_h = (color_map[:, :, 1:, :] - color_map[:, :, :-1, :]).abs().mean()
            smooth_w = (color_map[:, :, :, 1:] - color_map[:, :, :, :-1]).abs().mean()
            identity_reg = (color_map - 1.0).abs().mean()
            loss = loss + self.color_map_weight * (smooth_h + smooth_w + identity_reg)
        return loss


class FrequencyLoss(nn.Module):
    def __init__(self, use_log_magnitude: bool = True):
        super().__init__()
        self.use_log_magnitude = use_log_magnitude

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        pred_fft = torch.fft.rfft2(pred, norm="ortho")
        target_fft = torch.fft.rfft2(target, norm="ortho")
        pred_mag = pred_fft.abs()
        target_mag = target_fft.abs()
        if self.use_log_magnitude:
            pred_mag = torch.log1p(pred_mag)
            target_mag = torch.log1p(target_mag)
        return charbonnier_loss(pred_mag, target_mag)


class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor(
            [[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]], dtype=torch.float32
        )
        sobel_y = torch.tensor(
            [[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]], dtype=torch.float32
        )
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3), persistent=False)
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3), persistent=False)

    def _edges(self, x: Tensor) -> Tensor:
        gray = x.mean(dim=1, keepdim=True)
        grad_x = F.conv2d(gray, self.sobel_x.to(dtype=x.dtype), padding=1)
        grad_y = F.conv2d(gray, self.sobel_y.to(dtype=x.dtype), padding=1)
        return torch.cat([grad_x, grad_y], dim=1)

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return charbonnier_loss(self._edges(pred), self._edges(target))


class MeasurementConsistencyLoss(nn.Module):
    def __init__(self, operator: Callable[[Tensor], Tensor]):
        super().__init__()
        self.operator = operator

    def forward(self, x0_pred: Tensor, measurement: Tensor) -> Tensor:
        return charbonnier_loss(self.operator(x0_pred), measurement)


class DegradationConsistencyLoss(nn.Module):
    def __init__(self, operator: Callable[[Tensor], Tensor]):
        super().__init__()
        self.operator = operator

    def forward(self, x0_pred: Tensor, degraded_target: Tensor) -> Tensor:
        degraded_pred = self.operator(x0_pred)
        return charbonnier_loss(degraded_pred, degraded_target)


@dataclass
class LossWeights:
    consistency: float = 1.0
    charbonnier: float = 1.0
    l1: float = 0.0
    ssim: float = 0.5
    perceptual: float = 0.1
    color: float = 0.2
    frequency: float = 0.0
    edge: float = 0.0
    measurement: float = 0.0
    degradation: float = 0.0


class UnderwaterLossAssembler(nn.Module):
    def __init__(
        self,
        weights: Optional[LossWeights] = None,
        perceptual_backend: str = "auto",
        measurement_operator: Optional[Callable[[Tensor], Tensor]] = None,
        degradation_operator: Optional[Callable[[Tensor], Tensor]] = None,
        clamp_range: tuple[float, float] = (0.0, 1.0),
    ):
        super().__init__()
        self.weights = weights or LossWeights()
        self.clamp_range = clamp_range
        self.consistency_loss = ConsistencyLoss()
        self.perceptual_loss = PerceptualLoss(backend=perceptual_backend)
        self.color_loss = ColorCorrectionLoss()
        self.frequency_loss = FrequencyLoss()
        self.edge_loss = EdgeLoss()
        self.measurement_loss = (
            MeasurementConsistencyLoss(measurement_operator) if measurement_operator is not None else None
        )
        self.degradation_loss = (
            DegradationConsistencyLoss(degradation_operator) if degradation_operator is not None else None
        )

    def forward(
        self,
        output: TensorOrDict,
        *,
        target_x0: Optional[Tensor] = None,
        teacher_output: Optional[TensorOrDict] = None,
        teacher_x0: Optional[Tensor] = None,
        x_t: Optional[Tensor] = None,
        y: Optional[Tensor] = None,
        t: Optional[Tensor | float | int] = None,
        measurement: Optional[Tensor] = None,
        degraded_target: Optional[Tensor] = None,
        measurement_operator: Optional[Callable[[Tensor], Tensor]] = None,
        degradation_operator: Optional[Callable[[Tensor], Tensor]] = None,
    ) -> Dict[str, Tensor]:
        x0_pred_raw = resolve_x0_prediction(output, x_t=x_t)
        x0_pred = clamp_image_prediction(x0_pred_raw, self.clamp_range)
        total = x0_pred.new_zeros(())
        terms: Dict[str, Tensor] = {}

        if teacher_x0 is None and teacher_output is not None:
            teacher_x0 = resolve_x0_prediction(teacher_output, x_t=x_t)
        if teacher_x0 is not None:
            teacher_x0 = clamp_image_prediction(teacher_x0, self.clamp_range)
        if teacher_x0 is not None and self.weights.consistency > 0:
            consistency = self.consistency_loss(x0_pred, teacher_x0, output=output, x_t=x_t)
            total = total + self.weights.consistency * consistency
            terms["consistency"] = consistency

        if target_x0 is not None:
            if self.weights.charbonnier > 0:
                charbonnier = charbonnier_loss(x0_pred, target_x0)
                total = total + self.weights.charbonnier * charbonnier
                terms["charbonnier"] = charbonnier
            if self.weights.l1 > 0:
                l1_value = l1_loss(x0_pred, target_x0)
                total = total + self.weights.l1 * l1_value
                terms["l1"] = l1_value
            if self.weights.ssim > 0:
                ssim_value = ssim_loss(x0_pred, target_x0)
                total = total + self.weights.ssim * ssim_value
                terms["ssim"] = ssim_value
            if self.weights.perceptual > 0:
                perceptual = self.perceptual_loss(x0_pred, target_x0)
                total = total + self.weights.perceptual * perceptual
                terms["perceptual"] = perceptual
            if self.weights.color > 0:
                color_value = self.color_loss(x0_pred, target_x0, color_map=output.get("color_map"))
                total = total + self.weights.color * color_value
                terms["color"] = color_value
            if self.weights.frequency > 0:
                frequency = self.frequency_loss(x0_pred, target_x0)
                total = total + self.weights.frequency * frequency
                terms["frequency"] = frequency
            if self.weights.edge > 0:
                edge = self.edge_loss(x0_pred, target_x0)
                total = total + self.weights.edge * edge
                terms["edge"] = edge

        active_measurement_loss = self.measurement_loss
        if measurement_operator is not None:
            active_measurement_loss = MeasurementConsistencyLoss(measurement_operator)
        if measurement is not None and active_measurement_loss is not None and self.weights.measurement > 0:
            measurement_value = active_measurement_loss(x0_pred, measurement)
            total = total + self.weights.measurement * measurement_value
            terms["measurement"] = measurement_value

        active_degradation_loss = self.degradation_loss
        if degradation_operator is not None:
            active_degradation_loss = DegradationConsistencyLoss(degradation_operator)
        if degraded_target is None and y is not None:
            degraded_target = y
        if degraded_target is not None and active_degradation_loss is not None and self.weights.degradation > 0:
            degradation_value = active_degradation_loss(x0_pred, degraded_target)
            total = total + self.weights.degradation * degradation_value
            terms["degradation"] = degradation_value

        if t is not None:
            time_map = _to_4d_time(t, x0_pred.shape[0], x0_pred.device, x0_pred.dtype)
            terms["time_mean"] = time_map.mean()

        terms["total"] = total
        return terms
