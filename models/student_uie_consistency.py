from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def count_parameters(module: nn.Module) -> int:
    return sum(param.numel() for param in module.parameters() if param.requires_grad)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int, max_period: int = 10000) -> None:
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.dim() == 0:
            timesteps = timesteps[None]
        timesteps = timesteps.float().view(-1)
        half_dim = self.dim // 2
        exponent = -math.log(self.max_period) * torch.arange(
            half_dim, device=timesteps.device, dtype=torch.float32
        )
        exponent = exponent / max(half_dim - 1, 1)
        freqs = torch.exp(exponent)
        args = timesteps[:, None] * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class TimeMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, time_emb: torch.Tensor) -> torch.Tensor:
        return self.net(time_emb)


class ConvGNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 8,
        activation: bool = True,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        group_count = max(1, min(groups, out_channels))
        while out_channels % group_count != 0 and group_count > 1:
            group_count -= 1
        layers: List[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.GroupNorm(group_count, out_channels),
        ]
        if activation:
            layers.append(nn.SiLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class TimeConditionedResidualBlock(nn.Module):
    def __init__(self, channels: int, time_dim: int, expansion: float = 2.0) -> None:
        super().__init__()
        hidden_channels = int(channels * expansion)
        self.norm1 = nn.GroupNorm(max(1, min(8, channels)), channels)
        self.conv1 = nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False)
        self.dwconv = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            padding=1,
            groups=hidden_channels,
            bias=False,
        )
        self.norm2 = nn.GroupNorm(max(1, min(8, hidden_channels)), hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)
        self.time_proj = nn.Linear(time_dim, channels * 2)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        scale, shift = self.time_proj(time_emb).chunk(2, dim=1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]
        residual = self.norm1(x)
        residual = residual * (1.0 + scale) + shift
        residual = F.silu(residual, inplace=True)
        residual = self.conv1(residual)
        residual = self.dwconv(residual)
        residual = self.norm2(residual)
        residual = F.silu(residual, inplace=True)
        residual = self.conv2(residual)
        return x + residual


class ConditionFusionBlock(nn.Module):
    def __init__(self, in_channels: int, cond_channels: int) -> None:
        super().__init__()
        self.cond_proj = nn.Conv2d(cond_channels, in_channels, kernel_size=1, bias=False)
        self.mix = DepthwiseSeparableConv(in_channels)
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if cond.shape[-2:] != x.shape[-2:]:
            cond = F.interpolate(cond, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cond = self.cond_proj(cond)
        gate = self.gate(torch.cat([x, cond], dim=1))
        fused = x + gate * cond
        return fused + self.mix(fused)


class ColorCorrectionHead(nn.Module):
    def __init__(self, in_channels: int, cond_channels: int) -> None:
        super().__init__()
        hidden = max(16, in_channels // 2)
        self.cond_proj = nn.Conv2d(cond_channels, in_channels, kernel_size=1, bias=False)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels * 2, hidden, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 6, kernel_size=1),
        )

    def forward(self, feature: torch.Tensor, cond: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        if cond.shape[-2:] != feature.shape[-2:]:
            cond = F.interpolate(cond, size=feature.shape[-2:], mode="bilinear", align_corners=False)
        cond = self.cond_proj(cond)
        pooled = torch.cat([self.pool(feature), self.pool(cond)], dim=1)
        scale, bias = self.mlp(pooled).chunk(2, dim=1)
        scale = torch.tanh(scale)
        bias = 0.1 * torch.tanh(bias)
        return base * (1.0 + scale) + bias


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = ConvGNAct(in_channels, out_channels, kernel_size=3, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.proj = ConvGNAct(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        return self.proj(x)


class ConditionEncoder(nn.Module):
    def __init__(self, in_channels: int, channels: Sequence[int]) -> None:
        super().__init__()
        if not channels:
            raise ValueError("channels must not be empty")
        self.stem = ConvGNAct(in_channels, channels[0], kernel_size=3)
        self.blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for idx, channel in enumerate(channels):
            self.blocks.append(
                nn.Sequential(
                    ConvGNAct(channel, channel, kernel_size=3),
                    DepthwiseSeparableConv(channel),
                    nn.SiLU(inplace=True),
                )
            )
            if idx < len(channels) - 1:
                self.downsamples.append(DownsampleBlock(channel, channels[idx + 1]))

    def forward(self, y: torch.Tensor) -> List[torch.Tensor]:
        features: List[torch.Tensor] = []
        x = self.stem(y)
        for idx, block in enumerate(self.blocks):
            x = block(x) + x
            features.append(x)
            if idx < len(self.downsamples):
                x = self.downsamples[idx](x)
        return features


class EncoderStage(nn.Module):
    def __init__(self, channels: int, time_dim: int) -> None:
        super().__init__()
        self.block = TimeConditionedResidualBlock(channels, time_dim)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        return self.block(x, time_emb)


class DecoderStage(nn.Module):
    def __init__(self, channels: int, skip_channels: int, time_dim: int) -> None:
        super().__init__()
        self.merge = ConvGNAct(channels + skip_channels, channels, kernel_size=1)
        self.block = TimeConditionedResidualBlock(channels, time_dim)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.merge(torch.cat([x, skip], dim=1))
        return self.block(x, time_emb)


class LightConsistencyStudent(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        cond_in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 32,
        channel_multipliers: Sequence[int] = (1, 2, 4),
        time_embed_dim: int = 128,
        use_residual_output: bool = True,
        clamp_output: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.cond_in_channels = cond_in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.use_residual_output = use_residual_output
        self.clamp_output = clamp_output

        channels = [base_channels * multiplier for multiplier in channel_multipliers]
        self.channels = channels

        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        self.time_mlp = TimeMLP(time_embed_dim, time_embed_dim)

        self.input_proj = ConvGNAct(in_channels, channels[0], kernel_size=3)
        self.condition_encoder = ConditionEncoder(cond_in_channels, channels)

        self.encoder_stages = nn.ModuleList()
        self.encoder_fusions = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        for idx, channel in enumerate(channels):
            self.encoder_stages.append(EncoderStage(channel, time_embed_dim))
            self.encoder_fusions.append(ConditionFusionBlock(channel, channel))
            if idx < len(channels) - 1:
                self.downsamples.append(DownsampleBlock(channel, channels[idx + 1]))

        self.bottleneck = TimeConditionedResidualBlock(channels[-1], time_embed_dim)
        self.bottleneck_fusion = ConditionFusionBlock(channels[-1], channels[-1])

        self.upsamples = nn.ModuleList()
        self.decoder_stages = nn.ModuleList()
        self.decoder_fusions = nn.ModuleList()

        for idx in reversed(range(len(channels) - 1)):
            in_channel = channels[idx + 1]
            out_channel = channels[idx]
            self.upsamples.append(UpsampleBlock(in_channel, out_channel))
            self.decoder_stages.append(DecoderStage(out_channel, out_channel, time_embed_dim))
            self.decoder_fusions.append(ConditionFusionBlock(out_channel, out_channel))

        self.output_head = nn.Sequential(
            ConvGNAct(channels[0], channels[0], kernel_size=3),
            nn.Conv2d(channels[0], out_channels, kernel_size=3, padding=1),
        )
        self.color_head = ColorCorrectionHead(channels[0], channels[0])
        self.size_multiple = 2 ** (len(channels) - 1)

    def _pad_to_multiple(self, tensor: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        _, _, height, width = tensor.shape
        target_height = math.ceil(height / self.size_multiple) * self.size_multiple
        target_width = math.ceil(width / self.size_multiple) * self.size_multiple
        pad_h = target_height - height
        pad_w = target_width - width
        if pad_h == 0 and pad_w == 0:
            return tensor, (height, width)
        pad_mode = "reflect" if height > 1 and width > 1 else "replicate"
        padded = F.pad(tensor, (0, pad_w, 0, pad_h), mode=pad_mode)
        return padded, (height, width)

    def _crop_to_size(self, tensor: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        height, width = size
        return tensor[:, :, :height, :width]

    def _format_timestep(self, t: Union[int, float, torch.Tensor], batch_size: int, device: torch.device) -> torch.Tensor:
        if not torch.is_tensor(t):
            t = torch.tensor([t], device=device, dtype=torch.float32)
        t = t.to(device=device, dtype=torch.float32)
        if t.dim() == 0:
            t = t.repeat(batch_size)
        elif t.numel() == 1 and batch_size > 1:
            t = t.view(1).repeat(batch_size)
        return t.view(batch_size)

    def forward(
        self,
        xt: torch.Tensor,
        y: torch.Tensor,
        t: Union[int, float, torch.Tensor],
        return_dict: bool = True,
        return_aux: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if xt.shape != y.shape:
            raise ValueError(f"xt and y must share shape, got {xt.shape} and {y.shape}")

        xt, original_size = self._pad_to_multiple(xt)
        y, _ = self._pad_to_multiple(y)
        batch_size = xt.shape[0]
        timesteps = self._format_timestep(t, batch_size=batch_size, device=xt.device)
        time_emb = self.time_mlp(self.time_embed(timesteps))

        cond_features = self.condition_encoder(y)

        skips: List[torch.Tensor] = []
        x = self.input_proj(xt)
        for idx, stage in enumerate(self.encoder_stages):
            x = stage(x, time_emb)
            x = self.encoder_fusions[idx](x, cond_features[idx])
            skips.append(x)
            if idx < len(self.downsamples):
                x = self.downsamples[idx](x)

        x = self.bottleneck(x, time_emb)
        x = self.bottleneck_fusion(x, cond_features[-1])

        for idx, upsample in enumerate(self.upsamples):
            x = upsample(x)
            skip = skips[-(idx + 2)]
            x = self.decoder_stages[idx](x, skip, time_emb)
            cond = cond_features[-(idx + 2)]
            x = self.decoder_fusions[idx](x, cond)

        residual = self.output_head(x)
        if self.use_residual_output:
            x0 = y + residual
        else:
            x0 = residual
        x0 = self.color_head(x, cond_features[0], x0)
        residual = self._crop_to_size(residual, original_size)
        x0 = self._crop_to_size(x0, original_size)

        if self.clamp_output:
            x0 = torch.clamp(x0, 0.0, 1.0)

        if not return_dict:
            return x0

        outputs: Dict[str, torch.Tensor] = {"x0": x0}
        if return_aux:
            outputs["residual"] = residual
            outputs["condition"] = cond_features[0]
        return outputs

    @torch.no_grad()
    def one_step(self, xt: torch.Tensor, y: torch.Tensor, t: Union[int, float, torch.Tensor]) -> torch.Tensor:
        return self.forward(xt, y, t, return_dict=False)

    @torch.no_grad()
    def multi_step(
        self,
        xt: torch.Tensor,
        y: torch.Tensor,
        timesteps: Union[Sequence[float], torch.Tensor],
        keep_history: bool = False,
    ) -> Union[torch.Tensor, Dict[str, Union[torch.Tensor, List[torch.Tensor]]]]:
        if isinstance(timesteps, torch.Tensor):
            schedule: Iterable[torch.Tensor] = timesteps.view(-1)
        else:
            schedule = list(timesteps)

        history: List[torch.Tensor] = []
        state = xt
        for step_t in schedule:
            state = self.one_step(state, y, step_t)
            if keep_history:
                history.append(state)

        if keep_history:
            return {"x0": state, "history": history}
        return state

    def parameter_summary(self) -> Dict[str, Union[int, float, bool]]:
        params = count_parameters(self)
        return {
            "total_params": params,
            "total_params_million": round(params / 1_000_000.0, 4),
            "under_3m": params < 3_000_000,
        }


def build_student_model(config: Dict) -> LightConsistencyStudent:
    model_cfg = config.get("model", config)
    return LightConsistencyStudent(
        in_channels=model_cfg.get("in_channels", 3),
        cond_in_channels=model_cfg.get("cond_in_channels", 3),
        out_channels=model_cfg.get("out_channels", 3),
        base_channels=model_cfg.get("base_channels", 32),
        channel_multipliers=tuple(model_cfg.get("channel_multipliers", (1, 2, 4))),
        time_embed_dim=model_cfg.get("time_embed_dim", 128),
        use_residual_output=model_cfg.get("use_residual_output", True),
        clamp_output=model_cfg.get("clamp_output", True),
    )
