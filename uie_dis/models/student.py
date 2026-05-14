from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class StageSpec:
    name: str
    channels: int
    resolution_divisor: int
    num_blocks: int
    num_heads: int


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight + self.bias


class TimeModulation(nn.Module):
    def __init__(self, channels: int, time_dim: int) -> None:
        super().__init__()
        self.to_scale_shift = nn.Linear(time_dim, channels * 2)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        scale, shift = self.to_scale_shift(time_emb).chunk(2, dim=1)
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)
        return x * (1.0 + scale) + shift


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 0:
            t = t[None]
        t = t.float().view(-1)
        half_dim = self.dim // 2
        exponent = -torch.log(torch.tensor(10000.0, device=t.device)) / max(half_dim - 1, 1)
        freqs = torch.exp(torch.arange(half_dim, device=t.device, dtype=t.dtype) * exponent)
        emb = t[:, None] * freqs[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class TimeMLP(nn.Module):
    def __init__(self, base_dim: int, time_dim: int) -> None:
        super().__init__()
        self.pos_emb = SinusoidalTimeEmbedding(base_dim)
        self.mlp = nn.Sequential(
            nn.Linear(base_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.pos_emb(t))


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class Downsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        return self.proj(x)


class MDTA(nn.Module):
    def __init__(self, channels: int, num_heads: int) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels={channels} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, stride=1, padding=0)
        self.qkv_dw = nn.Conv2d(
            channels * 3,
            channels * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=channels * 3,
        )
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q, k, v = self.qkv_dw(self.qkv(x)).chunk(3, dim=1)
        head_dim = c // self.num_heads
        q = q.view(b, self.num_heads, head_dim, h * w)
        k = k.view(b, self.num_heads, head_dim, h * w)
        v = v.view(b, self.num_heads, head_dim, h * w)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.temperature.unsqueeze(0)
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v).view(b, c, h, w)
        return self.project_out(out)


class GDFN(nn.Module):
    def __init__(self, channels: int, expansion: float = 2.0) -> None:
        super().__init__()
        hidden = int(channels * expansion)
        self.project_in = nn.Conv2d(channels, hidden * 2, kernel_size=1, stride=1, padding=0)
        self.dw_conv = nn.Conv2d(
            hidden * 2,
            hidden * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden * 2,
        )
        self.project_out = nn.Conv2d(hidden, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.dw_conv(self.project_in(x)).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)


class LiteTransformerBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int, time_dim: int, ffn_expansion: float = 2.0) -> None:
        super().__init__()
        self.norm1 = LayerNorm2d(channels)
        self.time1 = TimeModulation(channels, time_dim)
        self.attn = MDTA(channels, num_heads)
        self.norm2 = LayerNorm2d(channels)
        self.time2 = TimeModulation(channels, time_dim)
        self.ffn = GDFN(channels, expansion=ffn_expansion)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.time1(self.norm1(x), time_emb))
        x = x + self.ffn(self.time2(self.norm2(x), time_emb))
        return x


class GatedFusion(nn.Module):
    def __init__(self, channels: int, cond_channels: int, time_dim: int) -> None:
        super().__init__()
        self.cond_proj = nn.Conv2d(cond_channels, channels, kernel_size=1, stride=1, padding=0)
        self.time_proj = nn.Linear(time_dim, channels)
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        cond = self.cond_proj(cond)
        time_map = self.time_proj(time_emb).unsqueeze(-1).unsqueeze(-1).expand_as(x)
        gate = self.gate(torch.cat([x, cond, time_map], dim=1))
        return x + self.out_proj(gate * cond)


class CrossAttentionMixer(nn.Module):
    def __init__(self, channels: int, cond_channels: int, time_dim: int, num_heads: int) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels={channels} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.cond_proj = nn.Conv2d(cond_channels, channels, kernel_size=1, stride=1, padding=0)
        self.time_proj = nn.Linear(time_dim, channels)
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.q_proj = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.kv_proj = nn.Conv2d(channels, channels * 2, kernel_size=1, stride=1, padding=0)
        self.kv_dw = nn.Conv2d(
            channels * 2,
            channels * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=channels * 2,
        )
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, time_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, c, h, w = x.shape
        cond = self.cond_proj(cond)
        time_map = self.time_proj(time_emb).unsqueeze(-1).unsqueeze(-1).expand_as(x)
        q = self.q_proj(x + time_map)
        k, v = self.kv_dw(self.kv_proj(cond + time_map)).chunk(2, dim=1)

        head_dim = c // self.num_heads
        q = q.view(b, self.num_heads, head_dim, h * w)
        k = k.view(b, self.num_heads, head_dim, h * w)
        v = v.view(b, self.num_heads, head_dim, h * w)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.temperature.unsqueeze(0)
        attn = attn.softmax(dim=-1)
        delta = torch.matmul(attn, v).view(b, c, h, w)
        return cond, time_map, self.out_proj(delta)


class CrossAttentionFusion(nn.Module):
    def __init__(self, channels: int, cond_channels: int, time_dim: int, num_heads: int) -> None:
        super().__init__()
        self.mixer = CrossAttentionMixer(
            channels=channels,
            cond_channels=cond_channels,
            time_dim=time_dim,
            num_heads=num_heads,
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        _, _, delta = self.mixer(x, cond, time_emb)
        return x + delta


class GatedCrossAttentionFusion(nn.Module):
    def __init__(self, channels: int, cond_channels: int, time_dim: int, num_heads: int) -> None:
        super().__init__()
        self.mixer = CrossAttentionMixer(
            channels=channels,
            cond_channels=cond_channels,
            time_dim=time_dim,
            num_heads=num_heads,
        )
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        cond, time_map, delta = self.mixer(x, cond, time_emb)
        gate = self.gate(torch.cat([x, cond, time_map], dim=1))
        return x + gate * delta


def build_condition_fusion(
    fusion_mode: str,
    channels: int,
    cond_channels: int,
    time_dim: int,
    num_heads: int,
) -> nn.Module:
    if fusion_mode == "gated":
        return GatedFusion(channels, cond_channels, time_dim)
    if fusion_mode == "cross-attn":
        return CrossAttentionFusion(channels, cond_channels, time_dim, num_heads)
    if fusion_mode == "gated-cross-attn":
        return GatedCrossAttentionFusion(channels, cond_channels, time_dim, num_heads)
    raise ValueError(
        f"Unsupported fusion_mode={fusion_mode!r}. Expected one of: "
        "'gated', 'cross-attn', 'gated-cross-attn'."
    )


class FusionTransformerStage(nn.Module):
    def __init__(
        self,
        channels: int,
        cond_channels: int,
        time_dim: int,
        num_blocks: int,
        num_heads: int,
        ffn_expansion: float = 2.0,
        fusion_mode: str = "gated",
    ) -> None:
        super().__init__()
        self.fusion_mode = fusion_mode
        self.fusion = build_condition_fusion(
            fusion_mode=fusion_mode,
            channels=channels,
            cond_channels=cond_channels,
            time_dim=time_dim,
            num_heads=num_heads,
        )
        self.blocks = nn.ModuleList(
            [
                LiteTransformerBlock(
                    channels=channels,
                    num_heads=num_heads,
                    time_dim=time_dim,
                    ffn_expansion=ffn_expansion,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        x = self.fusion(x, cond, time_emb)
        for block in self.blocks:
            x = block(x, time_emb)
        return x


class ConditionEncoder(nn.Module):
    def __init__(self, stage_dims: Sequence[int], latent_dim: int) -> None:
        super().__init__()
        if len(stage_dims) != 4:
            raise ValueError("ConditionEncoder expects 4 encoder stage dims.")

        self.stem = nn.Sequential(
            nn.Conv2d(3, stage_dims[0], kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(stage_dims[0], stage_dims[0], kernel_size=3, stride=1, padding=1),
            nn.GELU(),
        )
        self.down1 = Downsample(stage_dims[0], stage_dims[1])
        self.down2 = Downsample(stage_dims[1], stage_dims[2])
        self.down3 = Downsample(stage_dims[2], stage_dims[3])
        self.down4 = Downsample(stage_dims[3], latent_dim)

    def forward(self, y: torch.Tensor) -> List[torch.Tensor]:
        cond0 = self.stem(y)
        cond1 = self.down1(cond0)
        cond2 = self.down2(cond1)
        cond3 = self.down3(cond2)
        cond4 = self.down4(cond3)
        return [cond0, cond1, cond2, cond3, cond4]


class OutputHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 3) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class RestormerLiteStudent(nn.Module):
    """Restormer-lite student model for consistency distillation."""

    default_stage_specs: Tuple[StageSpec, ...] = (
        StageSpec("enc1", channels=48, resolution_divisor=1, num_blocks=1, num_heads=1),
        StageSpec("enc2", channels=72, resolution_divisor=2, num_blocks=1, num_heads=2),
        StageSpec("enc3", channels=96, resolution_divisor=4, num_blocks=2, num_heads=3),
        StageSpec("enc4", channels=128, resolution_divisor=8, num_blocks=2, num_heads=4),
        StageSpec("latent", channels=160, resolution_divisor=16, num_blocks=3, num_heads=5),
    )

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        stage_dims: Sequence[int] = (48, 72, 96, 128),
        latent_dim: int = 160,
        stage_blocks: Sequence[int] = (1, 1, 2, 2),
        decoder_blocks: Sequence[int] = (2, 2, 1, 1),
        encoder_heads: Sequence[int] = (1, 2, 3, 4),
        decoder_heads: Sequence[int] = (4, 3, 2, 1),
        latent_blocks: int = 3,
        latent_heads: int = 5,
        time_dim: int = 256,
        ffn_expansion: float = 2.0,
        enable_color_head: bool = True,
        fusion_mode: str = "gated",
    ) -> None:
        super().__init__()
        if not (len(stage_dims) == len(stage_blocks) == len(decoder_blocks) == 4):
            raise ValueError("stage_dims, stage_blocks and decoder_blocks must have length 4.")
        if not (len(encoder_heads) == len(decoder_heads) == 4):
            raise ValueError("encoder_heads and decoder_heads must have length 4.")
        if fusion_mode not in {"gated", "cross-attn", "gated-cross-attn"}:
            raise ValueError(
                f"Unsupported fusion_mode={fusion_mode!r}. Expected one of: "
                "'gated', 'cross-attn', 'gated-cross-attn'."
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stage_dims = tuple(stage_dims)
        self.stage_blocks = tuple(stage_blocks)
        self.decoder_blocks = tuple(decoder_blocks)
        self.encoder_heads = tuple(encoder_heads)
        self.decoder_heads = tuple(decoder_heads)
        self.latent_dim = latent_dim
        self.latent_blocks = latent_blocks
        self.latent_heads = latent_heads
        self.enable_color_head = enable_color_head
        self.fusion_mode = fusion_mode
        self.required_multiple = 16

        self.time_mlp = TimeMLP(base_dim=time_dim // 2, time_dim=time_dim)
        self.patch_embed = OverlapPatchEmbed(in_channels * 2, stage_dims[0])
        self.condition_encoder = ConditionEncoder(stage_dims=stage_dims, latent_dim=latent_dim)

        self.encoder_stages = nn.ModuleList(
            [
                FusionTransformerStage(
                    channels=stage_dims[idx],
                    cond_channels=stage_dims[idx],
                    time_dim=time_dim,
                    num_blocks=stage_blocks[idx],
                    num_heads=encoder_heads[idx],
                    ffn_expansion=ffn_expansion,
                    fusion_mode=fusion_mode,
                )
                for idx in range(4)
            ]
        )
        self.encoder_downsamples = nn.ModuleList(
            [Downsample(stage_dims[idx], stage_dims[idx + 1]) for idx in range(3)]
            + [Downsample(stage_dims[-1], latent_dim)]
        )

        self.latent_stage = FusionTransformerStage(
            channels=latent_dim,
            cond_channels=latent_dim,
            time_dim=time_dim,
            num_blocks=latent_blocks,
            num_heads=latent_heads,
            ffn_expansion=ffn_expansion,
            fusion_mode=fusion_mode,
        )

        self.decoder_upsamples = nn.ModuleList(
            [
                Upsample(latent_dim, stage_dims[-1]),
                Upsample(stage_dims[-1], stage_dims[-2]),
                Upsample(stage_dims[-2], stage_dims[-3]),
                Upsample(stage_dims[-3], stage_dims[-4]),
            ]
        )
        self.decoder_merges = nn.ModuleList(
            [
                nn.Conv2d(stage_dims[-1] * 2, stage_dims[-1], kernel_size=1, stride=1, padding=0),
                nn.Conv2d(stage_dims[-2] * 2, stage_dims[-2], kernel_size=1, stride=1, padding=0),
                nn.Conv2d(stage_dims[-3] * 2, stage_dims[-3], kernel_size=1, stride=1, padding=0),
                nn.Conv2d(stage_dims[-4] * 2, stage_dims[-4], kernel_size=1, stride=1, padding=0),
            ]
        )
        decoder_dims = (stage_dims[-1], stage_dims[-2], stage_dims[-3], stage_dims[-4])
        decoder_cond_dims = decoder_dims
        self.decoder_stages = nn.ModuleList(
            [
                FusionTransformerStage(
                    channels=decoder_dims[idx],
                    cond_channels=decoder_cond_dims[idx],
                    time_dim=time_dim,
                    num_blocks=decoder_blocks[idx],
                    num_heads=decoder_heads[idx],
                    ffn_expansion=ffn_expansion,
                    fusion_mode=fusion_mode,
                )
                for idx in range(4)
            ]
        )

        self.final_refine = LiteTransformerBlock(
            channels=stage_dims[0],
            num_heads=encoder_heads[0],
            time_dim=time_dim,
            ffn_expansion=ffn_expansion,
        )
        self.x0_head = OutputHead(stage_dims[0], out_channels=out_channels)
        self.residual_head = OutputHead(stage_dims[0], out_channels=out_channels)
        self.color_head = OutputHead(stage_dims[0], out_channels=out_channels) if enable_color_head else None

    @staticmethod
    def _pad_to_multiple(x: torch.Tensor, multiple: int) -> Tuple[torch.Tensor, Dict[str, int]]:
        _, _, h, w = x.shape
        pad_h = (multiple - h % multiple) % multiple
        pad_w = (multiple - w % multiple) % multiple
        if pad_h == 0 and pad_w == 0:
            return x, {"pad_h": 0, "pad_w": 0, "orig_h": h, "orig_w": w}
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return x, {"pad_h": pad_h, "pad_w": pad_w, "orig_h": h, "orig_w": w}

    @staticmethod
    def _crop_to_original(x: Optional[torch.Tensor], pad_info: Dict[str, int]) -> Optional[torch.Tensor]:
        if x is None:
            return None
        h = pad_info["orig_h"]
        w = pad_info["orig_w"]
        return x[..., :h, :w]

    def stage_description(self, input_resolution: Tuple[int, int] = (256, 256)) -> List[Dict[str, object]]:
        h, w = input_resolution
        specs: List[Dict[str, object]] = []
        encoder_resolutions = [1, 2, 4, 8]
        for idx, divisor in enumerate(encoder_resolutions):
            specs.append(
                {
                    "name": f"enc{idx + 1}",
                    "channels": self.stage_dims[idx],
                    "resolution": (h // divisor, w // divisor),
                    "num_blocks": self.stage_blocks[idx],
                    "num_heads": self.encoder_heads[idx],
                }
            )
        specs.append(
            {
                "name": "latent",
                "channels": self.latent_dim,
                "resolution": (h // 16, w // 16),
                "num_blocks": self.latent_blocks,
                "num_heads": self.latent_heads,
            }
        )
        decoder_resolutions = [8, 4, 2, 1]
        decoder_dims = [self.stage_dims[3], self.stage_dims[2], self.stage_dims[1], self.stage_dims[0]]
        for idx, divisor in enumerate(decoder_resolutions):
            specs.append(
                {
                    "name": f"dec{4 - idx}",
                    "channels": decoder_dims[idx],
                    "resolution": (h // divisor, w // divisor),
                    "num_blocks": self.decoder_blocks[idx],
                    "num_heads": self.decoder_heads[idx],
                }
            )
        return specs

    def forward(
        self,
        x_t: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
    ) -> Dict[str, object]:
        if x_t.shape != y.shape:
            raise ValueError(f"x_t and y must have the same shape, got {x_t.shape} and {y.shape}.")
        if x_t.dim() != 4 or x_t.shape[1] != self.in_channels:
            raise ValueError(f"x_t must be BCHW with {self.in_channels} channels, got {x_t.shape}.")

        original_input_shape = tuple(x_t.shape)
        x_t, pad_info = self._pad_to_multiple(x_t, self.required_multiple)
        y, _ = self._pad_to_multiple(y, self.required_multiple)
        time_emb = self.time_mlp(t.to(device=x_t.device))

        cond_pyramid = self.condition_encoder(y)
        x = self.patch_embed(torch.cat([x_t, y], dim=1))

        encoder_feats: List[torch.Tensor] = []
        encoder_shapes: List[Tuple[int, ...]] = []
        for idx, stage in enumerate(self.encoder_stages):
            x = stage(x, cond_pyramid[idx], time_emb)
            encoder_feats.append(x)
            encoder_shapes.append(tuple(x.shape))
            x = self.encoder_downsamples[idx](x)

        x = self.latent_stage(x, cond_pyramid[-1], time_emb)
        latent_feature = x
        latent_shape = tuple(x.shape)

        decoder_feats: List[torch.Tensor] = []
        decoder_shapes: List[Tuple[int, ...]] = []
        decoder_cond = [cond_pyramid[3], cond_pyramid[2], cond_pyramid[1], cond_pyramid[0]]
        decoder_skips = [encoder_feats[3], encoder_feats[2], encoder_feats[1], encoder_feats[0]]
        for up, merge, stage, skip, cond in zip(
            self.decoder_upsamples,
            self.decoder_merges,
            self.decoder_stages,
            decoder_skips,
            decoder_cond,
        ):
            x = up(x)
            x = merge(torch.cat([x, skip], dim=1))
            x = stage(x, cond, time_emb)
            decoder_feats.append(x)
            decoder_shapes.append(tuple(x.shape))

        x = self.final_refine(x, time_emb)

        x0_pred = self._crop_to_original(self.x0_head(x), pad_info)
        residual_pred = self._crop_to_original(self.residual_head(x), pad_info)
        color_map = self._crop_to_original(self.color_head(x), pad_info) if self.color_head is not None else None

        features = {
            "encoder": encoder_feats,
            "latent": latent_feature,
            "decoder": decoder_feats,
            "refined": x,
            "condition": cond_pyramid,
        }
        meta = {
            "input_shape": original_input_shape,
            "padded_shape": tuple(x_t.shape),
            "original_shape": (pad_info["orig_h"], pad_info["orig_w"]),
            "padding": {"pad_h": pad_info["pad_h"], "pad_w": pad_info["pad_w"]},
            "encoder_shapes": encoder_shapes,
            "latent_shape": latent_shape,
            "decoder_shapes": decoder_shapes,
            "stage_specs_256": self.stage_description((256, 256)),
            "fusion_mode": self.fusion_mode,
            "available_fusion_modes": ("gated", "cross-attn", "gated-cross-attn"),
            "fusion": (
                "gated residual fusion before transformer blocks at every encoder/latent/decoder stage"
                if self.fusion_mode == "gated"
                else "lightweight conditional cross-attention fusion before transformer blocks at every "
                "encoder/latent/decoder stage"
                if self.fusion_mode == "cross-attn"
                else "gated lightweight conditional cross-attention fusion before transformer blocks at every "
                "encoder/latent/decoder stage"
            ),
            "time_injection": "scale-shift modulation before attention and FFN in each block",
        }
        return {
            "x0_pred": x0_pred,
            "residual_pred": residual_pred,
            "color_map": color_map,
            "features": features,
            "meta": meta,
        }
