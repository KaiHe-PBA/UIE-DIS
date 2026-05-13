import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_time_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    if t.dim() == 0:
        t = t[None]
    t = t.float()
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, device=t.device).float() / max(half, 1))
    args = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        return (x - mean) / torch.sqrt(var + self.eps) * self.weight + self.bias


class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class TimeMLP(nn.Module):
    def __init__(self, time_dim: int = 256, hidden_dim: int = 512, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, temb: torch.Tensor) -> torch.Tensor:
        return self.net(temb)


class NAFBlock(nn.Module):
    def __init__(self, dim: int, expansion: int = 2, dw_kernel: int = 3, dropout: float = 0.0):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        hidden = dim * expansion
        self.pw1 = nn.Conv2d(dim, hidden * 2, 1, 1, 0)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, dw_kernel, 1, dw_kernel // 2, groups=hidden * 2)
        self.gate = SimpleGate()
        self.sca_pool = nn.AdaptiveAvgPool2d(1)
        self.sca_conv = nn.Conv2d(hidden, hidden, 1, 1, 0)
        self.pw2 = nn.Conv2d(hidden, dim, 1, 1, 0)
        self.drop1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

        self.norm2 = LayerNorm2d(dim)
        ffn_hidden = dim * expansion
        self.ffn_pw1 = nn.Conv2d(dim, ffn_hidden * 2, 1, 1, 0)
        self.ffn_gate = SimpleGate()
        self.ffn_pw2 = nn.Conv2d(ffn_hidden, dim, 1, 1, 0)
        self.drop2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        nn.init.constant_(self.beta, 1e-2)
        nn.init.constant_(self.gamma, 1e-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = self.pw1(h)
        h = self.dwconv(h)
        h = self.gate(h)
        h = h * self.sca_conv(self.sca_pool(h))
        h = self.pw2(h)
        x = x + self.drop1(h) * self.beta

        h2 = self.norm2(x)
        h2 = self.ffn_pw1(h2)
        h2 = self.ffn_gate(h2)
        h2 = self.ffn_pw2(h2)
        x = x + self.drop2(h2) * self.gamma
        return x


class CondFiLM(nn.Module):
    def __init__(self, x_dim: int, cond_dim: int, time_dim: int, use_gate: bool = True):
        super().__init__()
        self.cond_proj = nn.Conv2d(cond_dim, x_dim, 1, 1, 0)
        self.time_proj = nn.Linear(time_dim, x_dim * (3 if use_gate else 2))
        self.norm = LayerNorm2d(x_dim)
        self.use_gate = use_gate

    def forward(self, x: torch.Tensor, cond: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if cond.shape[-2:] != (h, w):
            cond = F.interpolate(cond, size=(h, w), mode='bilinear', align_corners=False)
        cond = self.cond_proj(cond)
        tp = self.time_proj(temb).view(b, -1, 1, 1)
        if self.use_gate:
            gamma, beta, gate = tp.chunk(3, dim=1)
            gate = torch.sigmoid(gate)
        else:
            gamma, beta = tp.chunk(2, dim=1)
            gate = None
        base = self.norm(x)
        out = base * (1.0 + gamma) + beta
        if gate is not None:
            out = base + gate * (out - base)
        return out + cond


class Downsample(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim * 2, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim * 2, 1, 1, 0)
        self.ps = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ps(self.conv(x))


class ConditionEncoder(nn.Module):
    def __init__(
        self,
        in_ch: int = 3,
        base_dim: int = 32,
        num_stages: int = 4,
        blocks_per_stage: Tuple[int, ...] = (1, 1, 1, 1),
        expansion: int = 2,
    ):
        super().__init__()
        dims = [base_dim * (2 ** i) for i in range(num_stages)]
        self.stem = nn.Conv2d(in_ch, dims[0], 3, 1, 1)
        self.stages = nn.ModuleList()
        self.downs = nn.ModuleList()
        for si in range(num_stages):
            dim = dims[si]
            self.stages.append(nn.Sequential(*[NAFBlock(dim, expansion=expansion) for _ in range(blocks_per_stage[si])]))
            if si < num_stages - 1:
                self.downs.append(Downsample(dim))

    def forward(self, y: torch.Tensor) -> List[torch.Tensor]:
        feats = []
        x = self.stem(y)
        for si, stage in enumerate(self.stages):
            x = stage(x)
            feats.append(x)
            if si < len(self.downs):
                x = self.downs[si](x)
        return feats


class UWCNAFConsistencyStudent(nn.Module):
    def __init__(
        self,
        in_ch_xt: int = 3,
        in_ch_y: int = 3,
        out_ch: int = 3,
        base_dim: int = 32,
        num_stages: int = 4,
        enc_blocks: Tuple[int, ...] = (1, 1, 2, 2),
        dec_blocks: Tuple[int, ...] = (1, 1, 1, 1),
        expansion: int = 2,
        time_emb_dim: int = 256,
        time_mlp_dim: int = 512,
        use_residual_head: bool = True,
        use_gate_in_film: bool = True,
    ):
        super().__init__()
        dims = [base_dim * (2 ** i) for i in range(num_stages)]
        self.time_dim = time_emb_dim
        self.time_mlp = TimeMLP(time_dim=time_emb_dim, hidden_dim=time_mlp_dim, out_dim=time_emb_dim)
        self.cond_encoder = ConditionEncoder(
            in_ch=in_ch_y,
            base_dim=base_dim,
            num_stages=num_stages,
            blocks_per_stage=(1, 1, 1, 1),
            expansion=expansion,
        )
        self.stem = nn.Conv2d(in_ch_xt + in_ch_y, dims[0], 3, 1, 1)

        self.enc_inject = nn.ModuleList()
        self.enc_stages = nn.ModuleList()
        self.downs = nn.ModuleList()
        for si in range(num_stages):
            dim = dims[si]
            self.enc_inject.append(CondFiLM(dim, dims[si], time_emb_dim, use_gate=use_gate_in_film))
            self.enc_stages.append(nn.Sequential(*[NAFBlock(dim, expansion=expansion) for _ in range(enc_blocks[si])]))
            if si < num_stages - 1:
                self.downs.append(Downsample(dim))

        self.mid_block1 = NAFBlock(dims[-1], expansion=expansion)
        self.mid_inject = CondFiLM(dims[-1], dims[-1], time_emb_dim, use_gate=use_gate_in_film)
        self.mid_block2 = NAFBlock(dims[-1], expansion=expansion)

        self.ups = nn.ModuleList()
        self.dec_stages = nn.ModuleList()
        self.dec_inject = nn.ModuleList()
        for si in reversed(range(num_stages - 1)):
            dim = dims[si]
            self.ups.append(Upsample(dims[si + 1]))
            self.dec_stages.append(nn.Sequential(
                nn.Conv2d(dim * 2, dim, 1, 1, 0),
                *[NAFBlock(dim, expansion=expansion) for _ in range(dec_blocks[si])],
            ))
            self.dec_inject.append(CondFiLM(dim, dims[si], time_emb_dim, use_gate=use_gate_in_film))

        self.out_norm = LayerNorm2d(dims[0])
        self.out_conv = nn.Conv2d(dims[0], out_ch, 3, 1, 1)
        self.use_residual_head = use_residual_head
        if use_residual_head:
            self.res_conv = nn.Conv2d(dims[0], out_ch, 3, 1, 1)

        # Start from an identity-like restoration target: x0 ~= y.
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)
        if use_residual_head:
            nn.init.zeros_(self.res_conv.weight)
            nn.init.zeros_(self.res_conv.bias)

    def forward(self, x_t: torch.Tensor, y: torch.Tensor, t: torch.Tensor, return_residual: bool = False) -> Dict[str, torch.Tensor]:
        temb = self.time_mlp(sinusoidal_time_embedding(t, self.time_dim))
        cond_feats = self.cond_encoder(y)
        x = self.stem(torch.cat([x_t, y], dim=1))

        skips = []
        for si, stage in enumerate(self.enc_stages):
            x = self.enc_inject[si](x, cond_feats[si], temb)
            x = stage(x)
            skips.append(x)
            if si < len(self.downs):
                x = self.downs[si](x)

        x = self.mid_block1(x)
        x = self.mid_inject(x, cond_feats[-1], temb)
        x = self.mid_block2(x)

        for di in range(len(self.ups)):
            x = self.ups[di](x)
            skip = skips[-2 - di]
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = self.dec_stages[di](x)
            x = self.dec_inject[di](x, cond_feats[-2 - di], temb)

        x = self.out_norm(x)
        x0 = y + self.out_conv(x)
        out = {'x0': x0}
        if self.use_residual_head or return_residual:
            residual = self.res_conv(x) if self.use_residual_head else x0 - y
            out['residual'] = residual
            if self.use_residual_head:
                out['x0_from_residual'] = y + residual
        return out
