from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch import Tensor, nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uie_consistency import (
    LossWeights,
    UnderwaterLossAssembler,
    infer_four_steps,
    infer_one_step,
    infer_three_steps,
    infer_two_steps,
)


class DemoEnhancementModel(nn.Module):
    def forward(self, x_t: Tensor, y: Tensor, t: Tensor) -> dict[str, Tensor]:
        t_map = t.view(-1, 1, 1, 1)
        residual_pred = (y - x_t) * (0.35 + 0.65 * (1.0 - t_map))
        color_map = torch.clamp(1.0 + 0.1 * (y.mean(dim=1, keepdim=True) - 0.5), 0.85, 1.15)
        x0_pred = torch.clamp((x_t + residual_pred) * color_map, 0.0, 1.0)
        return {
            "x0_pred": x0_pred,
            "residual_pred": residual_pred,
            "color_map": color_map,
        }


def downsample_measurement(x: Tensor) -> Tensor:
    return F.avg_pool2d(x, kernel_size=2, stride=2)


def underwater_degradation(x: Tensor) -> Tensor:
    x = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    attenuation = torch.tensor([0.85, 0.93, 1.05], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return torch.clamp(x * attenuation, 0.0, 1.0)


def main() -> None:
    torch.manual_seed(7)
    device = "cpu"

    y = torch.rand(2, 3, 32, 32, device=device)
    x_t = torch.clamp(0.7 * y + 0.3 * torch.rand_like(y), 0.0, 1.0)
    target_x0 = torch.clamp(1.1 * y - 0.05, 0.0, 1.0)
    teacher_x0 = torch.clamp(0.9 * target_x0 + 0.1 * x_t, 0.0, 1.0)

    model = DemoEnhancementModel().to(device)
    output = model(x_t, y, torch.ones(x_t.shape[0], device=device))

    weights = LossWeights(
        consistency=1.0,
        charbonnier=1.0,
        l1=0.2,
        ssim=0.4,
        perceptual=0.1,
        color=0.2,
        frequency=0.1,
        edge=0.1,
        measurement=0.2,
        degradation=0.2,
    )
    assembler = UnderwaterLossAssembler(
        weights=weights,
        measurement_operator=downsample_measurement,
        degradation_operator=underwater_degradation,
    )

    loss_terms = assembler(
        output,
        target_x0=target_x0,
        teacher_x0=teacher_x0,
        x_t=x_t,
        y=y,
        t=torch.ones(x_t.shape[0], device=device),
        measurement=downsample_measurement(y),
    )

    step_1 = infer_one_step(model, x_t, y)
    step_2 = infer_two_steps(model, x_t, y)
    step_3 = infer_three_steps(model, x_t, y)
    step_4 = infer_four_steps(model, x_t, y)

    report = {
        "loss_terms": {key: float(value.detach().cpu()) for key, value in loss_terms.items() if value.ndim == 0},
        "step_shapes": {
            "1_step": list(step_1.final_state.shape),
            "2_step": list(step_2.final_state.shape),
            "3_step": list(step_3.final_state.shape),
            "4_step": list(step_4.final_state.shape),
        },
        "final_means": {
            "1_step": float(step_1.final_state.mean().cpu()),
            "2_step": float(step_2.final_state.mean().cpu()),
            "3_step": float(step_3.final_state.mean().cpu()),
            "4_step": float(step_4.final_state.mean().cpu()),
        },
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
