from __future__ import annotations

import unittest

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from uie_consistency import (
    LossWeights,
    UnderwaterLossAssembler,
    infer_four_steps,
    infer_one_step,
    infer_three_steps,
    infer_two_steps,
    make_time_schedule,
    run_multistep_inference,
    ssim_loss,
)


class DummyModel(nn.Module):
    def forward(self, x_t: Tensor, y: Tensor, t: Tensor) -> dict[str, Tensor]:
        t_map = t.view(-1, 1, 1, 1)
        residual_pred = (y - x_t) * (0.4 + 0.6 * (1.0 - t_map))
        color_map = torch.clamp(1.0 + 0.05 * (y.mean(dim=1, keepdim=True) - 0.5), 0.9, 1.1)
        x0_pred = torch.clamp((x_t + residual_pred) * color_map, 0.0, 1.0)
        return {
            "x0_pred": x0_pred,
            "residual_pred": residual_pred,
            "color_map": color_map,
        }


def measurement_operator(x: Tensor) -> Tensor:
    return F.avg_pool2d(x, kernel_size=2, stride=2)


def degradation_operator(x: Tensor) -> Tensor:
    return F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)


class LossAndInferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)
        self.model = DummyModel()
        self.y = torch.rand(2, 3, 16, 16)
        self.x_t = torch.clamp(0.75 * self.y + 0.25 * torch.rand_like(self.y), 0.0, 1.0)
        self.target_x0 = torch.clamp(1.05 * self.y - 0.03, 0.0, 1.0)
        self.teacher_x0 = torch.clamp(0.9 * self.target_x0 + 0.1 * self.x_t, 0.0, 1.0)

    def test_ssim_loss_is_near_zero_for_identical_inputs(self) -> None:
        value = ssim_loss(self.y, self.y)
        self.assertLess(float(value), 1e-4)

    def test_loss_assembler_returns_expected_terms(self) -> None:
        output = self.model(self.x_t, self.y, torch.ones(self.x_t.shape[0]))
        weights = LossWeights(
            consistency=1.0,
            charbonnier=1.0,
            l1=0.1,
            ssim=0.3,
            perceptual=0.1,
            color=0.2,
            frequency=0.1,
            edge=0.1,
            measurement=0.2,
            degradation=0.2,
        )
        assembler = UnderwaterLossAssembler(
            weights=weights,
            measurement_operator=measurement_operator,
            degradation_operator=degradation_operator,
        )
        losses = assembler(
            output,
            target_x0=self.target_x0,
            teacher_x0=self.teacher_x0,
            x_t=self.x_t,
            y=self.y,
            t=torch.ones(self.x_t.shape[0]),
            measurement=measurement_operator(self.y),
        )

        expected = {
            "consistency",
            "charbonnier",
            "l1",
            "ssim",
            "perceptual",
            "color",
            "frequency",
            "edge",
            "measurement",
            "degradation",
            "time_mean",
            "total",
        }
        self.assertTrue(expected.issubset(losses.keys()))
        self.assertGreater(float(losses["total"]), 0.0)

    def test_multistep_inference_runs_for_1_to_4_steps(self) -> None:
        step_1 = infer_one_step(self.model, self.x_t, self.y)
        step_2 = infer_two_steps(self.model, self.x_t, self.y)
        step_3 = infer_three_steps(self.model, self.x_t, self.y)
        step_4 = infer_four_steps(self.model, self.x_t, self.y)

        self.assertEqual(tuple(step_1.final_state.shape), tuple(self.x_t.shape))
        self.assertEqual(len(step_1.history), 1)
        self.assertEqual(len(step_2.history), 2)
        self.assertEqual(len(step_3.history), 3)
        self.assertEqual(len(step_4.history), 4)

    def test_custom_schedule_is_supported(self) -> None:
        schedule = make_time_schedule(4, t_start=0.9, t_end=0.1)
        result = run_multistep_inference(
            self.model,
            self.x_t,
            self.y,
            num_steps=4,
            time_schedule=schedule,
            blend_schedule=[0.4, 0.6, 0.8, 1.0],
        )
        self.assertEqual(len(result.history), 4)
        self.assertTrue(torch.all(result.final_state >= 0.0))
        self.assertTrue(torch.all(result.final_state <= 1.0))


if __name__ == "__main__":
    unittest.main()
