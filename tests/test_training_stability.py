from __future__ import annotations

import torch
from torch import nn
from torch.amp import GradScaler
from torch.utils.data import DataLoader, Dataset

from uie_dis.training import ModelEMA, TrainConfig, create_loss_assembler, create_optimizer, train_one_epoch


class TinyDataset(Dataset[dict[str, torch.Tensor | str]]):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        x = torch.rand(3, 32, 32)
        return {"input": x, "target": x.clone(), "key": f"sample_{index}"}


class GoodStudent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.9))

    def forward(self, x_t: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> dict[str, torch.Tensor]:
        t_map = t.view(-1, 1, 1, 1)
        x0_pred = torch.clamp(self.scale * y + (1.0 - self.scale) * x_t * (1.0 - 0.1 * t_map), 0.0, 1.0)
        return {"x0_pred": x0_pred, "residual_pred": x0_pred - x_t}


class BadStudent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, x_t: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> dict[str, torch.Tensor]:
        nan_value = self.weight * torch.full_like(y, float("nan"))
        return {"x0_pred": nan_value, "residual_pred": nan_value}


def test_training_without_teacher_disables_consistency_path():
    device = torch.device("cpu")
    config = TrainConfig(
        epochs=1,
        batch_size=1,
        num_workers=0,
        amp=False,
        loss_profile="memory_efficient",
        log_every=1000,
    )
    config.schedule.noise_std = 0.0
    model = GoodStudent().to(device)
    optimizer = create_optimizer(model, config)
    scaler = GradScaler("cpu", enabled=False)
    ema = ModelEMA(model, decay=0.9)
    loss_assembler, _ = create_loss_assembler(config)
    loss_assembler.weights.consistency = 0.0
    loader = DataLoader(TinyDataset(), batch_size=1, shuffle=False)

    summary, _ = train_one_epoch(
        epoch=1,
        student=model,
        teacher=None,
        dataloader=loader,
        optimizer=optimizer,
        loss_assembler=loss_assembler,
        scaler=scaler,
        ema=ema,
        device=device,
        config=config,
        global_step=0,
    )

    assert "consistency" not in summary
    assert summary["loss"] > 0.0
    assert summary["skipped_batches"] == 0.0


def test_non_finite_loss_batch_is_skipped():
    device = torch.device("cpu")
    config = TrainConfig(
        epochs=1,
        batch_size=1,
        num_workers=0,
        amp=False,
        loss_profile="memory_efficient",
        log_every=1000,
    )
    config.schedule.noise_std = 0.0
    model = BadStudent().to(device)
    optimizer = create_optimizer(model, config)
    scaler = GradScaler("cpu", enabled=False)
    ema = ModelEMA(model, decay=0.9)
    loss_assembler, _ = create_loss_assembler(config)
    loss_assembler.weights.consistency = 0.0
    loader = DataLoader(TinyDataset(), batch_size=1, shuffle=False)

    summary, _ = train_one_epoch(
        epoch=1,
        student=model,
        teacher=None,
        dataloader=loader,
        optimizer=optimizer,
        loss_assembler=loss_assembler,
        scaler=scaler,
        ema=ema,
        device=device,
        config=config,
        global_step=0,
    )

    assert summary["loss"] == 0.0
    assert summary["skipped_batches"] == 2.0
