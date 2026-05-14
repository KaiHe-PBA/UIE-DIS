from __future__ import annotations

import copy
import json
import math
import time
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from uie_consistency import UnderwaterLossAssembler, build_recommended_loss_setup, run_multistep_inference, ssim
from .models import RestormerLiteStudent


@dataclass
class ConsistencyScheduleConfig:
    min_t: float = 0.0
    max_t: float = 1.0
    noise_std: float = 0.03
    t_power: float = 1.0
    clamp_range: tuple[float, float] = (0.0, 1.0)


@dataclass
class TrainConfig:
    output_dir: str = "./outputs/consistency_student"
    seed: int = 42
    epochs: int = 200
    batch_size: int = 4
    val_batch_size: int = 1
    num_workers: int = 4
    train_patch_size: int = 256
    val_patch_size: int | None = None
    lr: float = 2e-4
    weight_decay: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.99)
    grad_clip_norm: float = 1.0
    amp: bool = True
    ema_decay: float = 0.999
    ema_update_every: int = 1
    validate_every: int = 1
    save_every: int = 1
    log_every: int = 20
    val_num_steps: int = 1
    loss_profile: str = "balanced"
    random_flip: bool = True
    center_crop_validation: bool = False
    device: str = "auto"
    scheduler_t_max: int | None = None
    scheduler_min_lr: float = 1e-6
    teacher_checkpoint: str | None = None
    resume_checkpoint: str | None = None
    teacher_use_checkpoint_kwargs: bool = True
    schedule: ConsistencyScheduleConfig = field(default_factory=ConsistencyScheduleConfig)


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in model.state_dict().items()
        }
        self.num_updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.num_updates += 1
        decay = self.decay
        model_state = model.state_dict()
        for name, value in model_state.items():
            self.shadow[name].mul_(decay).add_(value.detach(), alpha=1.0 - decay)

    def state_dict(self) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow": self.shadow,
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        self.decay = float(state_dict["decay"])
        self.num_updates = int(state_dict["num_updates"])
        self.shadow = {name: tensor.detach().clone() for name, tensor in state_dict["shadow"].items()}

    @contextmanager
    def apply_to(self, model: nn.Module):
        backup = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=True)
        try:
            yield model
        finally:
            model.load_state_dict(backup, strict=True)


class AverageMeter:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total / self.count


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def build_model_kwargs_from_config(args: Any) -> dict[str, Any]:
    return {
        "stage_dims": tuple(args.stage_dims),
        "latent_dim": args.latent_dim,
        "stage_blocks": tuple(args.stage_blocks),
        "decoder_blocks": tuple(args.decoder_blocks),
        "encoder_heads": tuple(args.encoder_heads),
        "decoder_heads": tuple(args.decoder_heads),
        "latent_blocks": args.latent_blocks,
        "latent_heads": args.latent_heads,
        "time_dim": args.time_dim,
        "ffn_expansion": args.ffn_expansion,
        "enable_color_head": args.enable_color_head,
        "fusion_mode": args.fusion_mode,
    }


def build_student_model(model_kwargs: Mapping[str, Any], device: torch.device) -> RestormerLiteStudent:
    model = RestormerLiteStudent(**dict(model_kwargs))
    return model.to(device)


def load_teacher_model(
    *,
    device: torch.device,
    student_model_kwargs: Mapping[str, Any],
    teacher_checkpoint: str | None,
    teacher_use_checkpoint_kwargs: bool,
) -> tuple[RestormerLiteStudent, dict[str, Any]]:
    teacher_kwargs = dict(student_model_kwargs)
    teacher = RestormerLiteStudent(**teacher_kwargs).to(device)

    metadata: dict[str, Any] = {"loaded_from": None, "model_kwargs": teacher_kwargs}
    if teacher_checkpoint is None:
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        return teacher, metadata

    checkpoint = torch.load(teacher_checkpoint, map_location=device)
    checkpoint_kwargs = checkpoint.get("model_kwargs")
    if teacher_use_checkpoint_kwargs and checkpoint_kwargs is not None:
        teacher_kwargs = dict(checkpoint_kwargs)
        teacher = RestormerLiteStudent(**teacher_kwargs).to(device)
    state_dict = checkpoint.get("model") or checkpoint.get("student") or checkpoint.get("state_dict")
    if state_dict is None:
        raise KeyError(
            f"Teacher checkpoint {teacher_checkpoint} does not contain `model`, `student`, or `state_dict`."
        )
    teacher.load_state_dict(state_dict, strict=True)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    metadata["loaded_from"] = teacher_checkpoint
    metadata["model_kwargs"] = teacher_kwargs
    return teacher, metadata


def maybe_autocast(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)
    if device.type == "cpu":
        return torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True)
    return nullcontext()


def sample_time_steps(
    batch_size: int,
    *,
    device: torch.device,
    min_t: float,
    max_t: float,
    t_power: float = 1.0,
) -> Tensor:
    t = torch.rand(batch_size, device=device)
    if t_power != 1.0:
        t = t.pow(t_power)
    return t * (max_t - min_t) + min_t


def build_consistency_state(
    target_x0: Tensor,
    y: Tensor,
    t: Tensor,
    *,
    noise_std: float,
    clamp_range: tuple[float, float] = (0.0, 1.0),
) -> Tensor:
    t_map = t.view(-1, 1, 1, 1)
    mixed = (1.0 - t_map) * target_x0 + t_map * y
    if noise_std > 0.0:
        mixed = mixed + noise_std * t_map * torch.randn_like(mixed)
    return mixed.clamp(*clamp_range)


def compute_psnr(pred: Tensor, target: Tensor, eps: float = 1e-8) -> Tensor:
    mse = F.mse_loss(pred, target, reduction="none").mean(dim=(1, 2, 3))
    return 10.0 * torch.log10(1.0 / (mse + eps))


def compute_reconstruction_metrics(pred: Tensor, target: Tensor) -> dict[str, float]:
    psnr = compute_psnr(pred, target).mean().item()
    ssim_value = ssim(pred, target).mean().item()
    l1 = F.l1_loss(pred, target).item()
    return {"psnr": psnr, "ssim": ssim_value, "l1": l1}


def create_optimizer(model: nn.Module, config: TrainConfig) -> AdamW:
    return AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay, betas=config.betas)


def create_scheduler(optimizer: AdamW, config: TrainConfig) -> CosineAnnealingLR:
    t_max = config.scheduler_t_max or config.epochs
    return CosineAnnealingLR(optimizer, T_max=t_max, eta_min=config.scheduler_min_lr)


def create_loss_assembler(config: TrainConfig) -> tuple[UnderwaterLossAssembler, dict[str, Any]]:
    recommended = build_recommended_loss_setup(config.loss_profile)
    assembler = UnderwaterLossAssembler(
        weights=recommended.weights,
        perceptual_backend=recommended.perceptual_backend,
        clamp_range=config.schedule.clamp_range,
    )
    return assembler, {
        "loss_profile": config.loss_profile,
        "enabled_losses": recommended.enabled_losses,
        "disabled_losses": recommended.disabled_losses,
        "rationale": recommended.rationale,
        "lightweight_suggestions": recommended.further_lightweight_suggestions,
        "weights": asdict(recommended.weights),
        "perceptual_backend": recommended.perceptual_backend,
    }


def save_checkpoint(
    *,
    output_dir: str | Path,
    epoch: int,
    global_step: int,
    student: nn.Module,
    optimizer: AdamW,
    scheduler: CosineAnnealingLR,
    scaler: GradScaler,
    ema: ModelEMA,
    model_kwargs: Mapping[str, Any],
    config: TrainConfig,
    best_metric: float,
    filename: str,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / filename
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "student": student.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "ema": ema.state_dict(),
            "model_kwargs": dict(model_kwargs),
            "config": config_to_dict(config),
            "best_metric": best_metric,
        },
        checkpoint_path,
    )
    return checkpoint_path


def load_training_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: torch.device,
    student: nn.Module,
    optimizer: AdamW,
    scheduler: CosineAnnealingLR,
    scaler: GradScaler,
    ema: ModelEMA,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("student") or checkpoint.get("model")
    if state_dict is None:
        raise KeyError(f"Checkpoint {checkpoint_path} does not contain `student` or `model`.")
    student.load_state_dict(state_dict, strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])
    ema.load_state_dict(checkpoint["ema"])
    return checkpoint


def tensor_to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            output[key] = value.to(device, non_blocking=True)
        else:
            output[key] = value
    return output


def train_one_epoch(
    *,
    epoch: int,
    student: nn.Module,
    teacher: Optional[nn.Module],
    dataloader: DataLoader,
    optimizer: AdamW,
    loss_assembler: UnderwaterLossAssembler,
    scaler: GradScaler,
    ema: ModelEMA,
    device: torch.device,
    config: TrainConfig,
    global_step: int,
) -> tuple[dict[str, float], int]:
    student.train()
    if teacher is not None:
        teacher.eval()
    total_meter = AverageMeter()
    component_meters: dict[str, AverageMeter] = {}
    skipped_batches = 0
    start_time = time.time()

    for batch_index, batch in enumerate(dataloader, start=1):
        batch = tensor_to_device(batch, device)
        y = batch["input"]
        target_x0 = batch["target"]
        t = sample_time_steps(
            target_x0.shape[0],
            device=device,
            min_t=config.schedule.min_t,
            max_t=config.schedule.max_t,
            t_power=config.schedule.t_power,
        )
        x_t = build_consistency_state(
            target_x0,
            y,
            t,
            noise_std=config.schedule.noise_std,
            clamp_range=config.schedule.clamp_range,
        )

        optimizer.zero_grad(set_to_none=True)

        teacher_output = None
        if teacher is not None and loss_assembler.weights.consistency > 0:
            with torch.no_grad():
                with maybe_autocast(device, config.amp):
                    teacher_output = teacher(x_t, y, t)

        with maybe_autocast(device, config.amp):
            student_output = student(x_t, y, t)
            loss_terms = loss_assembler(
                student_output,
                target_x0=target_x0,
                teacher_output=teacher_output,
                x_t=x_t,
                y=y,
                t=t,
            )
            total_loss = loss_terms["total"]

        if not torch.isfinite(total_loss):
            skipped_batches += 1
            optimizer.zero_grad(set_to_none=True)
            print(
                f"[Warn] epoch={epoch} step={batch_index}/{len(dataloader)} "
                "non-finite loss detected, skipping batch."
            )
            continue

        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = None
        if config.grad_clip_norm > 0.0:
            grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), config.grad_clip_norm)
        if grad_norm is not None and not torch.isfinite(grad_norm):
            skipped_batches += 1
            optimizer.zero_grad(set_to_none=True)
            print(
                f"[Warn] epoch={epoch} step={batch_index}/{len(dataloader)} "
                "non-finite gradient norm detected, skipping optimizer step."
            )
            continue
        scaler.step(optimizer)
        scaler.update()

        if global_step % config.ema_update_every == 0:
            ema.update(student)

        batch_size = int(target_x0.shape[0])
        total_meter.update(float(total_loss.detach().item()), n=batch_size)
        for name, value in loss_terms.items():
            if name not in component_meters:
                component_meters[name] = AverageMeter()
            component_meters[name].update(float(value.detach().item()), n=batch_size)

        if batch_index % config.log_every == 0 or batch_index == len(dataloader):
            elapsed = time.time() - start_time
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"[Train] epoch={epoch} step={batch_index}/{len(dataloader)} "
                f"lr={lr:.6e} loss={total_meter.avg:.4f} "
                f"time={elapsed:.1f}s"
            )
        global_step += 1

    summary = {"loss": total_meter.avg}
    for name, meter in component_meters.items():
        summary[name] = meter.avg
    summary["skipped_batches"] = float(skipped_batches)
    return summary, global_step


@torch.no_grad()
def validate(
    *,
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_steps: int,
) -> dict[str, float]:
    model.eval()
    meters = {
        "psnr": AverageMeter(),
        "ssim": AverageMeter(),
        "l1": AverageMeter(),
    }
    for batch in dataloader:
        batch = tensor_to_device(batch, device)
        y = batch["input"]
        target = batch["target"]
        result = run_multistep_inference(model, x_t=y, y=y, num_steps=num_steps, clamp_range=(0.0, 1.0))
        pred = result.final_state
        metrics = compute_reconstruction_metrics(pred, target)
        batch_size = int(target.shape[0])
        for name, value in metrics.items():
            meters[name].update(value, n=batch_size)
    return {name: meter.avg for name, meter in meters.items()}


def config_to_dict(config: TrainConfig) -> dict[str, Any]:
    return {
        **{key: value for key, value in config.__dict__.items() if key != "schedule"},
        "schedule": asdict(config.schedule),
    }


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
