from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.amp import GradScaler

from uie_dis.data import build_train_val_dataloaders, summarize_dataset
from uie_dis.training import (
    ModelEMA,
    TrainConfig,
    build_model_kwargs_from_config,
    build_student_model,
    config_to_dict,
    create_loss_assembler,
    create_optimizer,
    create_scheduler,
    load_teacher_model,
    load_training_checkpoint,
    resolve_device,
    set_seed,
    train_one_epoch,
    validate,
    write_json,
)


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_bool(value: str) -> bool:
    normalized = value.lower().strip()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value: {value}")


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train underwater consistency distillation student model.")
    parser.add_argument("--train-input-dir", default="/data/train/input")
    parser.add_argument("--train-target-dir", default="/data/train/target")
    parser.add_argument("--val-input-dir", default="/data/val/input")
    parser.add_argument("--val-target-dir", default="/data/val/target")
    parser.add_argument("--output-dir", default="/workspace/outputs/consistency_student")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--val-batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--train-patch-size", type=int, default=256)
    parser.add_argument("--val-patch-size", type=int, default=None)
    parser.add_argument("--random-flip", type=parse_bool, default=True)
    parser.add_argument("--center-crop-validation", type=parse_bool, default=False)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--amp", type=parse_bool, default=True)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--ema-update-every", type=int, default=1)
    parser.add_argument("--validate-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--val-num-steps", type=int, default=1)
    parser.add_argument(
        "--loss-profile",
        default="balanced",
        choices=["balanced", "stability_first", "memory_efficient", "quality_first", "fast_train"],
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--scheduler-min-lr", type=float, default=1e-6)
    parser.add_argument("--teacher-checkpoint", default=None)
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--teacher-use-checkpoint-kwargs", type=parse_bool, default=True)
    parser.add_argument("--min-t", type=float, default=0.0)
    parser.add_argument("--max-t", type=float, default=1.0)
    parser.add_argument("--t-power", type=float, default=1.0)
    parser.add_argument("--noise-std", type=float, default=0.03)
    parser.add_argument("--clamp-min", type=float, default=0.0)
    parser.add_argument("--clamp-max", type=float, default=1.0)
    parser.add_argument("--stage-dims", type=parse_int_list, default=parse_int_list("48,72,96,128"))
    parser.add_argument("--latent-dim", type=int, default=160)
    parser.add_argument("--stage-blocks", type=parse_int_list, default=parse_int_list("1,1,2,2"))
    parser.add_argument("--decoder-blocks", type=parse_int_list, default=parse_int_list("2,2,1,1"))
    parser.add_argument("--encoder-heads", type=parse_int_list, default=parse_int_list("1,2,3,4"))
    parser.add_argument("--decoder-heads", type=parse_int_list, default=parse_int_list("4,3,2,1"))
    parser.add_argument("--latent-blocks", type=int, default=3)
    parser.add_argument("--latent-heads", type=int, default=5)
    parser.add_argument("--time-dim", type=int, default=256)
    parser.add_argument("--ffn-expansion", type=float, default=2.0)
    parser.add_argument("--enable-color-head", type=parse_bool, default=True)
    parser.add_argument("--fusion-mode", default="gated", choices=["gated", "cross-attn", "gated-cross-attn"])
    return parser


def build_train_config(args: argparse.Namespace) -> TrainConfig:
    config = TrainConfig(
        output_dir=args.output_dir,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        num_workers=args.num_workers,
        train_patch_size=args.train_patch_size,
        val_patch_size=args.val_patch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        amp=args.amp,
        ema_decay=args.ema_decay,
        ema_update_every=args.ema_update_every,
        validate_every=args.validate_every,
        save_every=args.save_every,
        log_every=args.log_every,
        val_num_steps=args.val_num_steps,
        loss_profile=args.loss_profile,
        random_flip=args.random_flip,
        center_crop_validation=args.center_crop_validation,
        device=args.device,
        scheduler_min_lr=args.scheduler_min_lr,
        teacher_checkpoint=args.teacher_checkpoint,
        resume_checkpoint=args.resume_checkpoint,
        teacher_use_checkpoint_kwargs=args.teacher_use_checkpoint_kwargs,
    )
    config.schedule.min_t = args.min_t
    config.schedule.max_t = args.max_t
    config.schedule.t_power = args.t_power
    config.schedule.noise_std = args.noise_std
    config.schedule.clamp_range = (args.clamp_min, args.clamp_max)
    return config


def main() -> None:
    parser = create_arg_parser()
    args = parser.parse_args()
    config = build_train_config(args)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(config.seed)
    device = resolve_device(config.device)
    print(f"Using device: {device}")

    train_loader, val_loader = build_train_val_dataloaders(
        train_input_dir=args.train_input_dir,
        train_target_dir=args.train_target_dir,
        val_input_dir=args.val_input_dir,
        val_target_dir=args.val_target_dir,
        train_patch_size=config.train_patch_size,
        val_patch_size=config.val_patch_size,
        batch_size=config.batch_size,
        val_batch_size=config.val_batch_size,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        random_flip=config.random_flip,
        center_crop_validation=config.center_crop_validation,
    )
    print("Train dataset:", summarize_dataset(train_loader.dataset.records))
    print("Val dataset:", summarize_dataset(val_loader.dataset.records))

    model_kwargs = build_model_kwargs_from_config(args)
    student = build_student_model(model_kwargs, device)
    teacher, teacher_metadata = load_teacher_model(
        device=device,
        student_model_kwargs=model_kwargs,
        teacher_checkpoint=config.teacher_checkpoint,
        teacher_use_checkpoint_kwargs=config.teacher_use_checkpoint_kwargs,
    )
    if config.teacher_checkpoint is None:
        teacher = None
        print(
            "[Warn] `--teacher-checkpoint` 未提供，当前训练将关闭 consistency distillation，"
            "仅使用监督重建损失训练学生模型。"
        )
    print("Teacher metadata:", teacher_metadata)

    optimizer = create_optimizer(student, config)
    scheduler = create_scheduler(optimizer, config)
    scaler = GradScaler(device.type, enabled=config.amp and device.type == "cuda")
    ema = ModelEMA(student, decay=config.ema_decay)
    loss_assembler, loss_metadata = create_loss_assembler(config)
    loss_assembler = loss_assembler.to(device)
    if teacher is None and loss_assembler.weights.consistency > 0:
        loss_assembler.weights.consistency = 0.0
        loss_metadata["weights"]["consistency"] = 0.0
        loss_metadata["enabled_losses"] = tuple(
            name for name in loss_metadata["enabled_losses"] if name != "consistency"
        )
        if "consistency" not in loss_metadata["disabled_losses"]:
            loss_metadata["disabled_losses"] = tuple(loss_metadata["disabled_losses"]) + ("consistency",)

    start_epoch = 1
    global_step = 0
    best_psnr = float("-inf")

    if config.resume_checkpoint is not None:
        checkpoint = load_training_checkpoint(
            config.resume_checkpoint,
            device=device,
            student=student,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            ema=ema,
        )
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint.get("global_step", 0))
        best_psnr = float(checkpoint.get("best_metric", float("-inf")))
        print(f"Resumed from {config.resume_checkpoint} at epoch {start_epoch}")

    write_json(
        output_dir / "train_config.json",
        {
            "train_config": config_to_dict(config),
            "model_kwargs": model_kwargs,
            "teacher_metadata": teacher_metadata,
            "loss_metadata": loss_metadata,
        },
    )

    history: list[dict[str, object]] = []
    for epoch in range(start_epoch, config.epochs + 1):
        train_stats, global_step = train_one_epoch(
            epoch=epoch,
            student=student,
            teacher=teacher,
            dataloader=train_loader,
            optimizer=optimizer,
            loss_assembler=loss_assembler,
            scaler=scaler,
            ema=ema,
            device=device,
            config=config,
            global_step=global_step,
        )
        scheduler.step()

        epoch_summary: dict[str, object] = {"epoch": epoch, "train": train_stats}

        if epoch % config.validate_every == 0:
            with ema.apply_to(student):
                val_stats = validate(model=student, dataloader=val_loader, device=device, num_steps=config.val_num_steps)
            epoch_summary["val"] = val_stats
            print(
                f"[Val] epoch={epoch} psnr={val_stats['psnr']:.3f} "
                f"ssim={val_stats['ssim']:.4f} l1={val_stats['l1']:.4f}"
            )
            if val_stats["psnr"] > best_psnr:
                best_psnr = val_stats["psnr"]
                torch.save(
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "student": student.state_dict(),
                        "ema": ema.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict(),
                        "model_kwargs": model_kwargs,
                        "best_metric": best_psnr,
                    },
                    output_dir / "checkpoint_best.pth",
                )

        history.append(epoch_summary)
        write_json(output_dir / "history.json", {"history": history, "best_psnr": best_psnr})

        if epoch % config.save_every == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "student": student.state_dict(),
                    "ema": ema.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "model_kwargs": model_kwargs,
                    "best_metric": best_psnr,
                },
                output_dir / "checkpoint_latest.pth",
            )

    print(f"Training finished. Best PSNR: {best_psnr:.3f}")


if __name__ == "__main__":
    main()
