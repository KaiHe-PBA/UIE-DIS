from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch.amp import GradScaler

from uie_dis import RestormerLiteStudent
from uie_dis.data import UnderwaterImagePairDataset, build_train_val_dataloaders
from uie_dis.training import (
    ModelEMA,
    TrainConfig,
    create_loss_assembler,
    create_optimizer,
    create_scheduler,
    train_one_epoch,
    validate,
)


def _write_rgb_image(path: Path, seed: int, size: int = 48) -> None:
    rng = np.random.default_rng(seed)
    image = (rng.random((size, size, 3)) * 255.0).astype(np.uint8)
    path.write_bytes(
        b"P6\n%d %d\n255\n" % (size, size) + image.tobytes()
    )


def _build_tiny_model() -> RestormerLiteStudent:
    return RestormerLiteStudent(
        stage_dims=(16, 24, 32, 48),
        latent_dim=64,
        stage_blocks=(1, 1, 1, 1),
        decoder_blocks=(1, 1, 1, 1),
        encoder_heads=(1, 2, 4, 4),
        decoder_heads=(4, 4, 2, 1),
        latent_blocks=1,
        latent_heads=4,
        time_dim=64,
        ffn_expansion=1.5,
        enable_color_head=True,
        fusion_mode="gated",
    )


def test_paired_dataset_matches_by_stem_and_crops_to_patch():
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        input_dir = root / "input"
        target_dir = root / "target"
        input_dir.mkdir()
        target_dir.mkdir()
        _write_rgb_image(input_dir / "sample_a.ppm", seed=1)
        _write_rgb_image(target_dir / "sample_a.ppm", seed=2)

        dataset = UnderwaterImagePairDataset(input_dir, target_dir, patch_size=32, training=True)
        item = dataset[0]

        assert tuple(item["input"].shape) == (3, 32, 32)
        assert tuple(item["target"].shape) == (3, 32, 32)
        assert item["key"] == "sample_a"


def test_train_and_validate_one_epoch_end_to_end():
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        for split in ("train", "val"):
            for branch in ("input", "target"):
                (root / split / branch).mkdir(parents=True, exist_ok=True)
        for index in range(2):
            _write_rgb_image(root / "train" / "input" / f"img_{index}.ppm", seed=index)
            _write_rgb_image(root / "train" / "target" / f"img_{index}.ppm", seed=index + 10)
        _write_rgb_image(root / "val" / "input" / "img_0.ppm", seed=101)
        _write_rgb_image(root / "val" / "target" / "img_0.ppm", seed=111)

        train_loader, val_loader = build_train_val_dataloaders(
            train_input_dir=root / "train" / "input",
            train_target_dir=root / "train" / "target",
            val_input_dir=root / "val" / "input",
            val_target_dir=root / "val" / "target",
            train_patch_size=32,
            val_patch_size=None,
            batch_size=1,
            val_batch_size=1,
            num_workers=0,
            pin_memory=False,
            random_flip=False,
        )

        device = torch.device("cpu")
        student = _build_tiny_model().to(device)
        teacher = _build_tiny_model().to(device)
        teacher.load_state_dict(student.state_dict(), strict=True)
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)

        config = TrainConfig(
            epochs=1,
            batch_size=1,
            val_batch_size=1,
            num_workers=0,
            train_patch_size=32,
            val_patch_size=None,
            amp=False,
            loss_profile="memory_efficient",
            log_every=1000,
            validate_every=1,
            val_num_steps=1,
        )
        config.schedule.noise_std = 0.01

        optimizer = create_optimizer(student, config)
        scheduler = create_scheduler(optimizer, config)
        scaler = GradScaler("cpu", enabled=False)
        ema = ModelEMA(student, decay=0.9)
        loss_assembler, _ = create_loss_assembler(config)
        loss_assembler = loss_assembler.to(device)

        train_stats, global_step = train_one_epoch(
            epoch=1,
            student=student,
            teacher=teacher,
            dataloader=train_loader,
            optimizer=optimizer,
            loss_assembler=loss_assembler,
            scaler=scaler,
            ema=ema,
            device=device,
            config=config,
            global_step=0,
        )
        scheduler.step()
        with ema.apply_to(student):
            val_stats = validate(model=student, dataloader=val_loader, device=device, num_steps=1)

        assert global_step == len(train_loader)
        assert "loss" in train_stats
        assert train_stats["loss"] > 0.0
        assert set(("psnr", "ssim", "l1")).issubset(val_stats.keys())
        assert val_stats["psnr"] == val_stats["psnr"]
