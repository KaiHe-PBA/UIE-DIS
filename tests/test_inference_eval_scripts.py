from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch.utils.data import DataLoader

from uie_dis.data import UnderwaterImagePairDataset, build_dataloader
from uie_dis.eval_tools import (
    ImageDirectoryDataset,
    load_model_from_checkpoint,
    run_paired_evaluation,
    run_unpaired_inference,
)
from uie_dis.models import RestormerLiteStudent
from uie_dis.training import ModelEMA


def _write_rgb_ppm(path: Path, seed: int, size: int = 48) -> None:
    rng = np.random.default_rng(seed)
    image = (rng.random((size, size, 3)) * 255.0).astype(np.uint8)
    path.write_bytes(b"P6\n%d %d\n255\n" % (size, size) + image.tobytes())


def _build_tiny_model_kwargs() -> dict[str, object]:
    return {
        "stage_dims": (16, 24, 32, 48),
        "latent_dim": 64,
        "stage_blocks": (1, 1, 1, 1),
        "decoder_blocks": (1, 1, 1, 1),
        "encoder_heads": (1, 2, 4, 4),
        "decoder_heads": (4, 4, 2, 1),
        "latent_blocks": 1,
        "latent_heads": 4,
        "time_dim": 64,
        "ffn_expansion": 1.5,
        "enable_color_head": True,
        "fusion_mode": "gated",
    }


def _save_checkpoint(path: Path) -> Path:
    model_kwargs = _build_tiny_model_kwargs()
    model = RestormerLiteStudent(**model_kwargs)
    ema = ModelEMA(model, decay=0.9)
    torch.save(
        {
            "epoch": 3,
            "student": model.state_dict(),
            "ema": ema.state_dict(),
            "model_kwargs": model_kwargs,
            "best_metric": 25.0,
        },
        path,
    )
    return path


def test_eval_tools_export_predictions_and_metrics():
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        checkpoint_path = _save_checkpoint(root / "checkpoint_best.pth")
        for split in ("input", "target"):
            (root / split).mkdir(parents=True, exist_ok=True)
        for index in range(2):
            _write_rgb_ppm(root / "input" / f"sample_{index}.ppm", seed=index)
            _write_rgb_ppm(root / "target" / f"sample_{index}.ppm", seed=index + 100)

        model, metadata = load_model_from_checkpoint(checkpoint_path, device=torch.device("cpu"), use_ema=True)
        dataset = UnderwaterImagePairDataset(root / "input", root / "target", patch_size=None, training=False)
        dataloader = build_dataloader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
        )

        summary = run_paired_evaluation(
            model=model,
            dataloader=dataloader,
            output_dir=root / "eval_outputs",
            device=torch.device("cpu"),
            num_steps=1,
            image_ext="ppm",
            amp=False,
            export_images=True,
            save_input=True,
            save_target=True,
        )

        assert metadata["loaded_state"] == "ema.shadow"
        assert summary["num_samples"] == 2
        assert set(("psnr", "ssim", "l1")).issubset(summary["metrics"].keys())
        assert (root / "eval_outputs" / "metrics_summary.json").exists()
        assert (root / "eval_outputs" / "metrics_per_image.csv").exists()
        assert (root / "eval_outputs" / "pred" / "sample_0.ppm").exists()
        assert (root / "eval_outputs" / "input" / "sample_0.ppm").exists()
        assert (root / "eval_outputs" / "target" / "sample_0.ppm").exists()


def test_unpaired_inference_exports_predictions():
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        checkpoint_path = _save_checkpoint(root / "checkpoint_latest.pth")
        input_dir = root / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        _write_rgb_ppm(input_dir / "demo.ppm", seed=999)

        model, _ = load_model_from_checkpoint(checkpoint_path, device=torch.device("cpu"), use_ema=False)
        dataset = ImageDirectoryDataset(input_dir)
        dataloader = build_dataloader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
        )
        summary = run_unpaired_inference(
            model=model,
            dataloader=dataloader,
            output_dir=root / "inference_outputs",
            device=torch.device("cpu"),
            num_steps=2,
            image_ext="ppm",
            amp=False,
            save_input=True,
            save_history=True,
        )

        assert summary["num_samples"] == 1
        assert (root / "inference_outputs" / "pred" / "demo.ppm").exists()
        assert (root / "inference_outputs" / "input" / "demo.ppm").exists()
        assert (root / "inference_outputs" / "history" / "demo" / "step_1.ppm").exists()
        assert (root / "inference_outputs" / "history" / "demo" / "step_2.ppm").exists()
