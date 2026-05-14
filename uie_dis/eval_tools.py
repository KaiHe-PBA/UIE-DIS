from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from uie_consistency import run_multistep_inference, ssim

from .data import IMAGE_EXTENSIONS, _list_image_files, _load_rgb_image
from .models import RestormerLiteStudent
from .training import compute_psnr, maybe_autocast, tensor_to_device


@dataclass(frozen=True)
class ImageRecord:
    key: str
    path: Path


class ImageDirectoryDataset(Dataset[dict[str, Tensor | str]]):
    def __init__(self, input_dir: str | Path) -> None:
        super().__init__()
        self.input_dir = Path(input_dir)
        self.records = self._build_records()

    def _build_records(self) -> List[ImageRecord]:
        records = [
            ImageRecord(key=path.stem, path=path)
            for path in _list_image_files(self.input_dir)
            if path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if not records:
            raise RuntimeError(f"No supported images found in {self.input_dir}")
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        record = self.records[index]
        return {"input": _load_rgb_image(record.path), "key": record.key}


def _normalize_image_extension(image_ext: str) -> str:
    image_ext = image_ext.strip().lower()
    if not image_ext:
        raise ValueError("image_ext cannot be empty.")
    if not image_ext.startswith("."):
        image_ext = "." + image_ext
    return image_ext


def _write_ppm(path: Path, tensor: Tensor) -> Path:
    path = path.with_suffix(".ppm")
    path.parent.mkdir(parents=True, exist_ok=True)
    array = (
        tensor.detach()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    height, width, _ = array.shape
    with path.open("wb") as handle:
        handle.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        handle.write(array.tobytes())
    return path


def save_tensor_image(tensor: Tensor, path: str | Path) -> Path:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".ppm":
        return _write_ppm(path, tensor)

    array = (
        tensor.detach()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:  # pragma: no cover - runtime dependent
        raise ModuleNotFoundError(
            f"Saving {suffix or 'images'} requires Pillow. Use `--image-ext ppm` in minimal environments."
        ) from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)
    return path


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return path


def write_metrics_csv(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write("")
        return path
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: torch.device,
    use_ema: bool = True,
) -> tuple[RestormerLiteStudent, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_kwargs = checkpoint.get("model_kwargs")
    if model_kwargs is None:
        raise KeyError(f"Checkpoint {checkpoint_path} does not contain `model_kwargs`.")
    model = RestormerLiteStudent(**model_kwargs).to(device)

    if use_ema and "ema" in checkpoint and "shadow" in checkpoint["ema"]:
        state_dict = checkpoint["ema"]["shadow"]
        loaded_state = "ema.shadow"
    else:
        state_dict = checkpoint.get("student") or checkpoint.get("model") or checkpoint.get("state_dict")
        loaded_state = "student"

    if state_dict is None:
        raise KeyError(f"Checkpoint {checkpoint_path} does not contain usable model weights.")

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, {
        "checkpoint_path": str(checkpoint_path),
        "loaded_state": loaded_state,
        "epoch": checkpoint.get("epoch"),
        "best_metric": checkpoint.get("best_metric"),
        "model_kwargs": model_kwargs,
    }


def _compute_batch_metrics(pred: Tensor, target: Tensor) -> dict[str, Tensor]:
    return {
        "psnr": compute_psnr(pred, target),
        "ssim": ssim(pred, target),
        "l1": F.l1_loss(pred, target, reduction="none").mean(dim=(1, 2, 3)),
    }


def _batch_keys(batch: Mapping[str, Any]) -> List[str]:
    keys = batch["key"]
    if isinstance(keys, list):
        return [str(key) for key in keys]
    return [str(keys)]


@torch.no_grad()
def run_unpaired_inference(
    *,
    model: nn.Module,
    dataloader: DataLoader,
    output_dir: str | Path,
    device: torch.device,
    num_steps: int = 1,
    image_ext: str = ".png",
    amp: bool = True,
    save_input: bool = False,
    save_history: bool = False,
    clamp_range: tuple[float, float] = (0.0, 1.0),
) -> dict[str, Any]:
    image_ext = _normalize_image_extension(image_ext)
    output_dir = Path(output_dir)
    pred_dir = output_dir / "pred"
    input_dir = output_dir / "input"
    history_dir = output_dir / "history"

    exported_rows: List[dict[str, Any]] = []
    model.eval()

    for batch in dataloader:
        batch = tensor_to_device(batch, device)
        y = batch["input"]
        keys = _batch_keys(batch)

        with maybe_autocast(device, amp):
            result = run_multistep_inference(
                model,
                x_t=y,
                y=y,
                num_steps=num_steps,
                clamp_range=clamp_range,
                return_history=save_history,
            )

        for sample_index, key in enumerate(keys):
            pred_path = save_tensor_image(result.final_state[sample_index], pred_dir / f"{key}{image_ext}")
            row = {"key": key, "pred_path": str(pred_path)}
            if save_input:
                input_path = save_tensor_image(y[sample_index], input_dir / f"{key}{image_ext}")
                row["input_path"] = str(input_path)
            if save_history:
                sample_history_dir = history_dir / key
                for step in result.history:
                    step_index = int(step["step_index"].item())
                    save_tensor_image(step["state"][sample_index], sample_history_dir / f"step_{step_index}{image_ext}")
            exported_rows.append(row)

    summary = {
        "num_samples": len(exported_rows),
        "num_steps": num_steps,
        "image_ext": image_ext,
        "output_dir": str(output_dir),
        "samples": exported_rows,
    }
    write_json(output_dir / "inference_summary.json", summary)
    return summary


@torch.no_grad()
def run_paired_evaluation(
    *,
    model: nn.Module,
    dataloader: DataLoader,
    output_dir: str | Path,
    device: torch.device,
    num_steps: int = 1,
    image_ext: str = ".png",
    amp: bool = True,
    export_images: bool = True,
    save_input: bool = False,
    save_target: bool = False,
    save_history: bool = False,
    clamp_range: tuple[float, float] = (0.0, 1.0),
) -> dict[str, Any]:
    image_ext = _normalize_image_extension(image_ext)
    output_dir = Path(output_dir)
    pred_dir = output_dir / "pred"
    input_dir = output_dir / "input"
    target_dir = output_dir / "target"
    history_dir = output_dir / "history"

    rows: List[dict[str, Any]] = []
    metric_sums = {"psnr": 0.0, "ssim": 0.0, "l1": 0.0}
    metric_count = 0

    model.eval()
    for batch in dataloader:
        batch = tensor_to_device(batch, device)
        y = batch["input"]
        target = batch["target"]
        keys = _batch_keys(batch)

        with maybe_autocast(device, amp):
            result = run_multistep_inference(
                model,
                x_t=y,
                y=y,
                num_steps=num_steps,
                clamp_range=clamp_range,
                return_history=save_history,
            )
        pred = result.final_state
        metric_tensors = _compute_batch_metrics(pred, target)

        for sample_index, key in enumerate(keys):
            row: dict[str, Any] = {
                "key": key,
                "psnr": float(metric_tensors["psnr"][sample_index].item()),
                "ssim": float(metric_tensors["ssim"][sample_index].item()),
                "l1": float(metric_tensors["l1"][sample_index].item()),
            }
            if export_images:
                pred_path = save_tensor_image(pred[sample_index], pred_dir / f"{key}{image_ext}")
                row["pred_path"] = str(pred_path)
            if save_input:
                input_path = save_tensor_image(y[sample_index], input_dir / f"{key}{image_ext}")
                row["input_path"] = str(input_path)
            if save_target:
                target_path = save_tensor_image(target[sample_index], target_dir / f"{key}{image_ext}")
                row["target_path"] = str(target_path)
            if save_history:
                sample_history_dir = history_dir / key
                for step in result.history:
                    step_index = int(step["step_index"].item())
                    save_tensor_image(step["state"][sample_index], sample_history_dir / f"step_{step_index}{image_ext}")
            rows.append(row)
            for metric_name in metric_sums:
                metric_sums[metric_name] += row[metric_name]
            metric_count += 1

    summary = {
        "num_samples": metric_count,
        "num_steps": num_steps,
        "image_ext": image_ext,
        "metrics": {
            metric_name: (metric_sums[metric_name] / metric_count if metric_count > 0 else 0.0)
            for metric_name in metric_sums
        },
        "rows": rows,
    }
    write_json(output_dir / "metrics_summary.json", summary)
    write_metrics_csv(output_dir / "metrics_per_image.csv", rows)
    return summary

