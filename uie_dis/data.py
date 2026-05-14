from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".ppm"}


def _list_image_files(directory: Path) -> List[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Image directory does not exist: {directory}")
    files = [path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(files)


def _image_array_to_tensor(array: np.ndarray) -> Tensor:
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    if array.shape[2] == 4:
        array = array[..., :3]
    tensor = torch.from_numpy(array.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
    return tensor


def _read_ppm(path: Path) -> Tensor:
    with path.open("rb") as handle:
        magic = handle.readline().strip()
        if magic != b"P6":
            raise ValueError(f"Only binary P6 PPM is supported for fallback loading, got {magic!r} in {path}")

        header_tokens: list[bytes] = []
        while len(header_tokens) < 3:
            line = handle.readline()
            if not line:
                raise ValueError(f"Incomplete PPM header: {path}")
            line = line.split(b"#", 1)[0].strip()
            if not line:
                continue
            header_tokens.extend(line.split())

        width, height, max_value = (int(token) for token in header_tokens[:3])
        if max_value != 255:
            raise ValueError(f"Unsupported PPM max value {max_value} in {path}; expected 255")

        raw = handle.read(width * height * 3)
        array = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
        return _image_array_to_tensor(array)


def _image_to_tensor(image: Any) -> Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    if array.shape[2] == 4:
        array = array[..., :3]
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return tensor


def _load_rgb_image(path: Path) -> Tensor:
    if path.suffix.lower() == ".ppm":
        return _read_ppm(path)
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on environment
        raise ModuleNotFoundError(
            "Loading png/jpg/bmp/tiff/webp images requires Pillow. "
            "Install Pillow or use .ppm files in environments without image libraries."
        ) from exc
    with Image.open(path) as image:
        return _image_to_tensor(image.convert("RGB"))


def _crop_pair(input_tensor: Tensor, target_tensor: Tensor, top: int, left: int, patch_size: int) -> tuple[Tensor, Tensor]:
    bottom = top + patch_size
    right = left + patch_size
    return input_tensor[:, top:bottom, left:right], target_tensor[:, top:bottom, left:right]


def _resize_to_minimum(input_tensor: Tensor, target_tensor: Tensor, minimum_size: int) -> tuple[Tensor, Tensor]:
    _, h, w = input_tensor.shape
    if h >= minimum_size and w >= minimum_size:
        return input_tensor, target_tensor
    scale = max(minimum_size / max(h, 1), minimum_size / max(w, 1))
    new_h = max(int(round(h * scale)), minimum_size)
    new_w = max(int(round(w * scale)), minimum_size)
    input_resized = torch.nn.functional.interpolate(
        input_tensor.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False
    ).squeeze(0)
    target_resized = torch.nn.functional.interpolate(
        target_tensor.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False
    ).squeeze(0)
    return input_resized, target_resized


@dataclass(frozen=True)
class ImagePairRecord:
    key: str
    input_path: Path
    target_path: Path


class UnderwaterImagePairDataset(Dataset[dict[str, Tensor | str]]):
    def __init__(
        self,
        input_dir: str | Path,
        target_dir: str | Path,
        *,
        patch_size: int | None = 256,
        training: bool = True,
        random_flip: bool = True,
        center_crop_validation: bool = False,
    ) -> None:
        super().__init__()
        self.input_dir = Path(input_dir)
        self.target_dir = Path(target_dir)
        self.patch_size = patch_size
        self.training = training
        self.random_flip = random_flip
        self.center_crop_validation = center_crop_validation
        self.records = self._build_records()

    def _build_records(self) -> List[ImagePairRecord]:
        input_files = _list_image_files(self.input_dir)
        target_files = _list_image_files(self.target_dir)
        target_by_stem = {path.stem: path for path in target_files}

        records: List[ImagePairRecord] = []
        missing_targets: List[str] = []
        for input_path in input_files:
            target_path = target_by_stem.get(input_path.stem)
            if target_path is None:
                missing_targets.append(input_path.name)
                continue
            records.append(ImagePairRecord(key=input_path.stem, input_path=input_path, target_path=target_path))

        if not records:
            raise RuntimeError(
                f"No paired samples found between {self.input_dir} and {self.target_dir}. "
                "Files are matched by basename stem."
            )
        if missing_targets:
            raise RuntimeError(
                "Some input images do not have matching targets by basename stem: "
                + ", ".join(sorted(missing_targets)[:10])
            )
        return records

    def __len__(self) -> int:
        return len(self.records)

    def _apply_patch_sampling(self, input_tensor: Tensor, target_tensor: Tensor) -> tuple[Tensor, Tensor]:
        if self.patch_size is None:
            return input_tensor, target_tensor

        input_tensor, target_tensor = _resize_to_minimum(input_tensor, target_tensor, self.patch_size)
        _, h, w = input_tensor.shape
        if h == self.patch_size and w == self.patch_size:
            return input_tensor, target_tensor

        if self.training:
            top = random.randint(0, h - self.patch_size)
            left = random.randint(0, w - self.patch_size)
        elif self.center_crop_validation:
            top = max((h - self.patch_size) // 2, 0)
            left = max((w - self.patch_size) // 2, 0)
        else:
            return input_tensor, target_tensor

        return _crop_pair(input_tensor, target_tensor, top, left, self.patch_size)

    def _apply_augmentation(self, input_tensor: Tensor, target_tensor: Tensor) -> tuple[Tensor, Tensor]:
        if not self.training or not self.random_flip:
            return input_tensor, target_tensor
        if random.random() < 0.5:
            input_tensor = torch.flip(input_tensor, dims=[2])
            target_tensor = torch.flip(target_tensor, dims=[2])
        if random.random() < 0.5:
            input_tensor = torch.flip(input_tensor, dims=[1])
            target_tensor = torch.flip(target_tensor, dims=[1])
        if random.random() < 0.5:
            input_tensor = input_tensor.transpose(1, 2).contiguous()
            target_tensor = target_tensor.transpose(1, 2).contiguous()
        return input_tensor, target_tensor

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        record = self.records[index]
        input_tensor = _load_rgb_image(record.input_path)
        target_tensor = _load_rgb_image(record.target_path)

        if input_tensor.shape != target_tensor.shape:
            raise ValueError(
                f"Mismatched image shapes for pair {record.key}: "
                f"{tuple(input_tensor.shape)} vs {tuple(target_tensor.shape)}"
            )

        input_tensor, target_tensor = self._apply_patch_sampling(input_tensor, target_tensor)
        input_tensor, target_tensor = self._apply_augmentation(input_tensor, target_tensor)
        return {
            "input": input_tensor,
            "target": target_tensor,
            "key": record.key,
        }


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataloader(
    dataset: Dataset[dict[str, Tensor | str]],
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    drop_last: bool,
    persistent_workers: bool | None = None,
) -> DataLoader:
    if persistent_workers is None:
        persistent_workers = num_workers > 0
    generator = torch.Generator()
    generator.manual_seed(42)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def build_train_val_dataloaders(
    *,
    train_input_dir: str | Path,
    train_target_dir: str | Path,
    val_input_dir: str | Path,
    val_target_dir: str | Path,
    train_patch_size: int = 256,
    val_patch_size: int | None = None,
    batch_size: int = 4,
    val_batch_size: int = 1,
    num_workers: int = 4,
    pin_memory: bool = True,
    random_flip: bool = True,
    center_crop_validation: bool = False,
) -> tuple[DataLoader, DataLoader]:
    train_dataset = UnderwaterImagePairDataset(
        train_input_dir,
        train_target_dir,
        patch_size=train_patch_size,
        training=True,
        random_flip=random_flip,
    )
    val_dataset = UnderwaterImagePairDataset(
        val_input_dir,
        val_target_dir,
        patch_size=val_patch_size,
        training=False,
        random_flip=False,
        center_crop_validation=center_crop_validation,
    )
    train_loader = build_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = build_dataloader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, val_loader


def summarize_dataset(records: Sequence[ImagePairRecord]) -> dict[str, object]:
    return {
        "num_samples": len(records),
        "first_keys": [record.key for record in records[:5]],
    }
