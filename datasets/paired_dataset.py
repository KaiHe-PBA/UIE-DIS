import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}


def _is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def _list_image_files(directory: str) -> List[Path]:
    root = Path(directory)
    if not root.exists():
        raise FileNotFoundError(f'目录不存在: {directory}')
    files = [p for p in root.rglob('*') if _is_image_file(p)]
    files.sort()
    if not files:
        raise RuntimeError(f'目录中未找到图像: {directory}')
    return files


def _stem_to_path_map(files: List[Path]) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for path in files:
        relative_stem = str(path.with_suffix('')).replace('\\', '/').split('/', 1)[-1]
        mapping[relative_stem] = path
        mapping[path.stem] = path
    return mapping


def pair_image_paths(input_dir: str, target_dir: str) -> List[Tuple[Path, Path]]:
    input_files = _list_image_files(input_dir)
    target_files = _list_image_files(target_dir)
    target_map = _stem_to_path_map(target_files)

    pairs: List[Tuple[Path, Path]] = []
    for input_path in input_files:
        relative_stem = str(input_path.relative_to(input_dir).with_suffix('')).replace('\\', '/')
        target_path = target_map.get(relative_stem) or target_map.get(input_path.stem)
        if target_path is None:
            raise RuntimeError(f'未找到配对标签: {input_path}')
        pairs.append((input_path, target_path))
    return pairs


def load_image(path: Path, image_size: Optional[int] = None) -> Image.Image:
    image = Image.open(path).convert('RGB')
    if image_size is not None:
        image = image.resize((image_size, image_size), Image.BICUBIC)
    return image


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    if array.ndim == 2:
        array = np.stack([array, array, array], axis=-1)
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return tensor.clamp(0.0, 1.0)


def pad_to_min_size(tensor: torch.Tensor, min_size: int) -> torch.Tensor:
    _, height, width = tensor.shape
    if min(height, width) >= min_size:
        return tensor
    pad_h = max(min_size - height, 0)
    pad_w = max(min_size - width, 0)
    return torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode='reflect')


def crop_tensor(tensor: torch.Tensor, top: int, left: int, crop_size: int) -> torch.Tensor:
    return tensor[:, top:top + crop_size, left:left + crop_size]


def apply_augmentation(
    tensor: torch.Tensor,
    flip_horizontal: bool,
    flip_vertical: bool,
    transpose_hw: bool,
) -> torch.Tensor:
    if flip_horizontal:
        tensor = torch.flip(tensor, dims=[2])
    if flip_vertical:
        tensor = torch.flip(tensor, dims=[1])
    if transpose_hw:
        tensor = tensor.transpose(1, 2)
    return tensor.contiguous()


def sample_time_value(num_steps: int, mode: str = 'uniform') -> Tuple[int, float]:
    if num_steps <= 1:
        return 0, 0.0
    if mode == 'quadratic':
        value = random.random() ** 2
        step = min(int(value * (num_steps - 1)), num_steps - 1)
    else:
        step = random.randint(0, num_steps - 1)
    normalized = step / float(num_steps - 1)
    return step, normalized


class PairedImageDataset(Dataset):
    def __init__(
        self,
        input_dir: str,
        target_dir: str,
        image_size: Optional[int] = None,
        crop_size: Optional[int] = None,
        augment: bool = False,
        num_time_steps: int = 1000,
        sigma_min: float = 0.0,
        sigma_max: float = 0.2,
        time_sampling: str = 'uniform',
        x_t_source: str = 'target',
        teacher_dir: Optional[str] = None,
    ) -> None:
        self.pairs = pair_image_paths(input_dir, target_dir)
        self.image_size = image_size
        self.crop_size = crop_size
        self.augment = augment
        self.num_time_steps = num_time_steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.time_sampling = time_sampling
        self.x_t_source = x_t_source
        self.teacher_dir = Path(teacher_dir) if teacher_dir else None

    def __len__(self) -> int:
        return len(self.pairs)

    def _load_teacher_tensor(self, input_path: Path) -> Optional[torch.Tensor]:
        if self.teacher_dir is None:
            return None
        candidate_base = self.teacher_dir / input_path.stem
        for suffix in ['.pt', '.pth', '.png', '.jpg', '.jpeg']:
            candidate = candidate_base.with_suffix(suffix)
            if not candidate.exists():
                continue
            if candidate.suffix in {'.pt', '.pth'}:
                tensor = torch.load(candidate, map_location='cpu')
                if isinstance(tensor, dict):
                    for key in ['x0', 'pred', 'prediction', 'image']:
                        if key in tensor:
                            tensor = tensor[key]
                            break
                if not isinstance(tensor, torch.Tensor):
                    raise TypeError(f'teacher 文件格式不支持: {candidate}')
                return tensor.float().clamp(0.0, 1.0)
            return image_to_tensor(load_image(candidate, self.image_size))
        return None

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        input_path, target_path = self.pairs[index]
        input_image = load_image(input_path, self.image_size)
        target_image = load_image(target_path, self.image_size)
        input_tensor = image_to_tensor(input_image)
        target_tensor = image_to_tensor(target_image)
        teacher_tensor = self._load_teacher_tensor(input_path)

        if self.crop_size is not None:
            input_tensor = pad_to_min_size(input_tensor, self.crop_size)
            target_tensor = pad_to_min_size(target_tensor, self.crop_size)
            if teacher_tensor is not None:
                teacher_tensor = pad_to_min_size(teacher_tensor, self.crop_size)
            _, height, width = input_tensor.shape
            top = random.randint(0, height - self.crop_size)
            left = random.randint(0, width - self.crop_size)
            input_tensor = crop_tensor(input_tensor, top, left, self.crop_size)
            target_tensor = crop_tensor(target_tensor, top, left, self.crop_size)
            if teacher_tensor is not None:
                teacher_tensor = crop_tensor(teacher_tensor, top, left, self.crop_size)
        if self.augment:
            flip_horizontal = random.random() < 0.5
            flip_vertical = random.random() < 0.5
            transpose_hw = random.random() < 0.5
            input_tensor = apply_augmentation(input_tensor, flip_horizontal, flip_vertical, transpose_hw)
            target_tensor = apply_augmentation(target_tensor, flip_horizontal, flip_vertical, transpose_hw)
            if teacher_tensor is not None:
                teacher_tensor = apply_augmentation(teacher_tensor, flip_horizontal, flip_vertical, transpose_hw)

        step, time_value = sample_time_value(self.num_time_steps, self.time_sampling)
        sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * time_value
        sigma = float(max(self.sigma_min, min(self.sigma_max, sigma)))
        base_tensor = target_tensor if self.x_t_source == 'target' else input_tensor
        noise = torch.randn_like(base_tensor) * sigma
        x_t = (base_tensor + noise).clamp(0.0, 1.0)

        sample: Dict[str, torch.Tensor] = {
            'input': input_tensor,
            'target': target_tensor,
            'x_t': x_t,
            't': torch.tensor(time_value, dtype=torch.float32),
            'step': torch.tensor(step, dtype=torch.long),
            'sigma': torch.tensor(sigma, dtype=torch.float32),
        }
        if teacher_tensor is not None:
            sample['teacher_target'] = teacher_tensor
        sample['input_path'] = str(input_path)
        sample['target_path'] = str(target_path)
        return sample


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 4,
    drop_last: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )
