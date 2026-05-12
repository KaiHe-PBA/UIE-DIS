from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
import random


IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}


def list_images(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob('*') if p.suffix.lower() in IMG_EXTS])


class PairedUnderwaterDataset(Dataset):
    """
    Expected layout:
      data_root/
        train/
          input/xxx.png
          target/xxx.png
        val/
          input/xxx.png
          target/xxx.png
    File names under input/target should match.
    """

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        patch_size: int = 256,
        augment: bool = True,
    ):
        self.data_root = Path(data_root)
        self.input_dir = self.data_root / split / 'input'
        self.target_dir = self.data_root / split / 'target'
        if not self.input_dir.exists() or not self.target_dir.exists():
            raise FileNotFoundError(f'Missing dataset folders: {self.input_dir} or {self.target_dir}')

        self.input_paths = list_images(self.input_dir)
        if not self.input_paths:
            raise RuntimeError(f'No images found in {self.input_dir}')

        self.pairs: List[Tuple[Path, Path]] = []
        for input_path in self.input_paths:
            rel = input_path.relative_to(self.input_dir)
            target_path = self.target_dir / rel
            if not target_path.exists():
                raise FileNotFoundError(f'Missing target for {input_path}: {target_path}')
            self.pairs.append((input_path, target_path))

        self.patch_size = patch_size
        self.augment = augment and split == 'train'

    def __len__(self) -> int:
        return len(self.pairs)

    def _load_pair(self, index: int):
        input_path, target_path = self.pairs[index]
        y = Image.open(input_path).convert('RGB')
        x_gt = Image.open(target_path).convert('RGB')
        return y, x_gt, str(input_path), str(target_path)

    def _paired_random_crop(self, y: Image.Image, x_gt: Image.Image):
        w, h = y.size
        if w < self.patch_size or h < self.patch_size:
            new_w = max(w, self.patch_size)
            new_h = max(h, self.patch_size)
            y = y.resize((new_w, new_h), Image.BICUBIC)
            x_gt = x_gt.resize((new_w, new_h), Image.BICUBIC)
            w, h = y.size
        left = random.randint(0, w - self.patch_size)
        top = random.randint(0, h - self.patch_size)
        y = TF.crop(y, top, left, self.patch_size, self.patch_size)
        x_gt = TF.crop(x_gt, top, left, self.patch_size, self.patch_size)
        return y, x_gt

    def _paired_aug(self, y: Image.Image, x_gt: Image.Image):
        if random.random() < 0.5:
            y = TF.hflip(y)
            x_gt = TF.hflip(x_gt)
        if random.random() < 0.5:
            y = TF.vflip(y)
            x_gt = TF.vflip(x_gt)
        rot_k = random.randint(0, 3)
        if rot_k > 0:
            angle = 90 * rot_k
            y = TF.rotate(y, angle)
            x_gt = TF.rotate(x_gt, angle)
        return y, x_gt

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        y, x_gt, input_path, target_path = self._load_pair(index)
        if self.augment:
            y, x_gt = self._paired_random_crop(y, x_gt)
            y, x_gt = self._paired_aug(y, x_gt)
        else:
            y = TF.center_crop(y, [min(y.height, self.patch_size), min(y.width, self.patch_size)])
            x_gt = TF.center_crop(x_gt, [min(x_gt.height, self.patch_size), min(x_gt.width, self.patch_size)])
        y = TF.to_tensor(y)
        x_gt = TF.to_tensor(x_gt)
        return {'y': y, 'x_gt': x_gt, 'input_path': input_path, 'target_path': target_path}
