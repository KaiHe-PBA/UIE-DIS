from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image


def load_image_tensor(path: str, image_size: Optional[int] = None) -> torch.Tensor:
    image = Image.open(path).convert('RGB')
    if image_size is not None:
        image = image.resize((image_size, image_size), Image.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous().clamp(0.0, 1.0)


def save_image_tensor(tensor: torch.Tensor, path: str) -> None:
    image = tensor.detach().cpu().clamp(0.0, 1.0)
    if image.dim() == 4:
        if image.shape[0] != 1:
            raise ValueError('只支持保存单张图像或 batch size 为 1 的张量')
        image = image[0]
    array = (image.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(output_path)
