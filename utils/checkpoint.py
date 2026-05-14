from pathlib import Path
from typing import Any, Dict, Optional

import torch


def save_checkpoint(
    save_dir: str,
    name: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    epoch: int = 0,
    step: int = 0,
    metrics: Optional[Dict[str, float]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> str:
    path = Path(save_dir)
    path.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        'model': model.state_dict(),
        'epoch': epoch,
        'step': step,
        'metrics': metrics or {},
        'config': config or {},
    }
    if optimizer is not None:
        checkpoint['optimizer'] = optimizer.state_dict()
    if scaler is not None:
        checkpoint['scaler'] = scaler.state_dict()
    file_path = path / f'{name}.pth'
    torch.save(checkpoint, file_path)
    return str(file_path)


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    strict: bool = True,
    map_location: str = 'cpu',
) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    state_dict = checkpoint.get('model', checkpoint)
    model.load_state_dict(state_dict, strict=strict)
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if scaler is not None and 'scaler' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler'])
    return checkpoint
