import importlib
import importlib.util
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


class IdentityEnhancer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x_t: torch.Tensor, y: torch.Tensor, t: torch.Tensor, **_: Any) -> Dict[str, torch.Tensor]:
        del t
        mixed = 0.5 * x_t + 0.5 * y + self.dummy.view(1, 1, 1, 1) * 0.0
        return {'x0': torch.clamp(mixed, 0.0, 1.0)}


def _load_object(spec: str) -> Any:
    if ':' not in spec:
        raise ValueError('模型规格必须为 module:Class 或 /path/file.py:Class')
    module_name, object_name = spec.split(':', 1)
    if module_name.endswith('.py') or module_name.startswith('/'):
        module_path = Path(module_name)
        if not module_path.exists():
            raise FileNotFoundError(f'模型文件不存在: {module_path}')
        spec_obj = importlib.util.spec_from_file_location(module_path.stem, module_path)
        if spec_obj is None or spec_obj.loader is None:
            raise ImportError(f'无法加载模型文件: {module_path}')
        module = importlib.util.module_from_spec(spec_obj)
        spec_obj.loader.exec_module(module)
    else:
        module = importlib.import_module(module_name)
    if not hasattr(module, object_name):
        raise AttributeError(f'模块中未找到对象: {spec}')
    return getattr(module, object_name)


def build_model(model_spec: Optional[str], model_kwargs: Optional[Dict[str, Any]] = None) -> nn.Module:
    if not model_spec:
        return IdentityEnhancer()
    model_kwargs = model_kwargs or {}
    obj = _load_object(model_spec)
    if isinstance(obj, nn.Module):
        return obj
    return obj(**model_kwargs)


def call_model(
    model: nn.Module,
    x_t: torch.Tensor,
    y: torch.Tensor,
    t: torch.Tensor,
    batch: Optional[Dict[str, Any]] = None,
    **extra_kwargs: Any,
) -> Any:
    batch = batch or {}
    call_kwargs = {'x_t': x_t, 'y': y, 't': t, **extra_kwargs}
    combined_batch = {**batch, 'x_t': x_t, 'y': y, 't': t, **extra_kwargs}
    attempts = [
        lambda: model(**call_kwargs),
        lambda: model(x_t, y, t, **extra_kwargs),
        lambda: model(combined_batch),
        lambda: model(x_t=x_t, cond=y, t=t, **extra_kwargs),
        lambda: model(x=x_t, y=y, t=t, **extra_kwargs),
        lambda: model(y, x_t, t),
    ]
    last_error = None
    for attempt in attempts:
        try:
            return attempt()
        except TypeError as exc:
            last_error = exc
            continue
    raise RuntimeError(f'无法匹配模型前向接口，请检查模型 API。最后错误: {last_error}')


def extract_prediction(output: Any) -> Tuple[torch.Tensor, Dict[str, Any]]:
    extras: Dict[str, Any] = {}
    if isinstance(output, torch.Tensor):
        return output, extras
    if isinstance(output, dict):
        extras = output
        for key in ['x0', 'pred', 'prediction', 'out', 'image', 'enhanced', 'restored']:
            if key in output and isinstance(output[key], torch.Tensor):
                return output[key], extras
        for value in output.values():
            if isinstance(value, torch.Tensor):
                return value, extras
    if isinstance(output, (tuple, list)):
        if not output:
            raise RuntimeError('模型返回空 tuple/list，无法提取预测结果')
        first = output[0]
        if isinstance(first, torch.Tensor):
            extras['sequence'] = output
            return first, extras
        if isinstance(first, dict):
            return extract_prediction(first)
    raise TypeError(f'不支持的模型输出类型: {type(output)}')
