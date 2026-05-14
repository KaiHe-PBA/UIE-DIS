from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Protocol, Sequence

import torch
from torch import Tensor

from .losses import resolve_x0_prediction


ModelOutput = Mapping[str, Tensor] | Dict[str, Tensor]


class ForwardModel(Protocol):
    def forward(self, x_t: Tensor, y: Tensor, t: Tensor) -> ModelOutput:
        ...


@dataclass
class StepResult:
    final_state: Tensor
    final_output: Dict[str, Tensor]
    history: list[Dict[str, Tensor]]


def make_time_schedule(
    num_steps: int,
    t_start: float = 1.0,
    t_end: float = 0.0,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    if not 1 <= num_steps <= 4:
        raise ValueError("num_steps must be in [1, 4].")
    return torch.linspace(t_start, t_end, steps=num_steps, device=device, dtype=dtype or torch.float32)


def make_blend_schedule(num_steps: int, minimum_blend: float = 0.5) -> list[float]:
    if not 1 <= num_steps <= 4:
        raise ValueError("num_steps must be in [1, 4].")
    if num_steps == 1:
        return [1.0]
    return torch.linspace(minimum_blend, 1.0, steps=num_steps).tolist()


def _expand_time(t_scalar: Tensor, batch_size: int) -> Tensor:
    if t_scalar.ndim == 0:
        return t_scalar.repeat(batch_size)
    if t_scalar.ndim == 1 and t_scalar.shape[0] == 1:
        return t_scalar.repeat(batch_size)
    if t_scalar.ndim == 1 and t_scalar.shape[0] == batch_size:
        return t_scalar
    raise ValueError(f"Each timestep must be scalar or shape [1]/[B], got {tuple(t_scalar.shape)}")


def run_multistep_inference(
    model: ForwardModel,
    x_t: Tensor,
    y: Tensor,
    *,
    num_steps: int = 4,
    time_schedule: Optional[Sequence[float] | Tensor] = None,
    blend_schedule: Optional[Iterable[float]] = None,
    clamp_range: Optional[tuple[float, float]] = (0.0, 1.0),
    return_history: bool = True,
) -> StepResult:
    if not 1 <= num_steps <= 4:
        raise ValueError("num_steps must be in [1, 4].")

    if time_schedule is None:
        time_schedule_tensor = make_time_schedule(num_steps, device=x_t.device, dtype=x_t.dtype)
    else:
        time_schedule_tensor = torch.as_tensor(time_schedule, device=x_t.device, dtype=x_t.dtype)
        if time_schedule_tensor.numel() != num_steps:
            raise ValueError("time_schedule length must equal num_steps.")

    blends = list(blend_schedule) if blend_schedule is not None else make_blend_schedule(num_steps)
    if len(blends) != num_steps:
        raise ValueError("blend_schedule length must equal num_steps.")

    state = x_t
    final_output: Dict[str, Tensor] = {}
    history: list[Dict[str, Tensor]] = []

    for step_idx, (t_value, blend) in enumerate(zip(time_schedule_tensor, blends), start=1):
        t_batch = _expand_time(t_value, x_t.shape[0]).to(device=x_t.device, dtype=x_t.dtype)
        output = dict(model.forward(state, y, t_batch))
        x0_pred = resolve_x0_prediction(output, x_t=state)

        state = torch.lerp(state, x0_pred, float(blend))
        if clamp_range is not None:
            state = state.clamp(*clamp_range)

        output["x0_pred"] = x0_pred
        output["state"] = state
        output["t"] = t_batch
        output["step_index"] = torch.tensor(step_idx, device=x_t.device)
        final_output = output

        if return_history:
            history.append(
                {
                    "x0_pred": x0_pred.detach().clone(),
                    "state": state.detach().clone(),
                    "t": t_batch.detach().clone(),
                    "step_index": output["step_index"].detach().clone(),
                }
            )

    return StepResult(final_state=state, final_output=final_output, history=history)


def infer_one_step(model: ForwardModel, x_t: Tensor, y: Tensor, **kwargs: object) -> StepResult:
    return run_multistep_inference(model, x_t, y, num_steps=1, **kwargs)


def infer_two_steps(model: ForwardModel, x_t: Tensor, y: Tensor, **kwargs: object) -> StepResult:
    return run_multistep_inference(model, x_t, y, num_steps=2, **kwargs)


def infer_three_steps(model: ForwardModel, x_t: Tensor, y: Tensor, **kwargs: object) -> StepResult:
    return run_multistep_inference(model, x_t, y, num_steps=3, **kwargs)


def infer_four_steps(model: ForwardModel, x_t: Tensor, y: Tensor, **kwargs: object) -> StepResult:
    return run_multistep_inference(model, x_t, y, num_steps=4, **kwargs)
