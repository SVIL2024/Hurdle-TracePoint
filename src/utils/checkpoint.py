"""Checkpoint compatibility helpers for VadCLIP experiments."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


@dataclass(frozen=True)
class CheckpointPayload:
    """A model state with optional optimizer and training metadata."""

    state_dict: Mapping[str, Tensor]
    metadata: Mapping[str, Any]
    state_format: str

    @property
    def has_optimizer_state(self) -> bool:
        return "optimizer_state_dict" in self.metadata

    @property
    def has_tracepoint_state(self) -> bool:
        return any(key.startswith("tracepoint.") for key in self.state_dict)


def _is_tensor_state_dict(value: object) -> bool:
    return (
        isinstance(value, Mapping)
        and bool(value)
        and all(isinstance(key, str) and torch.is_tensor(tensor) for key, tensor in value.items())
    )


def load_checkpoint_payload(path: str | Path, map_location: str | torch.device) -> CheckpointPayload:
    """Load either a raw model state dictionary or a training checkpoint dictionary."""

    source = Path(path)
    payload = torch.load(source, map_location=map_location)

    if isinstance(payload, Mapping) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
        if not _is_tensor_state_dict(state_dict):
            raise ValueError(f"{source} has an invalid model_state_dict")
        return CheckpointPayload(
            state_dict=state_dict,
            metadata=payload,
            state_format="training_checkpoint",
        )

    if _is_tensor_state_dict(payload):
        return CheckpointPayload(
            state_dict=payload,
            metadata={},
            state_format="full_model_state",
        )

    raise ValueError(
        f"{source} is neither a raw model state dictionary nor a training checkpoint"
    )
