"""Small helpers for isolated, reproducible pilot runs."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def configure_run_paths(args: Any, dataset: str) -> None:
    """Configure an isolated output directory without changing legacy defaults."""

    output_dir = getattr(args, "output_dir", None)
    if output_dir:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        best_path = root / "best.pth"
        args.checkpoint_path = str(best_path)
        args.model_path = str(best_path)
        args.current_model_path = str(root / "last.pth")
        args.metrics_path = str(root / "metrics.jsonl")

        manifest = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": dataset,
            "arguments": vars(args),
        }
        with (root / "run_config.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)

    for attribute in ("checkpoint_path", "model_path", "current_model_path", "metrics_path"):
        value = getattr(args, attribute, None)
        if value:
            Path(value).parent.mkdir(parents=True, exist_ok=True)


def append_metrics(metrics_path: str | None, payload: dict[str, Any]) -> None:
    """Append one JSON-serializable measurement record when logging is enabled."""

    if not metrics_path:
        return

    destination = Path(metrics_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)
        handle.write("\n")


def freeze_to_tracepoint_branch(model: Any) -> dict[str, int]:
    """Freeze an existing VAD model while leaving the TracePoint add-on trainable."""

    trainable = 0
    frozen = 0
    for name, parameter in model.named_parameters():
        is_tracepoint_parameter = (
            name.startswith("tracepoint.") or name == "tracepoint_fusion_logit"
        )
        parameter.requires_grad = is_tracepoint_parameter
        if is_tracepoint_parameter:
            trainable += parameter.numel()
        else:
            frozen += parameter.numel()

    if trainable == 0:
        raise ValueError(
            "TracePoint-only training requires a model with a TracePoint branch"
        )
    return {"trainable": trainable, "frozen": frozen}


def validate_isolated_tracepoint_args(args: Any) -> None:
    """Reject settings that would reconnect the event sidecar to VadCLIP."""

    if not getattr(args, "tracepoint", False):
        return

    if not getattr(args, "tracepoint_detach_mark_features", False):
        raise ValueError("isolated TracePoint requires detached semantic marks")
    if getattr(args, "tracepoint_fuse", False):
        raise ValueError("isolated TracePoint does not support score fusion")
    if getattr(args, "tracepoint_fusion_mil_weight", 0.0) != 0.0:
        raise ValueError("isolated TracePoint does not support fusion MIL loss")
    if getattr(args, "tracepoint_align_weight", 0.0) != 0.0:
        raise ValueError("isolated TracePoint keeps alignment loss disabled")
    if getattr(args, "tracepoint_freeze_backbone", False):
        raise ValueError("from-scratch isolated training cannot freeze VadCLIP")
    if getattr(args, "warm_start_path", None):
        raise ValueError("from-scratch isolated training cannot warm-start a VAD checkpoint")
    if getattr(args, "use_checkpoint", False):
        raise ValueError("from-scratch isolated training cannot resume a VAD checkpoint")


def isolated_parameter_groups(model: Any) -> tuple[list[Any], list[Any]]:
    """Return disjoint trainable VadCLIP and event-sidecar parameter lists."""

    if not getattr(model, "tracepoint_enabled", False) or model.tracepoint is None:
        raise ValueError("isolated parameter groups require an enabled TracePoint sidecar")

    primary = [
        parameter
        for _, parameter in model.primary_named_parameters()
        if parameter.requires_grad
    ]
    event = [
        parameter
        for _, parameter in model.event_named_parameters()
        if parameter.requires_grad
    ]
    primary_ids = {id(parameter) for parameter in primary}
    event_ids = {id(parameter) for parameter in event}
    if not primary or not event or primary_ids & event_ids:
        raise ValueError("primary and event parameter ownership must be non-empty and disjoint")
    return primary, event


def resolve_selection_metric(metrics: dict[str, Any], name: str) -> float:
    """Resolve a scalar checkpoint-selection metric from an evaluation record."""

    if name in metrics:
        return float(metrics[name])
    if name.startswith("map_"):
        threshold = name.removeprefix("map_")
        localization_map = metrics.get("localization_map", {})
        if threshold in localization_map:
            return float(localization_map[threshold])
    raise ValueError(
        f"unknown selection metric {name!r}; use an evaluation key or map_<IoU>"
    )
