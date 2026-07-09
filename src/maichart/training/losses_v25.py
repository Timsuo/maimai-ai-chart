"""Losses for the V2.5 multi-task frame predictor."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_v25_losses(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    loss_mask: torch.Tensor,
    *,
    note_weight: float = 1.0,
    buttons_weight: float = 1.0,
    type_weight: float = 1.0,
    density_weight: float = 1.0,
    note_start_weight: float = 1.0,
    pattern_start_weight: float = 0.5,
    chord_size_weight: float = 0.5,
    note_pos_weight: float | torch.Tensor | str | None = 1.0,
    button_pos_weight: float | torch.Tensor | str | None = "auto",
    pattern_class_weight: torch.Tensor | None = None,
    chord_class_weight: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    mask = loss_mask.to(device=outputs["note_presence_logits"].device, dtype=torch.float32)
    if mask.ndim != 2:
        raise ValueError(f"loss_mask must have shape [B, T], got {tuple(mask.shape)}.")

    note_target = targets["note_presence"].to(outputs["note_presence_logits"].device)
    buttons_target = targets["buttons"].to(outputs["button_logits"].device)
    type_target = targets["note_type"].to(outputs["type_logits"].device)
    density_target = targets["density"].to(outputs["density_pred"].device)
    note_start_target = targets.get("note_start")
    pattern_start_target = targets.get("pattern_start")
    chord_size_target = targets.get("chord_size_start")
    note_pos_weight_tensor = _resolve_pos_weight(
        note_pos_weight,
        note_target,
        mask.unsqueeze(-1),
        device=outputs["note_presence_logits"].device,
    )
    button_pos_weight_tensor = _resolve_pos_weight(
        button_pos_weight,
        buttons_target,
        mask.unsqueeze(-1),
        device=outputs["button_logits"].device,
    )

    loss_note = _masked_mean(
        F.binary_cross_entropy_with_logits(
            outputs["note_presence_logits"],
            note_target,
            reduction="none",
            pos_weight=note_pos_weight_tensor,
        ),
        mask.unsqueeze(-1),
    )
    loss_buttons = _masked_mean(
        F.binary_cross_entropy_with_logits(
            outputs["button_logits"],
            buttons_target,
            reduction="none",
            pos_weight=button_pos_weight_tensor,
        ),
        mask.unsqueeze(-1),
    )
    loss_type = _masked_mean(
        F.cross_entropy(
            outputs["type_logits"].transpose(1, 2),
            type_target,
            reduction="none",
        ),
        mask,
    )
    loss_density = _masked_mean(
        F.mse_loss(outputs["density_pred"], density_target, reduction="none"),
        mask.unsqueeze(-1),
    )
    if note_start_target is not None and "note_start_logits" in outputs:
        note_start_target = note_start_target.to(outputs["note_start_logits"].device)
        loss_note_start = _masked_mean(
            F.binary_cross_entropy_with_logits(
                outputs["note_start_logits"],
                note_start_target,
                reduction="none",
            ),
            mask.unsqueeze(-1),
        )
    else:
        loss_note_start = _zero_loss(outputs["note_presence_logits"])

    if pattern_start_target is not None and "pattern_start_logits" in outputs:
        loss_pattern_start = _masked_cross_entropy(
            outputs["pattern_start_logits"],
            pattern_start_target.to(outputs["pattern_start_logits"].device),
            mask.to(outputs["pattern_start_logits"].device),
            weight=pattern_class_weight,
        )
    else:
        loss_pattern_start = _zero_loss(outputs["note_presence_logits"])

    if chord_size_target is not None and "chord_size_logits" in outputs:
        loss_chord_size = _masked_cross_entropy(
            outputs["chord_size_logits"],
            chord_size_target.to(outputs["chord_size_logits"].device),
            mask.to(outputs["chord_size_logits"].device),
            weight=chord_class_weight,
        )
    else:
        loss_chord_size = _zero_loss(outputs["note_presence_logits"])

    total = (
        note_weight * loss_note
        + buttons_weight * loss_buttons
        + type_weight * loss_type
        + density_weight * loss_density
        + note_start_weight * loss_note_start
        + pattern_start_weight * loss_pattern_start
        + chord_size_weight * loss_chord_size
    )
    return {
        "loss": total,
        "loss_note": loss_note,
        "loss_buttons": loss_buttons,
        "loss_type": loss_type,
        "loss_density": loss_density,
        "loss_note_start": loss_note_start,
        "loss_pattern_start": loss_pattern_start,
        "loss_chord_size": loss_chord_size,
    }


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = values * mask
    denom = mask.expand_as(values).sum().clamp_min(1.0)
    return masked.sum() / denom


def _masked_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    weight: torch.Tensor | None = None,
    ignore_index: int = -100,
) -> torch.Tensor:
    valid = mask.to(dtype=torch.bool, device=logits.device) & (target != ignore_index)
    if not valid.any():
        return _zero_loss(logits)
    class_weight = weight.to(device=logits.device, dtype=logits.dtype) if weight is not None else None
    return F.cross_entropy(logits[valid], target[valid], weight=class_weight)


def _zero_loss(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


def _resolve_pos_weight(
    value: float | torch.Tensor | str | None,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor | None:
    if value is None:
        return None
    if isinstance(value, str):
        if value.strip().lower() != "auto":
            raise ValueError(f"Unsupported pos_weight value: {value!r}.")
        return _auto_pos_weight(target, mask).to(device)
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=target.dtype)
    return torch.tensor(float(value), device=device, dtype=target.dtype)


def _auto_pos_weight(target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded_mask = mask.expand_as(target).to(dtype=target.dtype, device=target.device)
    positives = (target * expanded_mask).sum(dim=tuple(range(target.ndim - 1)))
    total = expanded_mask.sum(dim=tuple(range(target.ndim - 1)))
    negatives = total - positives
    return torch.where(
        positives > 0,
        negatives / positives.clamp_min(1.0),
        torch.ones_like(positives),
    )
