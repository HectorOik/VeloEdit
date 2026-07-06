
import torch
import torch.nn.functional as F
from typing import Tuple, Optional
from .types import InterventionConfig


VALID_SIMILARITY_MODES = ("elementwise", "cosine")


def compute_reference_velocity(
    z_t: torch.Tensor,
    z_0: torch.Tensor,
    sigma: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    return (z_t - z_0) / (sigma + eps)


def compute_element_similarity(
    v_pred: torch.Tensor,
    v_ref: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    pred_float = v_pred.float()
    ref_float = v_ref.float()

    diff_abs = torch.abs(pred_float - ref_float)

    ref_abs = torch.abs(ref_float) + eps

    similarity = ref_abs / (ref_abs + diff_abs)

    return similarity


def normalize_similarity_mode(mode: Optional[str]) -> str:
    if mode is None:
        return "elementwise"

    normalized = str(mode).strip().lower()
    aliases = {
        "cos": "cosine",
        "cosine_token": "cosine",
        "token_cosine": "cosine",
    }
    normalized = aliases.get(normalized, normalized)

    if normalized not in VALID_SIMILARITY_MODES:
        raise ValueError(
            f"Unsupported similarity mode: {mode}. "
            f"Expected one of {', '.join(VALID_SIMILARITY_MODES)}"
        )

    return normalized


def compute_cosine_similarity(
    v_pred: torch.Tensor,
    v_ref: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    pred_float = v_pred.float()
    ref_float = v_ref.float()

    if pred_float.ndim == 1:
        cosine = F.cosine_similarity(
            pred_float.unsqueeze(0),
            ref_float.unsqueeze(0),
            dim=-1,
            eps=eps,
        )
        similarity = (cosine + 1.0) * 0.5
        return similarity.expand_as(pred_float)

    if pred_float.ndim == 4:
        cosine = F.cosine_similarity(pred_float, ref_float, dim=1, eps=eps)
        similarity = (cosine + 1.0) * 0.5
        return similarity.unsqueeze(1).expand_as(pred_float)

    cosine = F.cosine_similarity(pred_float, ref_float, dim=-1, eps=eps)
    similarity = (cosine + 1.0) * 0.5
    return similarity.unsqueeze(-1).expand_as(pred_float)


def compute_similarity(
    v_pred: torch.Tensor,
    v_ref: torch.Tensor,
    mode: str = "elementwise",
    eps: float = 1e-8,
) -> torch.Tensor:
    similarity_mode = normalize_similarity_mode(mode)

    if similarity_mode == "elementwise":
        return compute_element_similarity(v_pred, v_ref, eps)

    if similarity_mode == "cosine":
        return compute_cosine_similarity(v_pred, v_ref, eps)

    raise ValueError(f"Unsupported similarity mode: {similarity_mode}")


def apply_intervention(
    v_pred: torch.Tensor,
    v_ref: torch.Tensor,
    config: InterventionConfig,
    preserve_active: bool,
    edit_active: bool,
    similarity: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    dtype = v_pred.dtype

    if similarity is None:
        similarity = compute_similarity(
            v_pred,
            v_ref,
            mode=config.similarity_mode,
            eps=eps,
        )

    threshold = config.similarity_threshold
    high_sim_mask = similarity >= threshold
    low_sim_mask = ~high_sim_mask

    v_ref_dtype = v_ref.to(dtype)
    result = v_pred
    preserve_count = 0
    edit_count = 0

    if preserve_active:
        result = torch.where(high_sim_mask, v_ref_dtype, result)
        preserve_count = int(high_sim_mask.sum().item())

    if edit_active:
        a = config.blend_weight
        blended = a * v_ref_dtype + (1 - a) * v_pred
        result = torch.where(low_sim_mask, blended, result)
        edit_count = int(low_sim_mask.sum().item())

    similarity_mask = (similarity < threshold).float()

    return result, similarity_mask, preserve_count, edit_count


def log_intervention_stats(
    step: int,
    sigma: float,
    preserve_active: bool,
    edit_active: bool,
    preserve_count: int,
    edit_count: int,
    total_elements: int,
    similarity_mode: str = "elementwise",
    blend_weight: float = 0.5,
) -> None:
    preserve_ratio = preserve_count / total_elements * 100 if total_elements > 0 else 0
    edit_ratio = edit_count / total_elements * 100 if total_elements > 0 else 0
    preserve_state = "on" if preserve_active else "off"
    edit_state = "on" if edit_active else "off"

    print(
        f"  Step {step}: {similarity_mode} intervention, sigma={sigma:.4f}, "
        f"preserve[{preserve_state}]={preserve_count}/{total_elements} "
        f"({preserve_ratio:.1f}%) replaced, "
        f"edit[{edit_state}]={edit_count}/{total_elements} "
        f"({edit_ratio:.1f}%) blended (a={blend_weight:.2f})"
    )


def log_intervention_summary(
    total_preserve: int,
    total_edit: int,
    preserve_steps: int,
    edit_steps: int,
) -> None:
    total = total_preserve + total_edit
    if total > 0:
        print(
            f"[Intervention Summary] preserve_steps={preserve_steps}, edit_steps={edit_steps}, "
            f"preserve_affected={total_preserve}, edit_affected={total_edit}, "
            f"total_affected={total}"
        )
