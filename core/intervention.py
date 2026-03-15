
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
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    dtype = v_pred.dtype

    similarity = compute_similarity(
        v_pred,
        v_ref,
        mode=config.similarity_mode,
        eps=eps,
    )

    threshold = config.similarity_threshold
    high_sim_mask = similarity >= threshold
    low_sim_mask = ~high_sim_mask

    num_high_sim = high_sim_mask.sum().item()

    v_ref_dtype = v_ref.to(dtype)

    result = torch.where(high_sim_mask, v_ref_dtype, v_pred)

    if config.enable_blend:
        a = config.blend_weight
        blended = a * v_ref_dtype + (1 - a) * v_pred
        result = torch.where(low_sim_mask, blended, result)

    similarity_mask = (similarity < threshold).float()

    return result, similarity_mask, int(num_high_sim)


def log_intervention_stats(
    step: int,
    sigma: float,
    num_replaced: int,
    total_elements: int,
    similarity_mode: str = "elementwise",
    enable_blend: bool = False,
    blend_weight: float = 0.5,
) -> None:
    ratio = num_replaced / total_elements * 100 if total_elements > 0 else 0
    num_low_sim = total_elements - num_replaced

    if enable_blend:
        print(
            f"  Step {step}: sigma={sigma:.4f}, "
            f"high_sim={num_replaced} ({ratio:.1f}%) replaced, "
            f"low_sim={num_low_sim} ({100-ratio:.1f}%) blended (a={blend_weight:.2f})"
        )
    else:
        print(
            f"  Step {step}: {similarity_mode} intervention, sigma={sigma:.4f}, "
            f"replaced {num_replaced}/{total_elements} ({ratio:.1f}%)"
        )


def log_intervention_summary(
    total_replaced: int,
    total_checked: int,
    intervention_steps: int,
) -> None:
    if total_checked > 0:
        ratio = total_replaced / total_checked * 100
        print(
            f"[Intervention Summary] Total replaced: {total_replaced}/{total_checked} "
            f"({ratio:.2f}%) across {intervention_steps} steps"
        )
