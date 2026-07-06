
import torch
from typing import Callable, Optional, List, Tuple
from functools import partial
import torch.distributed as dist
import tqdm

from .types import SamplingResult, InterventionConfig
from .intervention import (
    compute_reference_velocity,
    compute_similarity,
    apply_intervention,
    log_intervention_stats,
    log_intervention_summary,
)

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


def euler_step(
    z: torch.Tensor,
    v_pred: torch.Tensor,
    sigma: float,
    sigma_next: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    dt = sigma_next - sigma

    x0_pred = z - sigma * v_pred

    z_next = z + dt * v_pred

    return z_next, x0_pred


def _sigma_values(sigmas: torch.Tensor) -> List[float]:
    return [
        sigma.item() if hasattr(sigma, "item") else float(sigma)
        for sigma in sigmas
    ]


def compute_sigma_deltas(sigmas: torch.Tensor) -> List[float]:
    values = _sigma_values(sigmas)
    return [values[i] - values[i + 1] for i in range(len(values) - 1)]


def format_sigma_schedule(sigmas: torch.Tensor) -> str:
    return "[" + ", ".join(f"{value:.6f}" for value in _sigma_values(sigmas)) + "]"


def format_sigma_deltas(sigmas: torch.Tensor) -> str:
    return "[" + ", ".join(f"{value:.6f}" for value in compute_sigma_deltas(sigmas)) + "]"


def _should_log() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def align_first_step_to_six_step(
    sigma_schedule_n: torch.Tensor,
    sigma_schedule_6: torch.Tensor,
    atol: float = 1e-6,
) -> torch.Tensor:
    requested_steps = len(sigma_schedule_n) - 1
    if requested_steps <= 6 or len(sigma_schedule_n) < 2 or len(sigma_schedule_6) < 2:
        return sigma_schedule_n

    sigma_n_float = sigma_schedule_n.detach().float()
    sigma_6_float = sigma_schedule_6.detach().float()
    delta6 = sigma_6_float[0] - sigma_6_float[1]
    target_sigma1_float = sigma_n_float[0] - delta6

    start = sigma_n_float[0].item()
    end = sigma_n_float[-1].item()
    target = target_sigma1_float.item()
    descending = start >= end

    if descending:
        if target >= start or target <= end:
            return sigma_schedule_n
        tail_mask = sigma_n_float < (target - atol)
    else:
        if target <= start or target >= end:
            return sigma_schedule_n
        tail_mask = sigma_n_float > (target + atol)

    target_sigma1 = target_sigma1_float.to(
        device=sigma_schedule_n.device,
        dtype=sigma_schedule_n.dtype,
    ).reshape(1)
    effective = torch.cat(
        [sigma_schedule_n[:1], target_sigma1, sigma_schedule_n[tail_mask]],
        dim=0,
    )

    return effective


def log_sampling_schedule(
    model_name: str,
    requested_steps: int,
    raw_sigma_schedule: torch.Tensor,
    effective_sigma_schedule: torch.Tensor,
    sigma_schedule_6: Optional[torch.Tensor] = None,
) -> None:
    if not _should_log():
        return

    print(f"[Sigma Schedule][{model_name}] requested_steps={requested_steps}")
    if sigma_schedule_6 is not None:
        print(
            f"[Sigma Schedule][{model_name}] 6-step raw sigmas: "
            f"{format_sigma_schedule(sigma_schedule_6)}"
        )
        print(
            f"[Sigma Schedule][{model_name}] 6-step deltas: "
            f"{format_sigma_deltas(sigma_schedule_6)}"
        )
    print(
        f"[Sigma Schedule][{model_name}] requested raw sigmas: "
        f"{format_sigma_schedule(raw_sigma_schedule)}"
    )
    print(
        f"[Sigma Schedule][{model_name}] requested raw deltas: "
        f"{format_sigma_deltas(raw_sigma_schedule)}"
    )
    print(
        f"[Sigma Schedule][{model_name}] effective_steps={len(effective_sigma_schedule) - 1}"
    )
    print(
        f"[Sigma Schedule][{model_name}] effective sigmas: "
        f"{format_sigma_schedule(effective_sigma_schedule)}"
    )
    print(
        f"[Sigma Schedule][{model_name}] effective deltas: "
        f"{format_sigma_deltas(effective_sigma_schedule)}"
    )


def resolve_intervention_steps(requested_steps: int, num_steps: int) -> int:
    if requested_steps < 0:
        requested_steps = num_steps + requested_steps
    return max(0, min(requested_steps, num_steps))


def run_deterministic_sampling(
    v_pred_fn: Callable[[torch.Tensor, float], torch.Tensor],
    z: torch.Tensor,
    sigma_schedule: torch.Tensor,
    reference_latent: Optional[torch.Tensor] = None,
    intervention_config: Optional[InterventionConfig] = None,
) -> SamplingResult:
    dtype = z.dtype
    device = z.device

    all_latents = [z.detach().clone()]
    all_velocities = []
    step_pred_x0 = []
    similarity_masks = []
    sigmas = [sigma_schedule[i].item() for i in range(len(sigma_schedule))]

    if intervention_config is None:
        intervention_config = InterventionConfig()
    enable_intervention = intervention_config.is_enabled() and reference_latent is not None
    generate_mask = reference_latent is not None

    total_preserve = 0
    total_edit = 0

    num_steps = len(sigma_schedule) - 1
    preserve_steps = resolve_intervention_steps(
        intervention_config.preserve_steps,
        num_steps,
    )
    edit_steps = resolve_intervention_steps(
        intervention_config.edit_steps,
        num_steps,
    )
    if _should_log() and (
        preserve_steps != intervention_config.preserve_steps
        or edit_steps != intervention_config.edit_steps
    ):
        print(
            "[Intervention Steps] "
            f"effective_sampling_steps={num_steps}, "
            f"preserve={intervention_config.preserve_steps}->{preserve_steps}, "
            f"edit={intervention_config.edit_steps}->{edit_steps}"
        )

    for i in tqdm(
        range(num_steps),
        desc="Deterministic Sampling",
        disable=dist.is_initialized() and dist.get_rank() != 0,
    ):
        sigma = sigma_schedule[i]
        sigma_next = sigma_schedule[i + 1]
        sigma_val = sigma.item() if hasattr(sigma, 'item') else float(sigma)
        sigma_next_val = sigma_next.item() if hasattr(sigma_next, 'item') else float(sigma_next)

        v_pred = v_pred_fn(z.to(dtype), sigma)
        v_ref = None
        similarity = None

        if generate_mask:
            v_ref = compute_reference_velocity(z, reference_latent, sigma_val)
            v_ref = v_ref.to(dtype)

            similarity = compute_similarity(
                v_pred,
                v_ref,
                mode=intervention_config.similarity_mode,
            )
            step_mask = (similarity < intervention_config.similarity_threshold).float()
            similarity_masks.append(step_mask.detach().clone().cpu())

        preserve_active = enable_intervention and i < preserve_steps
        edit_active = enable_intervention and i < edit_steps

        if (preserve_active or edit_active) and v_ref is not None:
            v_pred, _, preserve_count, edit_count = apply_intervention(
                v_pred,
                v_ref,
                intervention_config,
                preserve_active=preserve_active,
                edit_active=edit_active,
                similarity=similarity,
            )

            num_elements = v_pred.numel()
            total_preserve += preserve_count
            total_edit += edit_count

            log_intervention_stats(
                step=i,
                sigma=sigma_val,
                preserve_active=preserve_active,
                edit_active=edit_active,
                preserve_count=preserve_count,
                edit_count=edit_count,
                total_elements=num_elements,
                similarity_mode=intervention_config.similarity_mode,
                blend_weight=intervention_config.blend_weight,
            )

        all_velocities.append(v_pred.detach().clone().cpu())

        z, x0_pred = euler_step(z, v_pred, sigma_val, sigma_next_val)
        z = z.to(dtype)

        step_pred_x0.append(x0_pred.detach().clone().cpu())
        all_latents.append(z.detach().clone().cpu())

    if total_preserve + total_edit > 0:
        log_intervention_summary(
            total_preserve,
            total_edit,
            preserve_steps,
            edit_steps,
        )

    return SamplingResult(
        latents=z.to(dtype),
        all_latents=all_latents,
        all_velocities=all_velocities,
        step_pred_x0=step_pred_x0,
        sigmas=sigmas,
        similarity_masks=similarity_masks if generate_mask else None,
        interventions_applied=total_preserve + total_edit,
        preserve_interventions_applied=total_preserve,
        edit_interventions_applied=total_edit,
    )


def create_sigma_schedule(
    num_steps: int,
    sigma_max: float = 1.0,
    sigma_min: float = 0.0,
) -> torch.Tensor:
    return torch.linspace(sigma_max, sigma_min, num_steps + 1)
