from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Mapping

import numpy as np
import torch

from cardiac_ensure.ops import dynamic_a_adjoint, dynamic_a_forward


EPS = 1.0e-8


def _ensure_dynamic_mask(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 4:
        mask = mask.unsqueeze(0)
    if mask.ndim != 5 or mask.shape[2] != 1:
        raise ValueError(f"Expected mask shape [B,T,1,H,W] or [T,1,H,W], got {tuple(mask.shape)}")
    return mask


def _center_columns(width: int, small_acs_block: tuple[int, int]) -> slice:
    acs_cols = max(0, int(small_acs_block[1]))
    if acs_cols <= 0:
        return slice(0, 0)
    start = max(0, (int(width) - acs_cols) // 2)
    return slice(start, min(int(width), start + acs_cols))


def _seed_for_frame(seed: int, sample_seed: int | None, batch_idx: int, frame_idx: int) -> int:
    base = int(seed) if sample_seed is None else int(sample_seed)
    return (base + 1009 * int(batch_idx) + 9176 * int(frame_idx)) % (2**32)


def _sample_loss_columns(
    measured_cols: np.ndarray,
    *,
    rho: float,
    mask_type: str,
    rng: np.random.Generator,
    center_col: int,
    width: int,
    gaussian_std_scale: float,
) -> np.ndarray:
    candidates = np.flatnonzero(measured_cols)
    if candidates.size == 0:
        raise ValueError("Cannot split SSDU mask with no measured columns")

    num_loss = max(1, int(round(float(np.count_nonzero(measured_cols)) * float(rho))))
    num_loss = min(num_loss, int(candidates.size))
    if mask_type == "uniform":
        return rng.choice(candidates, size=num_loss, replace=False)
    if mask_type != "gaussian":
        raise ValueError(f"Unsupported ssdu mask_type={mask_type!r}; choose 'gaussian' or 'uniform'")

    sigma = max(float(width - 1) / max(float(gaussian_std_scale), EPS), 1.0)
    weights = np.exp(-((candidates.astype(np.float64) - float(center_col)) ** 2) / (2.0 * sigma * sigma))
    weights_sum = float(np.sum(weights))
    if not np.isfinite(weights_sum) or weights_sum <= 0.0:
        weights = None
    else:
        weights = weights / weights_sum
    return rng.choice(candidates, size=num_loss, replace=False, p=weights)


def split_ssdu_mask(
    mask: torch.Tensor,
    *,
    rho: float = 0.4,
    mask_type: str = "gaussian",
    small_acs_block: tuple[int, int] = (4, 4),
    seed: int = 0,
    sample_seeds: Sequence[int] | torch.Tensor | None = None,
    gaussian_std_scale: float = 4.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split acquired samples Ω into SSDU train Θ and loss Λ masks.

    The project uses Cartesian line masks, so the split is performed at the
    phase-encode-column level. The central ACS columns are kept in Θ whenever
    enough non-center sampled columns are available.
    """

    if not (0.0 < float(rho) < 1.0):
        raise ValueError(f"rho must be in (0, 1), got {rho}")
    if float(gaussian_std_scale) <= 0.0:
        raise ValueError(f"gaussian_std_scale must be positive, got {gaussian_std_scale}")

    mask_5d = _ensure_dynamic_mask(mask)
    train_mask = mask_5d.detach().clone()
    loss_mask = torch.zeros_like(train_mask)
    mask_cpu = mask_5d.detach().cpu()
    batch_size, num_frames, _, _, width = mask_cpu.shape

    if sample_seeds is None:
        seed_values: list[int | None] = [None] * batch_size
    elif torch.is_tensor(sample_seeds):
        seed_values = [int(value) for value in sample_seeds.detach().cpu().reshape(-1).tolist()]
    else:
        seed_values = [int(value) for value in sample_seeds]
    if len(seed_values) != batch_size:
        raise ValueError(f"sample_seeds length {len(seed_values)} does not match batch_size={batch_size}")

    acs_slice = _center_columns(width, small_acs_block)
    center_col = int(width // 2)

    for batch_idx in range(batch_size):
        for frame_idx in range(num_frames):
            frame_mask = mask_cpu[batch_idx, frame_idx, 0]
            measured_cols = torch.any(frame_mask > 0, dim=0).numpy()
            candidates = measured_cols.copy()
            if acs_slice.stop > acs_slice.start:
                candidates[acs_slice] = False
                if not np.any(candidates):
                    candidates = measured_cols.copy()
            rng = np.random.default_rng(
                _seed_for_frame(
                    int(seed),
                    seed_values[batch_idx],
                    batch_idx=batch_idx,
                    frame_idx=frame_idx,
                )
            )
            loss_cols = _sample_loss_columns(
                candidates,
                rho=rho,
                mask_type=str(mask_type),
                rng=rng,
                center_col=center_col,
                width=width,
                gaussian_std_scale=gaussian_std_scale,
            )
            col_index = torch.as_tensor(loss_cols, device=mask_5d.device, dtype=torch.long)
            train_mask[batch_idx, frame_idx, 0, :, col_index] = 0.0
            loss_mask[batch_idx, frame_idx, 0, :, col_index] = mask_5d[
                batch_idx,
                frame_idx,
                0,
                :,
                col_index,
            ]

    return train_mask, loss_mask


def normalized_complex_l1_l2(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    eps: float = EPS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    diff = prediction - target
    l2 = torch.linalg.vector_norm(diff.reshape(-1)) / torch.linalg.vector_norm(target.detach().reshape(-1)).clamp_min(
        float(eps)
    )
    l1 = torch.sum(torch.abs(diff)) / torch.sum(torch.abs(target.detach())).clamp_min(float(eps))
    return 0.5 * l2 + 0.5 * l1, l1, l2


def compute_ssdu_loss(
    model: torch.nn.Module,
    batch: Mapping[str, Any],
    *,
    rho: float = 0.4,
    mask_type: str = "gaussian",
    small_acs_block: tuple[int, int] = (4, 4),
    seed: int = 0,
    sample_seeds: Sequence[int] | torch.Tensor | None = None,
    gaussian_std_scale: float = 4.0,
) -> dict[str, torch.Tensor]:
    kspace = batch["kspace_us"]
    maps = batch["maps"]
    mask = batch["mask"]
    train_mask, loss_mask = split_ssdu_mask(
        mask,
        rho=rho,
        mask_type=mask_type,
        small_acs_block=small_acs_block,
        seed=seed,
        sample_seeds=sample_seeds,
        gaussian_std_scale=gaussian_std_scale,
    )
    train_kspace = kspace * train_mask.to(device=kspace.device, dtype=kspace.real.dtype)
    loss_target = kspace * loss_mask.to(device=kspace.device, dtype=kspace.real.dtype)
    train_zf = dynamic_a_adjoint(train_kspace, maps, train_mask)
    prediction = model(train_zf, maps=maps, mask=train_mask)
    loss_prediction = dynamic_a_forward(prediction, maps, loss_mask)
    loss, l1, l2 = normalized_complex_l1_l2(loss_prediction, loss_target)
    measured = torch.count_nonzero(mask.detach() > 0).to(dtype=torch.float32)
    heldout = torch.count_nonzero(loss_mask.detach() > 0).to(dtype=torch.float32)
    return {
        "loss": loss,
        "ssdu_l1": l1.detach(),
        "ssdu_l2": l2.detach(),
        "prediction": prediction,
        "train_mask": train_mask,
        "loss_mask": loss_mask,
        "train_zf": train_zf,
        "train_kspace": train_kspace,
        "loss_target": loss_target,
        "loss_mask_fraction": heldout / measured.clamp_min(1.0),
    }
