"""TRUE-ENSURE test-time adaptation utilities."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import time
from typing import Any, Dict, Mapping

import numpy as np
import torch

from cardiac_ensure.losses import compute_true_ensure_loss, ensure_data_term
from cardiac_ensure.ops import dynamic_a_adjoint, dynamic_a_forward, solve_rho_ls
from cardiac_ensure.scripts.eval_metrics import summarize_reconstruction_metrics, to_magnitude


EPS = 1.0e-8


@dataclass
class ENSURETTAResult:
    before_rec: np.ndarray
    after_rec: np.ndarray
    before_metrics: dict[str, float]
    after_metrics: dict[str, float]
    train_losses: list[float]
    self_val_losses: list[float]
    step_logs: list[dict[str, float | int | None]]
    num_tta_steps: int
    early_stop_step: int | None
    self_val_best: float
    runtime_sec: float
    adapt_runtime_sec: float
    best_step: int | None
    num_trainable_params: int


def _clone_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _load_state_dict_on_device(
    model: torch.nn.Module,
    state: Mapping[str, torch.Tensor],
    device: torch.device,
) -> None:
    model.load_state_dict({key: value.to(device) for key, value in state.items()}, strict=True)


def _as_float(value: torch.Tensor | float | int) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu().real.item())
    return float(value)


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _prediction_np(prediction: torch.Tensor) -> np.ndarray:
    return to_magnitude(prediction).detach().cpu().float().numpy()


def _target_or_none(batch: Mapping[str, Any]) -> torch.Tensor | None:
    target = batch.get("target_rss")
    return target if torch.is_tensor(target) else None


def _center_crop_to_shape(x: torch.Tensor, spatial_shape: tuple[int, int]) -> torch.Tensor:
    target_h, target_w = int(spatial_shape[0]), int(spatial_shape[1])
    if x.shape[-2] < target_h or x.shape[-1] < target_w:
        raise ValueError(
            f"Cannot center-crop tensor with spatial shape {tuple(x.shape[-2:])} to larger target {spatial_shape}"
        )
    h_start = (x.shape[-2] - target_h) // 2
    w_start = (x.shape[-1] - target_w) // 2
    return x[..., h_start : h_start + target_h, w_start : w_start + target_w]


def _align_prediction_to_target(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction.shape[-2:] == target.shape[-2:]:
        return prediction
    return _center_crop_to_shape(prediction, tuple(int(v) for v in target.shape[-2:]))


def _compute_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor | None,
    *,
    include_ssim: bool,
) -> dict[str, float]:
    if target is None:
        return {}
    return summarize_reconstruction_metrics(
        _align_prediction_to_target(prediction, target),
        target,
        include_ssim=include_ssim,
    )


def _count_trainable(parameters: list[torch.nn.Parameter]) -> int:
    return int(sum(param.numel() for param in parameters if param.requires_grad))


def _dc_weight_stats(model: torch.nn.Module) -> tuple[float | None, float | None]:
    values = [
        float(block.dc_weight_value().detach().cpu())
        for block in getattr(model, "dc_blocks", [])
        if hasattr(block, "dc_weight_value")
    ]
    if not values:
        return None, None
    return min(values), max(values)


def _center_slice(width: int, center_fraction: float) -> slice:
    num_cols = max(1, int(round(float(width) * float(center_fraction))))
    start = max(0, (int(width) - num_cols) // 2)
    return slice(start, start + num_cols)


def split_self_validation_mask(
    mask: torch.Tensor,
    *,
    fraction: float = 0.05,
    seed: int = 0,
    preserve_center: bool = True,
    center_fraction: float = 0.04,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split sampled phase-encode lines into train and held-out self-val masks."""

    if not (0.0 < float(fraction) < 1.0):
        raise ValueError(f"fraction must be in (0, 1), got {fraction}")
    if mask.ndim != 5 or mask.shape[2] != 1:
        raise ValueError(f"Expected mask shape [B, T, 1, H, W], got {tuple(mask.shape)}")

    train_mask = mask.detach().clone()
    val_mask = torch.zeros_like(train_mask)
    mask_cpu = mask.detach().cpu()
    batch_size, num_frames, _, _, width = mask_cpu.shape

    for batch_idx in range(batch_size):
        for frame_idx in range(num_frames):
            frame_mask = mask_cpu[batch_idx, frame_idx, 0]
            measured_cols = torch.any(frame_mask > 0, dim=0).numpy()
            candidates = measured_cols.copy()
            if preserve_center and width > 0:
                candidates[_center_slice(width, center_fraction)] = False
                if not np.any(candidates):
                    candidates = measured_cols.copy()
            candidate_cols = np.flatnonzero(candidates)
            if candidate_cols.size == 0:
                raise ValueError("Cannot create self-validation mask from an empty sampled mask")

            measured_count = int(np.count_nonzero(measured_cols))
            num_val = max(1, int(round(measured_count * float(fraction))))
            num_val = min(num_val, int(candidate_cols.size))
            rng = np.random.default_rng(int(seed) + 1009 * batch_idx + 9176 * frame_idx)
            val_cols = rng.choice(candidate_cols, size=num_val, replace=False)
            col_index = torch.as_tensor(val_cols, device=mask.device, dtype=torch.long)
            train_mask[batch_idx, frame_idx, 0, :, col_index] = 0.0
            val_mask[batch_idx, frame_idx, 0, :, col_index] = mask[batch_idx, frame_idx, 0, :, col_index]

    return train_mask, val_mask


def _masked_kspace(kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return kspace * mask.to(device=kspace.device, dtype=kspace.real.dtype)


def _normalized_complex_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    eps: float = EPS,
) -> torch.Tensor:
    numerator = torch.sum(torch.abs(prediction - target))
    denominator = torch.sum(torch.abs(target.detach())).clamp_min(float(eps))
    return numerator / denominator


def _self_validation_loss(
    model: torch.nn.Module,
    *,
    train_zf: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    kspace: torch.Tensor,
    maps: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    prediction = model(train_zf, maps=maps, mask=train_mask)
    val_prediction = dynamic_a_forward(prediction, maps, val_mask)
    val_target = _masked_kspace(kspace, val_mask)
    return _normalized_complex_l1(val_prediction, val_target), prediction


def _true_ensure_loss_on_mask(
    model: torch.nn.Module,
    *,
    zf_input: torch.Tensor,
    kspace: torch.Tensor,
    maps: torch.Tensor,
    mask: torch.Tensor,
    noise_sigma2: torch.Tensor,
    density_weight: torch.Tensor | None,
    cg_l2lam: float,
    cg_max_iter: int,
    cg_tol: float,
    divergence_eps: float | None,
    divergence_mc_samples: int,
    divergence_mode: str,
    project_divergence: bool,
) -> dict[str, torch.Tensor]:
    def model_fn(zf: torch.Tensor) -> torch.Tensor:
        return model(zf, maps=maps, mask=mask)

    return compute_true_ensure_loss(
        model_fn=model_fn,
        zf_input=zf_input,
        kspace=kspace,
        maps=maps,
        mask=mask,
        noise_sigma2=noise_sigma2,
        density_weight=density_weight,
        cg_l2lam=cg_l2lam,
        cg_max_iter=cg_max_iter,
        cg_tol=cg_tol,
        divergence_eps=divergence_eps,
        divergence_mc_samples=divergence_mc_samples,
        divergence_mode=divergence_mode,
        project_divergence=project_divergence,
    )


def _true_ensure_data_loss_on_mask(
    model: torch.nn.Module,
    *,
    zf_input: torch.Tensor,
    kspace: torch.Tensor,
    maps: torch.Tensor,
    mask: torch.Tensor,
    density_weight: torch.Tensor | None,
    cg_l2lam: float,
    cg_max_iter: int,
    cg_tol: float,
) -> dict[str, torch.Tensor]:
    prediction = model(zf_input, maps=maps, mask=mask)
    rho_ls, rho_info = solve_rho_ls(
        kspace=kspace,
        maps=maps,
        mask=mask,
        l2lam=cg_l2lam,
        max_iter=cg_max_iter,
        tol=cg_tol,
    )
    data_out = ensure_data_term(
        prediction=prediction,
        rho_ls=rho_ls,
        maps=maps,
        mask=mask,
        density_weight=density_weight,
        l2lam=cg_l2lam,
        max_iter=cg_max_iter,
        tol=cg_tol,
    )
    zero = torch.zeros((), device=prediction.device, dtype=prediction.real.dtype)
    data_term = data_out["data_term"]
    return {
        "loss": data_term,
        "data_term": data_term,
        "div_term": zero,
        "div_scale": zero,
        "div_contribution": zero,
        "risk_proxy": data_term,
        "prediction": prediction,
        "rho_ls": rho_ls,
        "projected_error": data_out["projected_error"],
        "rho_info": rho_info,
        "projection_info": data_out["projection_info"],
        "frame_energy": data_out["frame_energy"],
        "divergence_eps": zero,
    }


def run_true_ensure_tta(
    *,
    model: torch.nn.Module,
    batch: Mapping[str, Any],
    device: torch.device,
    seed: int,
    max_steps: int = 250,
    lr: float = 1.0e-5,
    weight_decay: float = 0.0,
    grad_clip: float | None = 1.0,
    self_val_fraction: float = 0.05,
    early_stop_window: int = 20,
    update_mode: str = "all_params",
    run_tta: bool = True,
    include_ssim: bool = False,
    cg_l2lam: float = 1.0e-6,
    cg_max_iter: int = 25,
    cg_tol: float = 1.0e-6,
    divergence_eps: float | None = None,
    divergence_mc_samples: int = 1,
    divergence_mode: str = "measurement",
    project_divergence: bool = True,
    tta_loss: str = "ensure",
) -> ENSURETTAResult:
    tic = time.time()
    if update_mode != "all_params":
        raise NotImplementedError("v1 cardiac ENSURE TTA only supports update_mode='all_params'")
    if tta_loss not in ("ensure", "ensure_data", "self_supervised"):
        raise ValueError(f"Unsupported tta_loss={tta_loss!r}; choose 'ensure', 'ensure_data', or 'self_supervised'")

    kspace = batch["kspace_us"]
    zf = batch["zf"]
    maps = batch["maps"]
    mask = batch["mask"]
    noise_sigma2 = batch["noise_sigma2"]
    density_weight = batch.get("inv_sqrt_density")
    target = _target_or_none(batch)

    model_was_training = model.training
    model.eval()
    with torch.no_grad():
        before_prediction = model(zf, maps=maps, mask=mask)
    model.train(model_was_training)
    before_metrics = _compute_metrics(before_prediction, target, include_ssim=include_ssim)
    before_rec = _prediction_np(before_prediction)

    if not run_tta or int(max_steps) <= 0:
        runtime = time.time() - tic
        return ENSURETTAResult(
            before_rec=before_rec,
            after_rec=before_rec,
            before_metrics=before_metrics,
            after_metrics=before_metrics,
            train_losses=[],
            self_val_losses=[],
            step_logs=[],
            num_tta_steps=0,
            early_stop_step=None,
            self_val_best=float("nan"),
            runtime_sec=runtime,
            adapt_runtime_sec=0.0,
            best_step=None,
            num_trainable_params=0,
        )

    train_mask, val_mask = split_self_validation_mask(mask, fraction=self_val_fraction, seed=seed)
    train_kspace = _masked_kspace(kspace, train_mask)
    train_zf = dynamic_a_adjoint(train_kspace, maps, train_mask)

    model_tta = copy.deepcopy(model).to(device)
    model_tta.train()
    params = [param for param in model_tta.parameters() if param.requires_grad]
    if not params:
        raise ValueError("No trainable model parameters selected for TTA")
    num_trainable_params = _count_trainable(params)
    optimizer = torch.optim.AdamW(params, lr=float(lr), weight_decay=float(weight_decay))

    train_losses: list[float] = []
    self_val_losses: list[float] = []
    step_logs: list[dict[str, float | int | None]] = []
    best_state = _clone_state_dict(model_tta)
    best_step: int | None = 0
    early_stop_step: int | None = None
    adapt_runtime_sec = 0.0

    _sync_if_cuda(device)
    initial_val_tic = time.perf_counter()
    with torch.no_grad():
        initial_val_loss, initial_val_prediction = _self_validation_loss(
            model_tta,
            train_zf=train_zf,
            train_mask=train_mask,
            val_mask=val_mask,
            kspace=kspace,
            maps=maps,
        )
    initial_val_loss_value = _as_float(initial_val_loss)
    best_val_loss = initial_val_loss_value if np.isfinite(initial_val_loss_value) else float("inf")
    _sync_if_cuda(device)
    adapt_runtime_sec += time.perf_counter() - initial_val_tic

    dc_min, dc_max = _dc_weight_stats(model_tta)
    initial_log: dict[str, float | int | None] = {
        "step": 0,
        "cumulative_adapt_runtime_sec": adapt_runtime_sec,
        "train_loss": None,
        "self_val_loss": initial_val_loss_value,
        "data_term": None,
        "div_term": None,
        "div_scale": None,
        "div_contribution": None,
        "risk_proxy": None,
        "divergence_eps": None,
        "dc_weight_min": dc_min,
        "dc_weight_max": dc_max,
    }
    if target is not None:
        metrics_step = _compute_metrics(initial_val_prediction, target, include_ssim=include_ssim)
        if "nmse" in metrics_step:
            initial_log["score_if_gt_available_nmse"] = float(metrics_step["nmse"])
        if "ssim" in metrics_step:
            initial_log["score_if_gt_available_ssim"] = float(metrics_step["ssim"])
    step_logs.append(initial_log)

    for step_idx in range(int(max_steps)):
        _sync_if_cuda(device)
        step_tic = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        if tta_loss == "ensure":
            loss_dict = _true_ensure_loss_on_mask(
                model_tta,
                zf_input=train_zf,
                kspace=train_kspace,
                maps=maps,
                mask=train_mask,
                noise_sigma2=noise_sigma2,
                density_weight=density_weight,
                cg_l2lam=cg_l2lam,
                cg_max_iter=cg_max_iter,
                cg_tol=cg_tol,
                divergence_eps=divergence_eps,
                divergence_mc_samples=divergence_mc_samples,
                divergence_mode=divergence_mode,
                project_divergence=project_divergence,
            )
            loss = loss_dict["loss"]
        elif tta_loss == "ensure_data":
            loss_dict = _true_ensure_data_loss_on_mask(
                model_tta,
                zf_input=train_zf,
                kspace=train_kspace,
                maps=maps,
                mask=train_mask,
                density_weight=density_weight,
                cg_l2lam=cg_l2lam,
                cg_max_iter=cg_max_iter,
                cg_tol=cg_tol,
            )
            loss = loss_dict["loss"]
        else:
            prediction = model_tta(train_zf, maps=maps, mask=train_mask)
            train_prediction = dynamic_a_forward(prediction, maps, train_mask)
            loss = _normalized_complex_l1(train_prediction, train_kspace)
            loss_dict = {
                "loss": loss,
                "data_term": loss.detach(),
                "div_term": torch.zeros((), device=loss.device, dtype=loss.real.dtype),
                "div_scale": torch.zeros((), device=loss.device, dtype=loss.real.dtype),
                "div_contribution": torch.zeros((), device=loss.device, dtype=loss.real.dtype),
                "risk_proxy": loss.detach(),
                "divergence_eps": torch.zeros((), device=loss.device, dtype=loss.real.dtype),
            }
        if not torch.isfinite(loss.detach()):
            raise FloatingPointError(f"Non-finite TTA loss at step {step_idx + 1}: {float(loss.detach())}")

        loss.backward()
        if grad_clip is not None and float(grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(params, float(grad_clip))
        optimizer.step()

        with torch.no_grad():
            val_loss, val_prediction = _self_validation_loss(
                model_tta,
                train_zf=train_zf,
                train_mask=train_mask,
                val_mask=val_mask,
                kspace=kspace,
                maps=maps,
            )

        train_loss_value = _as_float(loss)
        val_loss_value = _as_float(val_loss)
        train_losses.append(train_loss_value)
        self_val_losses.append(val_loss_value)

        if np.isfinite(val_loss_value) and val_loss_value < best_val_loss:
            best_val_loss = val_loss_value
            best_step = step_idx + 1
            best_state = _clone_state_dict(model_tta)

        should_stop = False
        if int(early_stop_window) > 0 and step_idx + 1 > 3 * int(early_stop_window):
            window = int(early_stop_window)
            curr = float(np.mean(self_val_losses[-window:]))
            prev = float(np.mean(self_val_losses[-2 * window : -window]))
            if np.isfinite(curr) and np.isfinite(prev) and curr > prev:
                early_stop_step = step_idx + 1
                should_stop = True

        _sync_if_cuda(device)
        adapt_runtime_sec += time.perf_counter() - step_tic

        dc_min, dc_max = _dc_weight_stats(model_tta)
        log_row: dict[str, float | int | None] = {
            "step": step_idx + 1,
            "cumulative_adapt_runtime_sec": adapt_runtime_sec,
            "train_loss": train_loss_value,
            "self_val_loss": val_loss_value,
            "data_term": _as_float(loss_dict["data_term"]),
            "div_term": _as_float(loss_dict["div_term"]),
            "div_scale": _as_float(loss_dict["div_scale"]),
            "div_contribution": _as_float(loss_dict["div_contribution"]),
            "risk_proxy": _as_float(loss_dict["risk_proxy"]),
            "divergence_eps": _as_float(loss_dict["divergence_eps"]),
            "dc_weight_min": dc_min,
            "dc_weight_max": dc_max,
        }
        if target is not None:
            metrics_step = _compute_metrics(val_prediction, target, include_ssim=include_ssim)
            if "nmse" in metrics_step:
                log_row["score_if_gt_available_nmse"] = float(metrics_step["nmse"])
            if "ssim" in metrics_step:
                log_row["score_if_gt_available_ssim"] = float(metrics_step["ssim"])
        step_logs.append(log_row)

        if should_stop:
            break

    model_tta.eval()
    if best_step is not None:
        _load_state_dict_on_device(model_tta, best_state, device)
    with torch.no_grad():
        after_prediction = model_tta(zf, maps=maps, mask=mask)
    after_metrics = _compute_metrics(after_prediction, target, include_ssim=include_ssim)
    after_rec = _prediction_np(after_prediction)
    runtime = time.time() - tic

    return ENSURETTAResult(
        before_rec=before_rec,
        after_rec=after_rec,
        before_metrics=before_metrics,
        after_metrics=after_metrics,
        train_losses=train_losses,
        self_val_losses=self_val_losses,
        step_logs=step_logs,
        num_tta_steps=len(train_losses),
        early_stop_step=early_stop_step,
        self_val_best=best_val_loss,
        runtime_sec=runtime,
        adapt_runtime_sec=adapt_runtime_sec,
        best_step=best_step,
        num_trainable_params=num_trainable_params,
    )
