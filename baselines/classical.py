from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np
import torch

from cardiac_ensure.scripts.eval_metrics import to_magnitude


def zero_filled_from_batch(batch: Mapping[str, Any]) -> torch.Tensor:
    """Return the dataset-provided adjoint zero-filled reconstruction."""

    if "zf" not in batch:
        raise KeyError("zero-filled baseline requires batch['zf']")
    return to_magnitude(batch["zf"]).float()


def _numpy_complex(tensor: torch.Tensor, name: str) -> np.ndarray:
    if not torch.is_tensor(tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not torch.is_complex(tensor):
        raise TypeError(f"{name} must be a complex tensor, got {tensor.dtype}")
    return tensor.detach().cpu().numpy().astype(np.complex64, copy=False)


def _ensure_batched_kspace(kspace: torch.Tensor) -> torch.Tensor:
    if kspace.ndim == 4:
        return kspace.unsqueeze(0)
    if kspace.ndim == 5:
        return kspace
    raise ValueError(f"Expected kspace [T,C,H,W] or [B,T,C,H,W], got {tuple(kspace.shape)}")


def _ensure_batched_maps(maps: torch.Tensor, batch_size: int) -> torch.Tensor:
    if maps.ndim == 3:
        maps = maps.unsqueeze(0)
    if maps.ndim != 4:
        raise ValueError(f"Expected maps [C,H,W] or [B,C,H,W], got {tuple(maps.shape)}")
    if maps.shape[0] == 1 and batch_size > 1:
        maps = maps.expand(batch_size, -1, -1, -1)
    if maps.shape[0] != batch_size:
        raise ValueError(f"maps batch={maps.shape[0]} does not match kspace batch={batch_size}")
    return maps


def _kspace_to_bart(kspace_tchw: np.ndarray) -> np.ndarray:
    """Convert [T,C,H,W] k-space into BART's 16D convention."""

    t_dim, c_dim, h_dim, w_dim = kspace_tchw.shape
    return (
        kspace_tchw.transpose(2, 3, 1, 0)
        .reshape(h_dim, w_dim, 1, c_dim, 1, 1, 1, 1, 1, 1, t_dim, 1, 1, 1, 1, 1)
        .astype(np.complex64, copy=False)
    )


def _maps_to_bart(maps_chw: np.ndarray) -> np.ndarray:
    """Convert [C,H,W] sensitivity maps into BART's 16D convention."""

    c_dim, h_dim, w_dim = maps_chw.shape
    return (
        maps_chw.transpose(1, 2, 0)
        .reshape(h_dim, w_dim, 1, c_dim, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
        .astype(np.complex64, copy=False)
    )


def _bart_output_to_t1hw(recon_bart: np.ndarray, *, num_frames: int, height: int, width: int) -> np.ndarray:
    recon = np.asarray(recon_bart)
    expected = int(num_frames) * int(height) * int(width)
    if recon.size != expected:
        raise ValueError(
            f"BART output has {recon.size} elements, expected {expected} for "
            f"[T,H,W]=[{num_frames},{height},{width}]"
        )
    recon_thw = recon.reshape(height, width, num_frames).transpose(2, 0, 1)
    return recon_thw[:, None, ...].astype(np.complex64, copy=False)


def bart_pics_single(
    kspace_tchw: np.ndarray,
    maps_chw: np.ndarray,
    *,
    bart_fn: Callable[..., np.ndarray],
    lamda: float = 0.01,
    command: str | None = None,
) -> np.ndarray:
    """Run BART PICS for one static/dynamic slice.

    Parameters use this project's tensor convention rather than the older
    UniPrompt cardiac cine convention: k-space is [T,C,H,W], maps are [C,H,W],
    and the returned image is [T,1,H,W].
    """

    kspace = np.asarray(kspace_tchw, dtype=np.complex64)
    maps = np.asarray(maps_chw, dtype=np.complex64)
    if kspace.ndim != 4:
        raise ValueError(f"kspace_tchw must be [T,C,H,W], got {kspace.shape}")
    if maps.ndim != 3:
        raise ValueError(f"maps_chw must be [C,H,W], got {maps.shape}")
    t_dim, c_dim, h_dim, w_dim = kspace.shape
    if maps.shape != (c_dim, h_dim, w_dim):
        raise ValueError(f"maps shape {maps.shape} does not match kspace {(c_dim, h_dim, w_dim)}")

    cmd = command or f"pics -g -d0 -S -R T:1024:0:{float(lamda)}"
    recon_bart = bart_fn(1, cmd, _kspace_to_bart(kspace), _maps_to_bart(maps))
    if recon_bart is None:
        raise RuntimeError("BART PICS returned None")
    return _bart_output_to_t1hw(recon_bart, num_frames=t_dim, height=h_dim, width=w_dim)


def bart_pics_from_batch(
    batch: Mapping[str, Any],
    *,
    bart_fn: Callable[..., np.ndarray],
    lamda: float = 0.01,
    command: str | None = None,
) -> torch.Tensor:
    """Run BART PICS on a dataloader batch and return [B,T,1,H,W]."""

    if "kspace_us" not in batch:
        raise KeyError("BART baseline requires batch['kspace_us']")
    if "maps" not in batch:
        raise KeyError("BART baseline requires batch['maps']")

    kspace_tensor = _ensure_batched_kspace(batch["kspace_us"])
    maps_tensor = _ensure_batched_maps(batch["maps"], batch_size=int(kspace_tensor.shape[0]))
    kspace_np = _numpy_complex(kspace_tensor, "kspace_us")
    maps_np = _numpy_complex(maps_tensor, "maps")

    recon = [
        bart_pics_single(
            kspace_np[batch_idx],
            maps_np[batch_idx],
            bart_fn=bart_fn,
            lamda=lamda,
            command=command,
        )
        for batch_idx in range(kspace_np.shape[0])
    ]
    out = torch.from_numpy(np.stack(recon, axis=0))
    return out.to(device=kspace_tensor.device)
