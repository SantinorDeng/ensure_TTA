from __future__ import annotations

import numpy as np
import torch

from cardiac_ensure.baselines import bart_pics_from_batch, bart_pics_single, zero_filled_from_batch


def test_zero_filled_from_batch_uses_dataset_zf_magnitude() -> None:
    zf = torch.randn(1, 1, 1, 4, 5, dtype=torch.complex64)

    out = zero_filled_from_batch({"zf": zf})

    assert torch.allclose(out, torch.abs(zf))


def test_bart_pics_single_uses_expected_16d_shapes_and_output_shape() -> None:
    calls = []

    def fake_bart(num_args: int, command: str, kspace_bart: np.ndarray, maps_bart: np.ndarray) -> np.ndarray:
        calls.append((num_args, command, kspace_bart.shape, maps_bart.shape, kspace_bart.dtype, maps_bart.dtype))
        height, width, num_frames = kspace_bart.shape[0], kspace_bart.shape[1], kspace_bart.shape[10]
        return np.ones((height, width, num_frames), dtype=np.complex64)

    kspace = np.ones((2, 3, 4, 5), dtype=np.complex64)
    maps = np.ones((3, 4, 5), dtype=np.complex64)

    out = bart_pics_single(kspace, maps, bart_fn=fake_bart, lamda=0.01)

    assert out.shape == (2, 1, 4, 5)
    assert out.dtype == np.complex64
    assert calls == [
        (
            1,
            "pics -g -d0 -S -R T:1024:0:0.01",
            (4, 5, 1, 3, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1),
            (4, 5, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
            np.dtype("complex64"),
            np.dtype("complex64"),
        )
    ]


def test_bart_pics_from_batch_returns_batched_tensor() -> None:
    def fake_bart(num_args: int, command: str, kspace_bart: np.ndarray, maps_bart: np.ndarray) -> np.ndarray:
        del num_args, command, maps_bart
        return np.zeros((kspace_bart.shape[0], kspace_bart.shape[1], kspace_bart.shape[10]), dtype=np.complex64)

    batch = {
        "kspace_us": torch.ones(1, 1, 2, 4, 5, dtype=torch.complex64),
        "maps": torch.ones(1, 2, 4, 5, dtype=torch.complex64),
    }

    out = bart_pics_from_batch(batch, bart_fn=fake_bart, lamda=0.01)

    assert out.shape == (1, 1, 1, 4, 5)
    assert torch.is_complex(out)
