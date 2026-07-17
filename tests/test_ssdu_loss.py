from __future__ import annotations

import torch

from cardiac_ensure.losses import compute_ssdu_loss, split_ssdu_mask


def test_split_ssdu_mask_is_disjoint_reproducible_and_preserves_center() -> None:
    mask = torch.ones(1, 1, 1, 4, 8)

    train_a, loss_a = split_ssdu_mask(
        mask,
        rho=0.4,
        mask_type="uniform",
        small_acs_block=(4, 2),
        seed=123,
        sample_seeds=[456],
    )
    train_b, loss_b = split_ssdu_mask(
        mask,
        rho=0.4,
        mask_type="uniform",
        small_acs_block=(4, 2),
        seed=123,
        sample_seeds=[456],
    )

    assert torch.equal(train_a, train_b)
    assert torch.equal(loss_a, loss_b)
    assert torch.count_nonzero(loss_a) > 0
    assert torch.count_nonzero(train_a * loss_a) == 0
    assert torch.equal(train_a + loss_a, mask)
    assert torch.count_nonzero(loss_a[..., 3:5]) == 0
    assert torch.count_nonzero(train_a[..., 3:5]) == mask[..., 3:5].numel()


def test_compute_ssdu_loss_returns_finite_loss_and_masks() -> None:
    class IdentityModel(torch.nn.Module):
        def forward(self, zf: torch.Tensor, *, maps: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            del maps, mask
            return zf

    kspace = torch.ones(1, 1, 1, 4, 8, dtype=torch.complex64)
    maps = torch.ones(1, 1, 4, 8, dtype=torch.complex64)
    mask = torch.ones(1, 1, 1, 4, 8)
    batch = {
        "kspace_us": kspace,
        "maps": maps,
        "mask": mask,
    }

    out = compute_ssdu_loss(
        IdentityModel(),
        batch,
        rho=0.25,
        mask_type="gaussian",
        small_acs_block=(4, 2),
        seed=7,
        sample_seeds=[99],
    )

    assert torch.isfinite(out["loss"])
    assert out["prediction"].shape == (1, 1, 1, 4, 8)
    assert torch.count_nonzero(out["loss_mask"]) > 0
    assert torch.count_nonzero(out["train_mask"] * out["loss_mask"]) == 0
