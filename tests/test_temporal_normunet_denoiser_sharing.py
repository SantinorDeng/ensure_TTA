from __future__ import annotations

import pytest
import torch

from cardiac_ensure.models import TemporalNormUnet
from cardiac_ensure.scripts.tta_shift_supervised_baseline import (
    load_model_from_checkpoint as load_supervised_tta_model,
)
from cardiac_ensure.scripts.tta_shift_true_ensure import (
    load_model_from_checkpoint as load_ensure_tta_model,
)


def _small_model(denoiser_sharing: str = "shared") -> TemporalNormUnet:
    return TemporalNormUnet(
        num_frames=1,
        chans=4,
        num_pools=1,
        num_unrolls=3,
        use_data_consistency=False,
        denoiser_sharing=denoiser_sharing,
    )


def test_shared_mode_preserves_legacy_state_dict_layout() -> None:
    default_model = _small_model()
    explicit_shared_model = _small_model("shared")

    assert hasattr(default_model, "norm_unet")
    assert not hasattr(default_model, "norm_unets")
    assert any(key.startswith("norm_unet.") for key in default_model.state_dict())
    default_model.load_state_dict(explicit_shared_model.state_dict(), strict=True)


def test_independent_mode_uses_and_trains_each_denoiser() -> None:
    model = _small_model("independent")
    input_image = torch.randn(1, 1, 1, 16, 16, dtype=torch.complex64)

    output = model(input_image)
    output.abs().mean().backward()

    assert output.shape == input_image.shape
    assert len(model.norm_unets) == model.num_unrolls
    assert not hasattr(model, "norm_unet")
    assert any(key.startswith("norm_unets.0.") for key in model.state_dict())
    assert all(
        any(parameter.grad is not None for parameter in denoiser.parameters())
        for denoiser in model.norm_unets
    )


def test_independent_mode_has_more_parameters_than_shared_mode() -> None:
    shared = _small_model("shared")
    independent = _small_model("independent")

    shared_params = sum(parameter.numel() for parameter in shared.parameters())
    independent_params = sum(parameter.numel() for parameter in independent.parameters())

    assert independent_params > shared_params


def test_invalid_denoiser_sharing_is_rejected() -> None:
    with pytest.raises(ValueError, match="denoiser_sharing"):
        _small_model("invalid")


@pytest.mark.parametrize(
    "loader",
    (load_supervised_tta_model, load_ensure_tta_model),
)
@pytest.mark.parametrize("denoiser_sharing", ("shared", "independent"))
def test_tta_checkpoint_loader_restores_denoiser_sharing(
    tmp_path,
    loader,
    denoiser_sharing: str,
) -> None:
    model = _small_model(denoiser_sharing)
    config = {
        "window_size": 1,
        "chans": 4,
        "num_pools": 1,
        "num_unrolls": 3,
        "drop_prob": 0.0,
        "no_residual": False,
    }
    if denoiser_sharing == "independent":
        config["denoiser_sharing"] = denoiser_sharing
    checkpoint_path = tmp_path / f"{denoiser_sharing}.pt"
    torch.save(
        {
            "config": config,
            "model_state_dict": model.state_dict(),
        },
        checkpoint_path,
    )

    restored, restored_config = loader(checkpoint_path, torch.device("cpu"))

    assert restored.denoiser_sharing == denoiser_sharing
    assert restored_config == config
