from __future__ import annotations

import torch

from cardiac_ensure.models import TemporalNormUnet
from cardiac_ensure.scripts.tta_shift_true_ensure_lora import (
    ConvLoRA2d,
    count_trainable_parameters,
    inject_lora_adapters,
    select_dc_only_parameters,
)


def _make_model() -> TemporalNormUnet:
    return TemporalNormUnet(
        num_frames=1,
        chans=8,
        num_pools=4,
        num_unrolls=2,
        denoiser_sharing="shared",
    )


def test_zero_initialized_lora_preserves_checkpoint_output() -> None:
    torch.manual_seed(7)
    model = _make_model().eval()
    image = torch.randn(1, 1, 16, 16, 2)
    with torch.no_grad():
        expected = model(image)

    names = inject_lora_adapters(
        model,
        rank=2,
        alpha=2.0,
        layer_indices=[1, 2, 3],
    )
    with torch.no_grad():
        actual = model(image)

    assert torch.equal(actual, expected)
    assert len(names) == 6
    assert count_trainable_parameters(model) == 3 * (8 * 2 * 3 * 3 + 8 * 2)


def test_only_lora_parameters_receive_gradients() -> None:
    torch.manual_seed(7)
    model = _make_model().train()
    inject_lora_adapters(
        model,
        rank=2,
        alpha=2.0,
        layer_indices=[1, 2, 3],
    )
    image = torch.randn(1, 1, 16, 16, 2)
    model(image).square().mean().backward()

    trainable = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    frozen = {name: parameter for name, parameter in model.named_parameters() if not parameter.requires_grad}
    assert trainable
    assert all("lora_" in name for name in trainable)
    assert all(parameter.grad is None for parameter in frozen.values())
    assert any(parameter.grad is not None for name, parameter in trainable.items() if "lora_up" in name)


def test_adapter_can_optionally_include_dc_scalars() -> None:
    model = _make_model()
    inject_lora_adapters(
        model,
        rank=1,
        alpha=1.0,
        layer_indices=[2],
        adapt_dc=True,
    )
    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert isinstance(model.norm_unet.cnn.layers[2].layers[0], ConvLoRA2d)
    assert trainable_names[-2:] == ["dc_blocks.0.dc_weight", "dc_blocks.1.dc_weight"]


def test_independent_denoisers_each_receive_adapters() -> None:
    model = TemporalNormUnet(
        num_frames=1,
        chans=8,
        num_pools=4,
        num_unrolls=3,
        denoiser_sharing="independent",
    )
    names = inject_lora_adapters(
        model,
        rank=2,
        alpha=2.0,
        layer_indices=[1, 2, 3],
        adapt_dc=True,
    )

    assert len(names) == 3 * 3 * 2 + 3
    assert count_trainable_parameters(model) == 3 * 3 * (8 * 2 * 3 * 3 + 8 * 2) + 3
    assert all(
        isinstance(denoiser.cnn.layers[2].layers[0], ConvLoRA2d)
        for denoiser in model.norm_unets
    )


def test_dc_only_selects_no_denoiser_parameters() -> None:
    model = _make_model()
    names = select_dc_only_parameters(model)

    assert names == ["dc_blocks.0.dc_weight", "dc_blocks.1.dc_weight"]
    assert count_trainable_parameters(model) == 2
    assert all(
        name.startswith("dc_blocks.")
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
