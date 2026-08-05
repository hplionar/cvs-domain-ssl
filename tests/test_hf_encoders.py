"""Tests for the HuggingFace-backed encoder wrappers.

Models are built from tiny configs rather than downloaded, so the suite runs
offline and in seconds while still exercising real geometry (224 px, patch 16,
16 frames, tubelet 2).
"""

from __future__ import annotations

import pytest
import torch

transformers = pytest.importorskip("transformers")

from models.encoders import build_encoder, available_encoders  # noqa: E402
from models.encoders.dinov3_encoder import DINOv3Encoder  # noqa: E402
from models.encoders.mae_encoder import MAEEncoder  # noqa: E402
from models.encoders.videomae_encoder import VideoMAEEncoder  # noqa: E402
from models.encoders.vjepa2_encoder import VJEPA2Encoder  # noqa: E402

TINY = dict(hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32)


@pytest.fixture(scope="module")
def videomae():
    from transformers import VideoMAEConfig, VideoMAEModel

    cfg = VideoMAEConfig(image_size=224, patch_size=16, num_frames=16, tubelet_size=2, **TINY)
    return VideoMAEEncoder(model=VideoMAEModel(cfg))


@pytest.fixture(scope="module")
def vjepa2():
    from transformers import VJEPA2Config, VJEPA2Model

    cfg = VJEPA2Config(
        crop_size=224, patch_size=16, frames_per_clip=16, tubelet_size=2,
        hidden_size=32, num_hidden_layers=1, num_attention_heads=2, mlp_ratio=1,
        pred_hidden_size=16, pred_num_hidden_layers=1, pred_num_attention_heads=2,
    )
    return VJEPA2Encoder(model=VJEPA2Model(cfg))


@pytest.fixture(scope="module")
def mae():
    from transformers import ViTMAEConfig, ViTMAEModel

    cfg = ViTMAEConfig(image_size=224, patch_size=16, **TINY)
    return MAEEncoder(model=ViTMAEModel(cfg))


def _dinov3(num_registers: int):
    from transformers import DINOv3ViTConfig, DINOv3ViTModel

    cfg = DINOv3ViTConfig(
        image_size=224, patch_size=16, num_register_tokens=num_registers, **TINY
    )
    return DINOv3Encoder(model=DINOv3ViTModel(cfg))


# -- geometry -------------------------------------------------------------


def test_videomae_grid(videomae):
    assert videomae.token_layout.grid == (8, 14, 14)
    assert videomae.token_layout.num_prefix_tokens == 0
    out = videomae(torch.randn(1, 16, 3, 224, 224))
    assert out.tokens.shape == (1, 1568, 32)
    assert out.prefix is None


def test_vjepa2_grid(vjepa2):
    assert vjepa2.token_layout.grid == (8, 14, 14)
    out = vjepa2(torch.randn(1, 16, 3, 224, 224))
    assert out.tokens.shape == (1, 1568, 32)


def test_video_arms_have_identical_layout(videomae, vjepa2):
    """E2 compares these two directly; mismatched geometry invalidates it."""
    assert videomae.token_layout.grid == vjepa2.token_layout.grid
    assert videomae.preprocess_spec == vjepa2.preprocess_spec


def test_mae_grid_and_cls(mae):
    assert mae.token_layout.grid == (14, 14)
    assert mae.token_layout.num_prefix_tokens == 1
    out = mae(torch.randn(2, 3, 224, 224))
    assert out.tokens.shape == (2, 196, 32)
    assert out.prefix.shape == (2, 1, 32)


@pytest.mark.parametrize("num_registers", [0, 4])
def test_dinov3_register_tokens_are_separated(num_registers):
    enc = _dinov3(num_registers)
    assert enc.num_register_tokens == num_registers
    assert enc.token_layout.num_prefix_tokens == 1 + num_registers
    out = enc(torch.randn(1, 3, 224, 224))
    assert out.tokens.shape == (1, 196, 32), "registers must not enter the patch grid"
    assert out.prefix.shape == (1, 1 + num_registers, 32)


# -- the ViT-MAE masking trap --------------------------------------------


def test_mae_masking_is_disabled(mae):
    assert mae.model.config.mask_ratio == 0.0


def test_mae_is_deterministic(mae):
    """ViTMAEModel drops a random 75% of tokens per call by default."""
    x = torch.randn(2, 3, 224, 224)
    a = mae.extract(x, to_dtype=None)
    b = mae.extract(x, to_dtype=None)
    assert torch.equal(a.tokens, b.tokens)


def test_mae_token_order_is_not_permuted(mae):
    """At mask_ratio=0 all tokens are kept, but ViT-MAE still shuffles them.

    Mean pooling is permutation-invariant and would hide this; reshaping to the
    14x14 grid, and every attention map, would be silently scrambled.
    """
    x = torch.randn(1, 3, 224, 224)
    raw = mae.model(pixel_values=x)
    assert not torch.equal(
        raw.ids_restore[0], torch.arange(196)
    ), "upstream no longer shuffles; this test is obsolete"

    out = mae(x)
    fixed = mae.model(pixel_values=x, noise=mae._monotonic_noise(1, x.device))
    assert torch.equal(fixed.ids_restore[0], torch.arange(196))
    assert torch.equal(out.tokens, fixed.last_hidden_state[:, 1:, :])


def test_mae_grid_reshape_is_stable(mae):
    x = torch.randn(1, 3, 224, 224)
    a = mae.extract(x, to_dtype=None).tokens.reshape(1, 14, 14, 32)
    b = mae.extract(x, to_dtype=None).tokens.reshape(1, 14, 14, 32)
    assert torch.equal(a, b)


def test_mae_wrapper_repairs_a_masking_model():
    """A caller-supplied model with masking still on must be corrected."""
    from transformers import ViTMAEConfig, ViTMAEModel

    cfg = ViTMAEConfig(image_size=224, patch_size=16, mask_ratio=0.75, **TINY)
    enc = MAEEncoder(model=ViTMAEModel(cfg))
    assert enc.model.config.mask_ratio == 0.0
    assert enc(torch.randn(1, 3, 224, 224)).tokens.shape == (1, 196, 32)


# -- V-JEPA 2 predictor and provenance -----------------------------------


def test_vjepa2_records_distillation_status(vjepa2):
    record = vjepa2.describe()
    assert record["distilled"] is False
    assert "variant" in record


def test_vjepa2_skips_predictor(vjepa2, monkeypatch):
    seen = {}
    original = vjepa2.model.forward

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(vjepa2.model, "forward", spy)
    vjepa2(torch.randn(1, 16, 3, 224, 224))
    assert seen.get("skip_predictor") is True


# -- shared contract ------------------------------------------------------


@pytest.mark.parametrize("name", ["videomae", "vjepa2", "mae"])
def test_all_frozen_and_extract_to_fp16(name, request):
    enc = request.getfixturevalue(name)
    assert enc.is_frozen
    shape = (1, 16, 3, 224, 224) if enc.modality == "video" else (1, 3, 224, 224)
    out = enc.extract(torch.randn(*shape))
    assert out.tokens.dtype == torch.float16
    assert not out.tokens.requires_grad


@pytest.mark.parametrize("name", ["videomae", "vjepa2", "mae"])
def test_describe_is_complete(name, request):
    enc = request.getfixturevalue(name)
    record = enc.describe()
    assert record["preprocess"]["image_size"] == 224
    assert record["token_layout"]["dim"] == 32
    assert record["num_parameters"] > 0


def test_registry_exposes_all_wrappers():
    for name in ["videomae_b", "videomae_l", "vjepa2_l", "vjepa2_b", "mae_b", "mae_l",
                 "dinov3_s", "dinov3_b"]:
        assert name in available_encoders()


def test_unknown_variant_is_rejected():
    with pytest.raises(ValueError, match="Unknown variant"):
        VideoMAEEncoder(variant="enormous")


# -- VideoMAE attention bias repair (transformers 5.x regression) ---------


def test_repair_maps_q_and_v_bias_and_zeros_key(monkeypatch, tmp_path):
    """VideoMAE checkpoints store q_bias/v_bias; transformers 5.x expects
    query/key/value.bias and provides no conversion, so trained biases are
    silently replaced with newly initialised ones."""
    import torch
    from safetensors.torch import save_file

    from transformers import VideoMAEConfig, VideoMAEModel
    from models.encoders.videomae_encoder import repair_qkv_bias

    cfg = VideoMAEConfig(image_size=224, patch_size=16, num_frames=16,
                         tubelet_size=2, qkv_bias=True, **TINY)
    model = VideoMAEModel(cfg)

    dim = cfg.hidden_size
    q = torch.full((dim,), 0.25)
    v = torch.full((dim,), -0.5)
    path = tmp_path / "model.safetensors"
    save_file({
        "videomae.encoder.layer.0.attention.attention.q_bias": q,
        "videomae.encoder.layer.0.attention.attention.v_bias": v,
    }, str(path))

    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download", lambda *a, **k: str(path)
    )

    with torch.no_grad():  # start from values that are definitely wrong
        model.encoder.layer[0].attention.attention.query.bias.fill_(9.0)
        model.encoder.layer[0].attention.attention.value.bias.fill_(9.0)
        model.encoder.layer[0].attention.attention.key.bias.fill_(9.0)

    record = repair_qkv_bias(model, "MCG-NJU/videomae-base")

    assert record["status"] == "repaired"
    assert record["num_repaired"] == 2
    assert record["num_unmapped"] == 0

    attn = model.encoder.layer[0].attention.attention
    assert torch.allclose(attn.query.bias, q)
    assert torch.allclose(attn.value.bias, v)
    assert torch.allclose(attn.key.bias, torch.zeros(dim)), \
        "the original implementation fixes key bias at zero"


def test_repair_is_a_no_op_for_modern_checkpoints(monkeypatch, tmp_path):
    import torch
    from safetensors.torch import save_file
    from transformers import VideoMAEConfig, VideoMAEModel
    from models.encoders.videomae_encoder import repair_qkv_bias

    cfg = VideoMAEConfig(image_size=224, patch_size=16, num_frames=16,
                         tubelet_size=2, **TINY)
    model = VideoMAEModel(cfg)
    path = tmp_path / "model.safetensors"
    save_file({"encoder.layer.0.attention.attention.query.bias":
               torch.zeros(cfg.hidden_size)}, str(path))
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda *a, **k: str(path))

    assert repair_qkv_bias(model, "some/checkpoint")["status"] == "not_needed"


def test_repair_status_recorded_in_describe(videomae):
    """The cache manifest must show whether the repair ran, so that two arms
    cannot silently differ in whether their weights were correct."""
    assert "qkv_bias_repair" in videomae.describe()