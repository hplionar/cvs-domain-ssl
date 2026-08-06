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
from models.encoders.dinov2_encoder import DINOv2ViTEncoder  # noqa: E402
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


# -- DINOv2: resolution is chosen, not inherited --------------------------


def _dinov2(num_registers: int = 0, image_size: int | None = None):
    """Built from a config declaring 518 px, as the released checkpoints do."""
    if num_registers:
        from transformers import Dinov2WithRegistersConfig, Dinov2WithRegistersModel

        cfg = Dinov2WithRegistersConfig(
            image_size=518, patch_size=14, num_register_tokens=num_registers, **TINY
        )
        model = Dinov2WithRegistersModel(cfg)
    else:
        from transformers import Dinov2Config, Dinov2Model

        model = Dinov2Model(Dinov2Config(image_size=518, patch_size=14, **TINY))
    return DINOv2ViTEncoder(model=model, image_size=image_size)


def test_dinov2_uses_224_not_the_configs_518():
    """The checkpoint config declares 518 px, i.e. a 37x37 grid and 1369 tokens.

    Inheriting that would multiply the cache and the extraction cost sevenfold
    at a resolution no other arm uses, so the wrapper pins 224 instead. DINOv2
    interpolates its position encodings, which makes that legitimate -- and
    silent, which is why it is asserted here.
    """
    enc = _dinov2()
    assert enc.model.config.image_size == 518
    assert enc.preprocess_spec.image_size == 224
    assert enc.token_layout.grid == (16, 16)
    assert enc.token_layout.num_tokens == 256

    out = enc(torch.randn(2, 3, 224, 224))
    assert out.tokens.shape == (2, 256, 32)
    assert out.prefix.shape == (2, 1, 32), "DINOv2 has a CLS token and no registers"


def test_dinov2_honours_an_explicit_resolution():
    enc = _dinov2(image_size=252)
    assert enc.token_layout.grid == (18, 18)
    assert enc(torch.randn(1, 3, 252, 252)).tokens.shape == (1, 324, 32)


def test_dinov2_rejects_a_resolution_that_is_not_a_multiple_of_the_patch():
    """256 is the obvious thing to try and is wrong: 256/14 is not an integer.

    Without the guard the model silently interpolates and returns a token count
    that disagrees with the declared grid.
    """
    with pytest.raises(ValueError, match="not divisible by DINOv2's patch size"):
        _dinov2(image_size=256)


def test_dinov2_rejects_input_at_the_wrong_resolution():
    """BaseEncoder validates against preprocess_spec, so a 518 px batch fails
    rather than quietly producing 1369 tokens against a 256-token layout."""
    enc = _dinov2()
    with pytest.raises(ValueError, match="Expected 224x224 input"):
        enc(torch.randn(1, 3, 518, 518))


@pytest.mark.parametrize("num_registers", [0, 4])
def test_dinov2_with_registers_keeps_them_out_of_the_patch_grid(num_registers):
    """facebook/dinov2-with-registers-* exists. Assuming zero registers would
    shift every patch token by four positions without raising."""
    enc = _dinov2(num_registers=num_registers)
    assert enc.num_register_tokens == num_registers
    assert enc.token_layout.num_prefix_tokens == 1 + num_registers
    out = enc(torch.randn(1, 3, 224, 224))
    assert out.tokens.shape == (1, 256, 32)
    assert out.prefix.shape == (1, 1 + num_registers, 32)


def test_dinov2_checkpoint_id_records_the_resolution():
    """Two caches at different resolutions are not comparable, and the manifest
    is where that has to be visible."""
    a = _dinov2(image_size=224).checkpoint_id
    b = _dinov2(image_size=252).checkpoint_id
    assert a != b
    assert "224" in a and "252" in b


def test_dinov2_cls_is_prefix_index_zero():
    """FusionHead reads prefix[:, 0] as h_ctx."""
    enc = _dinov2(num_registers=4)
    x = torch.randn(1, 3, 224, 224)
    hidden = enc.model(pixel_values=x).last_hidden_state
    assert torch.equal(enc(x).prefix[:, 0], hidden[:, 0])


def test_dinov2_ignores_a_register_count_the_architecture_does_not_implement():
    """Dinov2Config accepts num_register_tokens; Dinov2Model ignores it.

    Trusting the config here would declare a 5-token prefix against a model that
    emits 1, shifting every patch token by four positions. The count must come
    from the architecture, not the config.
    """
    from transformers import Dinov2Config, Dinov2Model

    cfg = Dinov2Config(image_size=518, patch_size=14, num_register_tokens=4, **TINY)
    assert cfg.num_register_tokens == 4, "config still accepts the field"

    enc = DINOv2ViTEncoder(model=Dinov2Model(cfg))
    assert enc.num_register_tokens == 0
    assert enc.token_layout.num_prefix_tokens == 1

    out = enc(torch.randn(1, 3, 224, 224))
    assert out.tokens.shape == (1, 256, 32)
    assert out.prefix.shape == (1, 1, 32)


def test_dinov2_legacy_wrapper_is_untouched():
    """train_cvs.py, train_cvs_clip_meanpool.py and both eval scripts import the
    pooled torch.hub wrapper; exp001-exp006 came through it."""
    from models.encoders.dinov2_encoder import DINOv2Encoder

    assert not issubclass(DINOv2Encoder, DINOv2ViTEncoder)
    assert not hasattr(DINOv2Encoder, "token_layout")


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
                 "dinov3_s", "dinov3_b", "dinov2_s", "dinov2_b", "dinov2_l"]:
        assert name in available_encoders()


def test_registry_exposes_an_ungated_encoder_with_a_cls_token():
    """The fusion head's global branch needs prefix tokens, and DINOv3 is behind
    a manual access review. At least one self-distilled, ungated alternative
    must be reachable by name or the comparison blocks on someone's approval
    queue."""
    assert "dinov2_b" in available_encoders()


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


# -- adapted checkpoint loading -------------------------------------------
#
# The route from pretraining to evaluation. Without it an adapted encoder
# cannot be probed, and adaptation gain -- the outcome the whole comparison
# rests on -- cannot be computed at all.


def _base_and_adapted(tmp_path, *, offset=1.0, drop_half=False):
    """A saved base checkpoint plus a checkpoint shaped like pretraining output."""
    from transformers import VideoMAEConfig, VideoMAEModel

    cfg = VideoMAEConfig(image_size=224, patch_size=16, num_frames=16,
                         tubelet_size=2, **TINY)
    base_dir = tmp_path / "base"
    VideoMAEModel(cfg).save_pretrained(base_dir)

    reference = VideoMAEModel(cfg).state_dict()
    adapted = {f"videomae.{k}": v + offset for k, v in reference.items()}
    if drop_half:
        adapted = {k: v for i, (k, v) in enumerate(adapted.items()) if i % 2 == 0}

    path = tmp_path / "latest.pt"
    torch.save({"model": adapted, "step": 500,
                "config": {"model": {"checkpoint": str(base_dir)}}}, path)
    return path, base_dir, reference


def test_adapted_checkpoint_loads_trained_weights(tmp_path):
    from models.encoders.videomae_encoder import load_adapted_checkpoint

    path, base_dir, reference = _base_and_adapted(tmp_path, offset=1.0)
    model, base, record = load_adapted_checkpoint(path, base_checkpoint=str(base_dir))

    loaded = model.state_dict()
    key = next(k for k in loaded if k.endswith("layernorm_before.weight"))
    assert torch.allclose(loaded[key], reference[key] + 1.0), "adapted weights not loaded"
    assert str(base_dir) in str(base)


def test_base_checkpoint_read_from_the_training_config(tmp_path):
    """The training config records which checkpoint the run started from, so the
    architecture can be reconstructed without the caller repeating it."""
    from models.encoders.videomae_encoder import load_adapted_checkpoint

    path, base_dir, _ = _base_and_adapted(tmp_path)
    _, base, _ = load_adapted_checkpoint(path)
    assert str(base_dir) in str(base)


def test_missing_file_is_rejected(tmp_path):
    from models.encoders.videomae_encoder import load_adapted_checkpoint

    with pytest.raises(FileNotFoundError):
        load_adapted_checkpoint(tmp_path / "absent.pt")


def test_wrong_payload_shape_is_rejected(tmp_path):
    from models.encoders.videomae_encoder import load_adapted_checkpoint

    path = tmp_path / "bad.pt"
    torch.save({"not_a_model": 1}, path)
    with pytest.raises(ValueError, match="does not look like a checkpoint"):
        load_adapted_checkpoint(path)


def test_missing_encoder_keys_are_fatal(tmp_path):
    """A partial state dict would leave freshly initialised weights in the
    encoder, which is precisely the failure this path exists to prevent."""
    from models.encoders.videomae_encoder import load_adapted_checkpoint

    path, base_dir, _ = _base_and_adapted(tmp_path, drop_half=True)
    with pytest.raises((RuntimeError, ValueError)):
        load_adapted_checkpoint(path, base_checkpoint=str(base_dir))