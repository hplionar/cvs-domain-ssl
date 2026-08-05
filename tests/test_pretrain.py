"""Tests for SSL pretraining: masking, schedule, checkpoint contract, clips."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from data.ssl_clip_dataset import ClipIndex, ClipTransform
from train.pretrain_videomae import (
    TrainState,
    TubeMaskGenerator,
    cosine_with_warmup,
    load_checkpoint,
    save_checkpoint,
)


# -- tube masking ---------------------------------------------------------


def test_mask_shape_and_ratio():
    m = TubeMaskGenerator(temporal=8, spatial=196, mask_ratio=0.9)
    mask = m()
    assert mask.shape == (8 * 196,)
    assert mask.dtype == torch.bool
    assert abs(mask.float().mean().item() - 0.9) < 0.01


def test_mask_is_a_tube():
    """The same spatial positions must be masked at every temporal position.

    Independent per-frame masking would let the model copy an unmasked patch
    from an adjacent frame, which at 90% masking makes the objective trivial on
    temporally redundant video.
    """
    m = TubeMaskGenerator(temporal=8, spatial=196, mask_ratio=0.9)
    grid = m().reshape(8, 196)
    for t in range(1, 8):
        assert torch.equal(grid[0], grid[t]), f"temporal position {t} differs from 0"


def test_mask_leaves_visible_patches():
    m = TubeMaskGenerator(temporal=8, spatial=196, mask_ratio=0.9)
    assert (~m()).sum() > 0


def test_mask_is_reproducible_from_generator():
    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)
    m = TubeMaskGenerator(8, 196, 0.9)
    assert torch.equal(m(g1), m(g2))


def test_mask_varies_across_calls():
    g = torch.Generator().manual_seed(0)
    m = TubeMaskGenerator(8, 196, 0.9)
    assert not torch.equal(m(g), m(g))


def test_batch_masks_differ_per_sample():
    m = TubeMaskGenerator(8, 196, 0.9)
    batch = m.batch(4, torch.Generator().manual_seed(0))
    assert batch.shape == (4, 8 * 196)
    assert not torch.equal(batch[0], batch[1])


@pytest.mark.parametrize("ratio", [0.0, 1.0, 1.5])
def test_degenerate_mask_ratio_rejected(ratio):
    with pytest.raises(ValueError, match="must leave at least one visible|mask_ratio"):
        TubeMaskGenerator(8, 196, ratio)


# -- schedule -------------------------------------------------------------

def test_warmup_is_linear_from_zero():
    assert cosine_with_warmup(0, 100, 1000) == 0.0
    assert cosine_with_warmup(50, 100, 1000) == pytest.approx(0.5)
    assert cosine_with_warmup(100, 100, 1000) == pytest.approx(1.0)


def test_cosine_decays_to_minimum():
    assert cosine_with_warmup(1000, 100, 1000) == pytest.approx(0.0, abs=1e-6)
    assert cosine_with_warmup(1000, 100, 1000, min_ratio=0.01) == pytest.approx(0.01, abs=1e-6)


def test_schedule_is_monotone_after_warmup():
    values = [cosine_with_warmup(s, 100, 1000) for s in range(100, 1000, 50)]
    assert all(a >= b for a, b in zip(values, values[1:]))


def test_schedule_clamps_past_total():
    assert cosine_with_warmup(2000, 100, 1000) == pytest.approx(0.0, abs=1e-6)


# -- checkpoint contract --------------------------------------------------


def _fixture():
    model = torch.nn.Linear(8, 4)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    for _ in range(3):  # populate optimiser moments
        opt.zero_grad()
        model(torch.randn(2, 8)).sum().backward()
        opt.step()
    return model, opt, scaler


def test_checkpoint_roundtrip_restores_everything(tmp_path):
    model, opt, scaler = _fixture()
    state = TrainState(step=137, epoch=3, best_loss=0.42, history=[{"step": 20, "loss": 1.0}])
    path = tmp_path / "latest.pt"
    save_checkpoint(path, model=model, optimizer=opt, scaler=scaler, state=state,
                    config={"seed": 0})

    model2 = torch.nn.Linear(8, 4)
    opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    scaler2 = torch.amp.GradScaler("cuda", enabled=False)
    restored = load_checkpoint(path, model=model2, optimizer=opt2, scaler=scaler2,
                              device=torch.device("cpu"))

    assert restored.step == 137
    assert restored.epoch == 3
    assert restored.best_loss == 0.42
    assert restored.history == [{"step": 20, "loss": 1.0}]
    assert torch.equal(model.weight, model2.weight)


def test_optimizer_moments_survive(tmp_path):
    """Adam's moments are the state most likely to be silently dropped; losing
    them makes the first steps after a resume behave like a fresh run."""
    model, opt, scaler = _fixture()
    path = tmp_path / "c.pt"
    save_checkpoint(path, model=model, optimizer=opt, scaler=scaler,
                    state=TrainState(step=1), config={})

    model2 = torch.nn.Linear(8, 4)
    opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    load_checkpoint(path, model=model2, optimizer=opt2,
                    scaler=torch.amp.GradScaler("cuda", enabled=False),
                    device=torch.device("cpu"))

    before = opt.state[list(opt.state)[0]]
    after = opt2.state[list(opt2.state)[0]]
    assert torch.allclose(before["exp_avg"], after["exp_avg"])
    assert torch.allclose(before["exp_avg_sq"], after["exp_avg_sq"])
    assert before["step"] == after["step"]


class _StubScaler:
    """Stands in for GradScaler, which disables itself without CUDA and then
    reports an empty state dict, hiding whether the state is actually saved."""

    def __init__(self, scale=1024.0):
        self._state = {"scale": scale, "growth_tracker": 7, "_growth_factor": 2.0}

    def state_dict(self):
        return dict(self._state)

    def load_state_dict(self, state):
        self._state = dict(state)


def test_scaler_state_is_saved_and_restored(tmp_path):
    """Omitting scaler state makes the loss scale re-warm on every resume,
    producing a spike at each job boundary that mimics genuine instability and
    wastes several hundred steps recovering."""
    model, opt, _ = _fixture()
    scaler = _StubScaler(scale=1024.0)
    path = tmp_path / "c.pt"
    save_checkpoint(path, model=model, optimizer=opt, scaler=scaler,
                    state=TrainState(step=1), config={})

    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["scaler"]["scale"] == 1024.0
    assert payload["scaler"]["growth_tracker"] == 7

    restored = _StubScaler(scale=65536.0)
    load_checkpoint(path, model=torch.nn.Linear(8, 4),
                    optimizer=torch.optim.AdamW(torch.nn.Linear(8, 4).parameters()),
                    scaler=restored, device=torch.device("cpu"))
    assert restored.state_dict()["scale"] == 1024.0, "loss scale was not restored"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GradScaler needs CUDA")
def test_real_gradscaler_state_roundtrips(tmp_path):
    model, opt, _ = _fixture()
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    path = tmp_path / "c.pt"
    save_checkpoint(path, model=model, optimizer=opt, scaler=scaler,
                    state=TrainState(step=1), config={})
    assert "scale" in torch.load(path, map_location="cpu", weights_only=False)["scaler"]


def test_rng_state_restores_sampling(tmp_path):
    model, opt, scaler = _fixture()
    random.seed(7); np.random.seed(7); torch.manual_seed(7)
    path = tmp_path / "c.pt"
    save_checkpoint(path, model=model, optimizer=opt, scaler=scaler,
                    state=TrainState(step=1), config={})
    expected = torch.randn(3)

    torch.manual_seed(999)
    load_checkpoint(path, model=torch.nn.Linear(8, 4),
                    optimizer=torch.optim.AdamW(torch.nn.Linear(8, 4).parameters()),
                    scaler=torch.amp.GradScaler("cuda", enabled=False),
                    device=torch.device("cpu"))
    assert torch.equal(torch.randn(3), expected)


def test_write_is_atomic(tmp_path):
    """A job killed mid-write must not leave a truncated checkpoint."""
    model, opt, scaler = _fixture()
    path = tmp_path / "latest.pt"
    save_checkpoint(path, model=model, optimizer=opt, scaler=scaler,
                    state=TrainState(step=1), config={})
    save_checkpoint(path, model=model, optimizer=opt, scaler=scaler,
                    state=TrainState(step=2), config={})
    assert not path.with_suffix(".tmp").exists()
    assert torch.load(path, map_location="cpu", weights_only=False)["step"] == 2


# -- clips ----------------------------------------------------------------


def test_clip_frame_indices():
    clip = ClipIndex("v.mp4", "v", start_frame=100, stride=4, num_frames=16)
    assert clip.frame_indices[0] == 100
    assert clip.frame_indices[-1] == 100 + 15 * 4
    assert clip.span_frames == 61


def test_clip_span_matches_videomae_kinetics_config():
    """16 frames at stride 4 spans 61 frames, 2.03 s at 30 fps — the temporal
    window VideoMAE was pretrained on."""
    clip = ClipIndex("v.mp4", "v", 0, stride=4, num_frames=16)
    assert clip.span_frames / 30.0 == pytest.approx(2.03, abs=0.01)


def test_transform_applies_one_crop_to_whole_clip():
    """Per-frame crops would introduce apparent motion absent from the video,
    which for a temporal objective is corruption rather than augmentation."""
    torch.manual_seed(0); random.seed(0)
    clip = torch.zeros(8, 3, 240, 320)
    clip[:, :, 100:110, 150:160] = 1.0        # identical marker in every frame
    out = ClipTransform(image_size=224, train=True)(clip)
    assert out.shape == (8, 3, 224, 224)
    for t in range(1, 8):
        assert torch.equal(out[0], out[t]), "frames diverged; crop was not shared"


def test_transform_eval_is_deterministic():
    clip = torch.rand(4, 3, 240, 320)
    t = ClipTransform(image_size=224, train=False)
    assert torch.allclose(t(clip), t(clip))


def test_transform_normalises():
    out = ClipTransform(image_size=224, train=False)(torch.full((4, 3, 240, 320), 0.485))
    assert abs(out[:, 0].mean().item()) < 0.1


# -- memory: the OOM that killed the first real run -----------------------


def test_transform_accepts_uint8_and_returns_float():
    """Clips stay uint8 until after cropping. Converting to float32 before the
    crop allocates the full decoded clip — 176 MiB at 1280x720 — which with
    several prefetching workers exhausts host memory."""
    out = ClipTransform(image_size=224, train=False)(
        torch.randint(0, 255, (8, 3, 256, 256), dtype=torch.uint8)
    )
    assert out.dtype == torch.float32
    assert out.shape == (8, 3, 224, 224)


def test_transform_uint8_and_float_paths_agree():
    clip_u8 = torch.randint(0, 255, (4, 3, 256, 256), dtype=torch.uint8)
    t = ClipTransform(image_size=224, train=False)
    assert torch.allclose(t(clip_u8), t(clip_u8.float().div(255.0)), atol=1e-5)


def test_reader_cache_is_bounded(tmp_path):
    """An unbounded cache holds one open container per video touched; under
    shuffling a worker eventually opens every video in the corpus."""
    from data.ssl_clip_dataset import SSLClipDataset

    class _Stub(SSLClipDataset):
        def __init__(self, cache_size):
            self.decode_size = 64
            self.reader_cache_size = cache_size
            from collections import OrderedDict
            self._readers = OrderedDict()

    stub = _Stub(cache_size=3)
    for i in range(10):
        stub._reader_for(f"/tmp/video{i}.mp4")
    assert len(stub._readers) == 3
    assert "/tmp/video9.mp4" in stub._readers
    assert "/tmp/video0.mp4" not in stub._readers


def test_reader_cache_is_lru():
    from collections import OrderedDict

    from data.ssl_clip_dataset import SSLClipDataset

    class _Stub(SSLClipDataset):
        def __init__(self):
            self.decode_size = 64
            self.reader_cache_size = 2
            self._readers = OrderedDict()

    stub = _Stub()
    stub._reader_for("a.mp4")
    stub._reader_for("b.mp4")
    stub._reader_for("a.mp4")   # refresh a
    stub._reader_for("c.mp4")   # should evict b, not a
    assert set(stub._readers) == {"a.mp4", "c.mp4"}


# -- bias repair must apply to the pretraining path -----------------------


def test_build_model_applies_bias_repair(monkeypatch, tmp_path):
    """The E1 baseline loads through VideoMAEEncoder, which repairs the BEiT-
    layout attention biases. If pretraining did not, the adapted run would start
    from different weights than the baseline it is compared against, and
    adaptation gain would absorb the discrepancy."""
    from transformers import VideoMAEConfig, VideoMAEForPreTraining

    import train.pretrain_videomae as module

    cfg = VideoMAEConfig(
        image_size=224, patch_size=16, num_frames=16, tubelet_size=2,
        hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=32, decoder_hidden_size=16, decoder_num_hidden_layers=1,
        decoder_num_attention_heads=2, decoder_intermediate_size=16,
    )
    saved = VideoMAEForPreTraining(cfg)
    saved.save_pretrained(tmp_path)

    calls = {}

    def _spy(model, name):
        calls["name"] = name
        return {"status": "repaired", "num_repaired": 2}

    monkeypatch.setattr("models.encoders.videomae_encoder.repair_qkv_bias", _spy)

    model, repair = module.build_model(
        {"model": {"checkpoint": str(tmp_path)}}, torch.device("cpu")
    )
    assert calls["name"] == str(tmp_path)
    assert repair["status"] == "repaired"


def test_failed_repair_stops_training(monkeypatch, tmp_path):
    """Silently training from mismatched weights is worse than not starting."""
    from transformers import VideoMAEConfig, VideoMAEForPreTraining

    import train.pretrain_videomae as module

    cfg = VideoMAEConfig(
        image_size=224, patch_size=16, num_frames=16, tubelet_size=2,
        hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=32, decoder_hidden_size=16, decoder_num_hidden_layers=1,
        decoder_num_attention_heads=2, decoder_intermediate_size=16,
    )
    VideoMAEForPreTraining(cfg).save_pretrained(tmp_path)

    monkeypatch.setattr(
        "models.encoders.videomae_encoder.repair_qkv_bias",
        lambda model, name: {"status": "failed", "num_repaired": 0},
    )

    with pytest.raises(RuntimeError, match="invalidate the adaptation-gain"):
        module.build_model({"model": {"checkpoint": str(tmp_path)}}, torch.device("cpu"))


# -- decoder return types -------------------------------------------------


def test_decord_ndarray_is_converted_via_asnumpy():
    """decord returns its own NDArray, which exposes asnumpy() rather than
    numpy(). np.asarray silently yields an object array instead of failing,
    producing a confusing TypeError several frames later at from_numpy."""
    from data.ssl_clip_dataset import _to_numpy

    class _DecordNDArray:
        def __init__(self, array):
            self._array = array

        def asnumpy(self):
            return self._array

    expected = np.zeros((3, 256, 256, 3), dtype=np.uint8)
    out = _to_numpy(_DecordNDArray(expected))
    assert out.dtype == np.uint8
    assert out.shape == (3, 256, 256, 3)


def test_torch_tensor_batch_is_converted():
    from data.ssl_clip_dataset import _to_numpy

    out = _to_numpy(torch.zeros(2, 8, 8, 3, dtype=torch.uint8))
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.uint8


def test_plain_numpy_passes_through():
    from data.ssl_clip_dataset import _to_numpy

    array = np.ones((2, 4, 4, 3), dtype=np.uint8)
    assert np.array_equal(_to_numpy(array), array)


def test_unconvertible_batch_reports_clearly():
    """A ragged or opaque batch must fail with a message naming the cause,
    rather than as a TypeError at torch.from_numpy several frames later."""
    from data.ssl_clip_dataset import _to_numpy

    ragged = np.empty(2, dtype=object)
    ragged[0] = np.zeros((4, 4, 3), dtype=np.uint8)
    ragged[1] = np.zeros((8, 8, 3), dtype=np.uint8)

    with pytest.raises(TypeError, match="unrecognised type|dtype object"):
        _to_numpy(ragged)


# -- bias repair across key layouts ---------------------------------------


def test_repair_handles_prefixed_model_keys():
    """VideoMAEForPreTraining keeps a `videomae.` prefix on encoder keys while
    VideoMAEModel does not. Stripping it unconditionally maps the decoder
    tensors and silently misses all 24 encoder ones."""
    from transformers import VideoMAEConfig, VideoMAEForPreTraining

    cfg = VideoMAEConfig(
        image_size=224, patch_size=16, num_frames=16, tubelet_size=2,
        hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
        intermediate_size=32, decoder_hidden_size=16, decoder_num_hidden_layers=1,
        decoder_num_attention_heads=2, decoder_intermediate_size=16,
    )
    model = VideoMAEForPreTraining(cfg)
    keys = set(model.state_dict())

    prefixed = [k for k in keys if k.startswith("videomae.") and k.endswith("query.bias")]
    assert prefixed, "encoder keys should carry the videomae. prefix"

    # The mapping the repair performs, applied to a prefixed source key.
    source = prefixed[0].replace(".query.bias", ".q_bias")
    base = source.replace(".q_bias", ".query.bias")
    import re as _re
    stripped = _re.sub(r"^videomae\.", "", base)
    target = next((c for c in (base, stripped, f"videomae.{stripped}") if c in keys), None)
    assert target == prefixed[0]


# -- overfitting check support --------------------------------------------


def _make_videos(tmp_path, count=6, frames=100):
    import av

    directory = tmp_path / "videos"
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for v in range(count):
        container = av.open(str(directory / f"vid{v:02d}.mp4"), "w")
        stream = container.add_stream("libx264", rate=30)
        stream.width, stream.height, stream.pix_fmt = 160, 120, "yuv420p"
        base = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
        for t in range(frames):
            frame = av.VideoFrame.from_ndarray(np.roll(base, t, axis=1), format="rgb24")
            container.mux(stream.encode(frame))
        for packet in stream.encode():
            container.mux(packet)
        container.close()
    return directory


def test_limit_videos_truncates_the_corpus(tmp_path):
    """The overfitting check is a handful of videos seen many times. Without a
    way to truncate, a 'small' run still draws from 700 procedures and each
    clip is seen about three times, which cannot show memorisation either way."""
    pytest.importorskip("av")
    from data.ssl_clip_dataset import SSLClipDataset

    directory = _make_videos(tmp_path, count=6)
    full = SSLClipDataset(directory, num_frames=4, stride=4, clips_per_video=2)
    limited = SSLClipDataset(directory, num_frames=4, stride=4, clips_per_video=2,
                             limit_videos=2)
    assert full.describe()["num_videos"] == 6
    assert limited.describe()["num_videos"] == 2
    assert len(limited) < len(full)


def test_limit_videos_is_recorded(tmp_path):
    pytest.importorskip("av")
    from data.ssl_clip_dataset import SSLClipDataset

    directory = _make_videos(tmp_path, count=3)
    dataset = SSLClipDataset(directory, num_frames=4, stride=4, clips_per_video=1,
                             limit_videos=2)
    assert dataset.describe()["limit_videos"] == 2


def test_limit_videos_selection_is_deterministic(tmp_path):
    pytest.importorskip("av")
    from data.ssl_clip_dataset import SSLClipDataset

    directory = _make_videos(tmp_path, count=5)
    a = SSLClipDataset(directory, num_frames=4, stride=4, clips_per_video=1, limit_videos=3)
    b = SSLClipDataset(directory, num_frames=4, stride=4, clips_per_video=1, limit_videos=3)
    assert [c.video_id for c in a.clips] == [c.video_id for c in b.clips]


def test_deterministic_transform_gives_identical_output():
    """With random crops the model never sees the same input twice, so a flat
    loss during an overfitting check would be uninformative."""
    clip = torch.randint(0, 255, (4, 3, 256, 256), dtype=torch.uint8)
    t = ClipTransform(image_size=224, train=False)
    assert torch.equal(t(clip), t(clip))


def test_augmented_transform_varies():
    clip = torch.randint(0, 255, (4, 3, 256, 256), dtype=torch.uint8)
    t = ClipTransform(image_size=224, train=True)
    random.seed(1)
    first = t(clip)
    random.seed(2)
    assert not torch.equal(first, t(clip))