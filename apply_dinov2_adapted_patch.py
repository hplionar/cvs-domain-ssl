#!/usr/bin/env python3
"""Add ``adapted_checkpoint`` support to ``DINOv2ViTEncoder``.

Without this, ``extract_features.py --checkpoint`` fails for DINOv2 with
"Encoder 'dinov2_b' does not support --checkpoint", and the arm adapted by
``train/pretrain_dino.py`` has no route into evaluation.

This is the fifth instance in this project of a capability existing for one
encoder and silently not for another. The previous four were: `--checkpoint`
missing for V-JEPA, `compare_weights.py` assuming VideoMAE, the VideoMAE bias
repair applied in the wrapper but not in pretraining, and `output_hidden_states`
unsupported by V-JEPA. The pattern is consistent enough to be worth naming: a
tool written while working on one architecture, tested against that
architecture, then failing on the other.

The DINOv2 loader is simpler than either existing one:

- **No bias repair.** DINOv2 stores query, key and value biases under the names
  transformers expects, unlike VideoMAE's BEiT layout.
- **No submodule prefix.** ``pretrain_dino.py`` exports
  ``teacher.backbone.state_dict()``, which is a ``Dinov2Model`` state dict
  directly. V-JEPA needed an ``encoder.`` prefix stripped; VideoMAE needed the
  ``videomae`` submodule.
- **The teacher, not the student.** The exported weights are the EMA teacher,
  which is what DINO's own linear-probe protocol evaluates and what collapse is
  a property of.

``latest.pt`` is also accepted, since it is written far more often than
``encoder_final.pt`` and is the only artefact available if a run is interrupted.
Its ``teacher`` entry holds the whole ``DINOModel``, so ``backbone.`` is
stripped and the projection head discarded — the head is a pretraining artefact
and feature extraction never runs it.

Run from the repository root:

    python apply_dinov2_adapted_patch.py
    PYTHONPATH=. python -m pytest tests/ -q
"""

from __future__ import annotations

import pathlib

PATH = pathlib.Path("models/encoders/dinov2_encoder.py")


def patch(old: str, new: str, *, description: str, marker: str) -> None:
    text = PATH.read_text()
    if marker in text:
        print(f"  SKIP  {description}")
        return
    if old not in text:
        raise SystemExit(
            f"FAILED: anchor for '{description}' not found in {PATH}.\n"
            f"Sought:\n{old[:400]}"
        )
    PATH.write_text(text.replace(old, new, 1))
    print(f"  OK    {description}")


# --------------------------------------------------------------------------
# 1. The loader
# --------------------------------------------------------------------------

patch(
    old="class DINOv2ViTEncoder(BaseEncoder):",
    new='''def load_adapted_checkpoint(
    path: str | Path,
    *,
    base_checkpoint: str | None = None,
    fallback_base: str | None = None,
):
    """Rebuild a ``Dinov2Model`` carrying weights from continued pretraining.

    ``train/pretrain_dino.py`` exports ``encoder_final.pt`` as a bare state dict
    of the **teacher** backbone, wrapped alongside the training config. That is
    not a HuggingFace directory, so ``from_pretrained`` cannot read it and the
    adapted arm has no route into ``extract_features.py``.

    The teacher rather than the student is exported deliberately: it is the EMA
    of the student, it is what DINO's own linear-probe protocol evaluates, and
    collapse is a property of the teacher.

    Unlike the VideoMAE counterpart there is no bias repair, because DINOv2
    stores query, key and value biases under the names transformers expects.
    Unlike the V-JEPA counterpart the exported weights are the whole backbone
    rather than a submodule, so no prefix needs stripping from
    ``encoder_final.pt`` -- though ``latest.pt`` does carry one.

    Returns ``(model, base_checkpoint, record)``, the record going into the
    cache manifest so that an adapted cache is distinguishable from a baseline
    one after the fact.
    """
    require_transformers()
    from transformers import Dinov2Model

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No adapted checkpoint at {path}.")

    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:  # noqa: BLE001 - checkpoints carry non-tensor config
        payload = torch.load(path, map_location="cpu", weights_only=False)

    if not isinstance(payload, dict):
        raise ValueError(
            f"{path} does not look like a checkpoint written by "
            f"pretrain_dino.py: expected a dict, got {type(payload).__name__}."
        )

    # encoder_final.pt carries the teacher backbone under "model".
    # latest.pt carries the whole DINOModel under "teacher", whose keys are
    # prefixed "backbone." and "head.".
    if "model" in payload:
        state = dict(payload["model"])
        source = "encoder_final"
    elif "teacher" in payload:
        state = dict(payload["teacher"])
        source = "latest"
    else:
        raise ValueError(
            f"{path} has neither a 'model' nor a 'teacher' key; found "
            f"{sorted(payload)[:8]}. It was not written by pretrain_dino.py."
        )

    if any(key.startswith("backbone.") for key in state):
        # Keep the backbone, drop the projection head: the head is a
        # pretraining artefact and extraction never runs it.
        state = {
            key[len("backbone.") :]: value
            for key, value in state.items()
            if key.startswith("backbone.")
        }

    recorded_base = (payload.get("config") or {}).get("model", {}).get("checkpoint")
    base = base_checkpoint or recorded_base or fallback_base
    if base is None:
        raise ValueError(
            f"{path} records no base checkpoint under config.model.checkpoint, "
            f"and none was supplied. Pass model_name explicitly so the "
            f"architecture can be reconstructed."
        )
    if base_checkpoint and recorded_base and base_checkpoint != recorded_base:
        raise ValueError(
            f"{path} was trained from {recorded_base!r} but {base_checkpoint!r} "
            f"was requested. Loading adapted weights into a different "
            f"architecture would either fail loudly or, worse, partially "
            f"succeed."
        )

    model = Dinov2Model.from_pretrained(base)
    result = model.load_state_dict(state, strict=False)

    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)
    if missing:
        raise ValueError(
            f"{len(missing)} parameters in the {base} architecture were not "
            f"present in {path.name}: {missing[:5]}. The adapted encoder would "
            f"carry a mixture of trained and pretrained weights, which is not a "
            f"state any arm of the comparison is supposed to be in."
        )

    record = {
        "path": str(path),
        "base_checkpoint": base,
        "base_recorded_in_checkpoint": recorded_base,
        "source_key": source,
        "step": payload.get("step"),
        "num_loaded": len(state),
        "num_ignored": len(unexpected),
        # The teacher is the EMA of the student. Recorded because a reader of
        # the manifest cannot otherwise tell which of the two was evaluated.
        "exported": "teacher_backbone",
    }
    return model, base, record


class DINOv2ViTEncoder(BaseEncoder):''',
    description="load_adapted_checkpoint",
    marker="def load_adapted_checkpoint",
)


# --------------------------------------------------------------------------
# 2. Constructor argument
# --------------------------------------------------------------------------

patch(
    old="""        random_init: bool = False,
        freeze: bool = True,
    ) -> None:
        super().__init__(freeze=freeze)""",
    new="""        random_init: bool = False,
        adapted_checkpoint: str | Path | None = None,
        freeze: bool = True,
    ) -> None:
        super().__init__(freeze=freeze)

        self._adaptation: dict[str, Any] | None = None
        if adapted_checkpoint is not None:
            if model is not None or random_init:
                raise ValueError(
                    "adapted_checkpoint cannot be combined with model= or "
                    "random_init=True."
                )
            model, base, self._adaptation = load_adapted_checkpoint(
                adapted_checkpoint,
                base_checkpoint=model_name,
                fallback_base=self.CHECKPOINTS.get(variant),
            )
            model_name = f"{base}+adapted:{Path(adapted_checkpoint).name}\"""",
    description="adapted_checkpoint constructor argument",
    marker="adapted_checkpoint: str | Path | None = None",
)


# --------------------------------------------------------------------------
# 3. Record it in describe(), so the manifest distinguishes the arms
# --------------------------------------------------------------------------

_text = PATH.read_text()
if "adaptation" not in _text.split("def describe")[-1][:600]:
    if "def describe" in _text:
        patch(
            old="    def describe(self) -> dict:\n        record = super().describe()",
            new="""    def describe(self) -> dict:
        record = super().describe()
        if getattr(self, "_adaptation", None) is not None:
            record["adaptation"] = self._adaptation""",
            description="describe() records the adaptation",
            marker='record["adaptation"] = self._adaptation',
        )
    else:
        print("  WARN  no describe() found; add the adaptation record by hand")
else:
    print("  SKIP  describe() already records the adaptation")


# --------------------------------------------------------------------------
# 4. Imports the loader needs
# --------------------------------------------------------------------------

_text = PATH.read_text()
_needed = []
if "from pathlib import Path" not in _text:
    _needed.append("from pathlib import Path")
if "import torch" not in _text.split("class ")[0]:
    _needed.append("import torch")
if "from typing import" not in _text or "Any" not in _text.split("class ")[0]:
    _needed.append("from typing import Any")

if _needed:
    lines = _text.splitlines()
    # after the __future__ import, or after the docstring
    insert_at = 0
    for i, line in enumerate(lines[:40]):
        if line.startswith("from __future__"):
            insert_at = i + 1
            break
    for offset, statement in enumerate(_needed):
        lines.insert(insert_at + 1 + offset, statement)
    PATH.write_text("\n".join(lines) + "\n")
    print(f"  OK    added imports: {', '.join(_needed)}")
else:
    print("  SKIP  imports already present")


print("\nDone. Now run:")
print("  python -c \"import ast; ast.parse(open('models/encoders/dinov2_encoder.py').read())\"")
print("  PYTHONPATH=. python -m pytest tests/ -q")
