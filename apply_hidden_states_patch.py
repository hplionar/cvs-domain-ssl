#!/usr/bin/env python3
"""Add hidden-state extraction to the encoder interface.

Feature extraction currently caches ``last_hidden_state`` only -- layer 12 of 12
for ViT-B. Layers 1-11 are computed and discarded. This patch lets an encoder
return a selected subset of intermediate layers so that a head can learn which
depths matter, potentially differently per CVS criterion.

Layers are specified as *relative* depths (0.25, 0.5, 0.75, 1.0) rather than
absolute indices, because VideoMAE ViT-B has 12 blocks and V-JEPA 2 ViT-L has
24. Absolute indices would make "layer 6" mean different things in the two arms.

The embedding output (index 0 of the HuggingFace hidden_states tuple) is never
selectable: it precedes every transformer block and is close to a linear
projection of pixels, so including it would let a "shallow layers do not help"
conclusion be driven by something that could not have helped.

Run from the repository root:

    python apply_hidden_states_patch.py
    PYTHONPATH=. python -m pytest tests/ -q
"""

from __future__ import annotations

import pathlib
import sys


def patch(path: str, old: str, new: str, *, description: str) -> None:
    p = pathlib.Path(path)
    text = p.read_text()
    if new.strip()[:60] in text:
        print(f"  SKIP  {description} (already applied)")
        return
    if old not in text:
        raise SystemExit(
            f"FAILED: could not find the anchor for '{description}' in {path}.\n"
            f"The file may have changed. Anchor sought:\n{old[:200]}"
        )
    p.write_text(text.replace(old, new, 1))
    print(f"  OK    {description}")


# --------------------------------------------------------------------------
# 1. EncoderOutput gains a hidden_states field
# --------------------------------------------------------------------------

patch(
    "models/encoders/base_encoder.py",
    old='''    tokens: torch.Tensor
    prefix: torch.Tensor | None = None''',
    new='''    hidden_states:
        Patch tokens from selected intermediate layers, shape
        ``[B, L, N, D]`` where ``L`` is the number of layers requested, or
        ``None`` when they were not requested. Prefix tokens are removed from
        each layer exactly as they are from ``tokens``, so the spatial layout is
        identical across the layer axis.

        Requested because feature extraction otherwise keeps only the final
        layer. If different CVS criteria are resolved at different levels of
        abstraction -- C2 (tissue cleared) plausibly earlier than C1 (counting
        two structures) -- a last-layer probe discards the evidence.
    """

    tokens: torch.Tensor
    prefix: torch.Tensor | None = None
    hidden_states: torch.Tensor | None = None''',
    description="EncoderOutput.hidden_states",
)

# The docstring above was inserted before the closing quotes of the existing
# docstring, so remove the now-duplicated terminator.
_p = pathlib.Path("models/encoders/base_encoder.py")
_t = _p.read_text()
_t = _t.replace(
    '''        as patches; keeping them here, separated, prevents that error.
    """
    hidden_states:''',
    '''        as patches; keeping them here, separated, prevents that error.
    hidden_states:''',
    1,
)
_p.write_text(_t)


# --------------------------------------------------------------------------
# 2. A helper for resolving relative depths to layer indices
# --------------------------------------------------------------------------

patch(
    "models/encoders/base_encoder.py",
    old='''class BaseEncoder(nn.Module, ABC):''',
    new='''def resolve_layer_indices(
    relative_depths: tuple[float, ...],
    num_layers: int,
) -> tuple[int, ...]:
    """Map relative depths in (0, 1] to indices into a hidden_states tuple.

    HuggingFace returns ``num_layers + 1`` tensors, index 0 being the embedding
    output before any block. Depth ``d`` maps to index ``round(d * num_layers)``,
    so 1.0 is the final block and 0.25 of a 12-block model is block 3.

    Index 0 is unreachable by construction: the embedding output precedes every
    transformer block and is close to a linear projection of pixels, so a
    "shallow layers do not help" conclusion should not be able to rest on it.
    """
    if not relative_depths:
        raise ValueError("relative_depths must not be empty.")
    if any(not 0.0 < d <= 1.0 for d in relative_depths):
        raise ValueError(
            f"relative depths must lie in (0, 1], got {relative_depths}. "
            f"Depth 0 is the embedding output, which is deliberately excluded."
        )
    indices = tuple(max(1, round(d * num_layers)) for d in relative_depths)
    if len(set(indices)) != len(indices):
        raise ValueError(
            f"Relative depths {relative_depths} collapse to duplicate layer "
            f"indices {indices} for a {num_layers}-layer model. Request fewer "
            f"depths or space them further apart."
        )
    return indices


class BaseEncoder(nn.Module, ABC):''',
    description="resolve_layer_indices helper",
)


# --------------------------------------------------------------------------
# 3. forward() and extract() thread the flag through
# --------------------------------------------------------------------------

patch(
    "models/encoders/base_encoder.py",
    old='''    def forward(self, x: torch.Tensor) -> EncoderOutput:
        self._validate_input(x)
        out = self._forward_tokens(x)
        self._validate_output(out, batch_size=x.shape[0])
        return out''',
    new='''    def forward(
        self,
        x: torch.Tensor,
        *,
        layer_depths: tuple[float, ...] | None = None,
    ) -> EncoderOutput:
        """Run the encoder.

        ``layer_depths`` requests intermediate layers as relative depths in
        (0, 1]; see ``resolve_layer_indices``. When omitted, only the final
        layer is returned and behaviour is unchanged.
        """
        self._validate_input(x)
        out = self._forward_tokens(x, layer_depths=layer_depths)
        self._validate_output(out, batch_size=x.shape[0])
        if layer_depths is not None and out.hidden_states is None:
            raise RuntimeError(
                f"{type(self).__name__} ignored layer_depths and returned no "
                f"hidden states. Silently falling back to the final layer would "
                f"make a depth comparison meaningless."
            )
        return out''',
    description="BaseEncoder.forward(layer_depths=...)",
)

patch(
    "models/encoders/base_encoder.py",
    old='''    def extract(self, x: torch.Tensor, *, to_dtype: torch.dtype | None = torch.float16) -> EncoderOutput:''',
    new='''    def extract(
        self,
        x: torch.Tensor,
        *,
        to_dtype: torch.dtype | None = torch.float16,
        layer_depths: tuple[float, ...] | None = None,
    ) -> EncoderOutput:''',
    description="BaseEncoder.extract(layer_depths=...)",
)

patch(
    "models/encoders/base_encoder.py",
    old='''        try:
            out = self.forward(x)
        finally:''',
    new='''        try:
            out = self.forward(x, layer_depths=layer_depths)
        finally:''',
    description="extract() passes layer_depths",
)

patch(
    "models/encoders/base_encoder.py",
    old='''        return EncoderOutput(
            tokens=out.tokens.to(to_dtype),
            prefix=None if out.prefix is None else out.prefix.to(to_dtype),
        )''',
    new='''        return EncoderOutput(
            tokens=out.tokens.to(to_dtype),
            prefix=None if out.prefix is None else out.prefix.to(to_dtype),
            hidden_states=(
                None if out.hidden_states is None else out.hidden_states.to(to_dtype)
            ),
        )''',
    description="extract() casts hidden states",
)

patch(
    "models/encoders/base_encoder.py",
    old='''    def _forward_tokens(self, x: torch.Tensor) -> EncoderOutput:
        """Run the underlying model and return patch tokens with prefix separated.''',
    new='''    def _forward_tokens(
        self,
        x: torch.Tensor,
        *,
        layer_depths: tuple[float, ...] | None = None,
    ) -> EncoderOutput:
        """Run the underlying model and return patch tokens with prefix separated.

        Implementations that support ``layer_depths`` must also strip prefix
        tokens from each returned layer, so that the spatial layout is identical
        along the layer axis.''',
    description="_forward_tokens signature",
)


# --------------------------------------------------------------------------
# 4. VideoMAE implements it
# --------------------------------------------------------------------------

patch(
    "models/encoders/videomae_encoder.py",
    old='''    def _forward_tokens(self, x: torch.Tensor) -> EncoderOutput:
        out = self.model(pixel_values=x)
        return EncoderOutput(tokens=out.last_hidden_state, prefix=None)''',
    new='''    def _forward_tokens(
        self,
        x: torch.Tensor,
        *,
        layer_depths: tuple[float, ...] | None = None,
    ) -> EncoderOutput:
        if layer_depths is None:
            out = self.model(pixel_values=x)
            return EncoderOutput(tokens=out.last_hidden_state, prefix=None)

        from models.encoders.base_encoder import resolve_layer_indices

        out = self.model(pixel_values=x, output_hidden_states=True)
        # VideoMAE emits no prefix token, so every hidden state is already pure
        # patch tokens and needs no slicing. Encoders with a CLS or register
        # tokens must strip them here.
        indices = resolve_layer_indices(
            layer_depths, self.model.config.num_hidden_layers
        )
        selected = torch.stack([out.hidden_states[i] for i in indices], dim=1)
        return EncoderOutput(
            tokens=out.last_hidden_state,
            prefix=None,
            hidden_states=selected,
        )''',
    description="VideoMAEEncoder._forward_tokens",
)


# --------------------------------------------------------------------------
# 5. Every other encoder needs the signature, even if unsupported
# --------------------------------------------------------------------------

for module, cls in [
    ("models/encoders/vjepa2_encoder.py", "VJEPA2Encoder"),
    ("models/encoders/mae_encoder.py", "ViTMAEEncoder"),
    ("models/encoders/dinov3_encoder.py", "DINOv3ViTEncoder"),
]:
    p = pathlib.Path(module)
    if not p.exists():
        print(f"  SKIP  {module} (not present)")
        continue
    text = p.read_text()
    if "layer_depths" in text:
        print(f"  SKIP  {module} (already applied)")
        continue
    old = "    def _forward_tokens(self, x: torch.Tensor) -> EncoderOutput:"
    if old not in text:
        print(f"  WARN  {module}: no _forward_tokens signature matched; "
              f"update by hand")
        continue
    new = (
        "    def _forward_tokens(\n"
        "        self,\n"
        "        x: torch.Tensor,\n"
        "        *,\n"
        "        layer_depths: tuple[float, ...] | None = None,\n"
        "    ) -> EncoderOutput:\n"
        "        if layer_depths is not None:\n"
        "            raise NotImplementedError(\n"
        f"                \"{cls} does not yet support layer_depths. \"\n"
        "                \"Returning the final layer silently would make a depth \"\n"
        "                \"comparison meaningless.\"\n"
        "            )"
    )
    p.write_text(text.replace(old, new, 1))
    print(f"  OK    {module}")


print("\nDone. Now run:")
print("  python -c \"import ast; ast.parse(open('models/encoders/base_encoder.py').read())\"")
print("  PYTHONPATH=. python -m pytest tests/ -q")
