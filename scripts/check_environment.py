#!/usr/bin/env python3
"""Verify that an environment is suitable for the cvs-domain-ssl experiments.

Runs the same checks on the local machine and on Kaya, so that divergence
between them is detected before a job fails at 3am in the queue rather than
after.

Usage:
    python scripts/check_environment.py
    python scripts/check_environment.py --json report.json

Exit codes:
    0  all required checks passed
    1  one or more required checks failed
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as md
import json
import platform
import sys
from dataclasses import dataclass, field
from typing import Any, Callable


PASS, WARN, FAIL, SKIP = "PASS", "WARN", "FAIL", "SKIP"

# Compute capabilities this project must support.
#   sm_70  Tesla V100 (Kaya)
#   sm_89  RTX 4060 Laptop (local)
REQUIRED_ARCHS = {"sm_70": "Tesla V100 (Kaya)", "sm_89": "RTX 4060 (local)"}


@dataclass
class Result:
    name: str
    status: str
    detail: str = ""
    data: dict[str, Any] = field(default_factory=dict)


class Report:
    def __init__(self) -> None:
        self.results: list[Result] = []

    def add(self, result: Result) -> Result:
        self.results.append(result)
        return result

    def check(self, name: str, required: bool = True) -> Callable:
        """Decorator running a check function and capturing exceptions."""

        def decorator(fn: Callable[[], Result]) -> Result:
            try:
                result = fn()
            except Exception as exc:  # noqa: BLE001 - report, do not crash
                result = Result(
                    name=name,
                    status=FAIL if required else WARN,
                    detail=f"{type(exc).__name__}: {exc}",
                )
            return self.add(result)

        return decorator

    @property
    def failed(self) -> list[Result]:
        return [r for r in self.results if r.status == FAIL]

    @property
    def warned(self) -> list[Result]:
        return [r for r in self.results if r.status == WARN]

    def render(self) -> str:
        width = max(len(r.name) for r in self.results) + 2
        lines = []
        for r in self.results:
            lines.append(f"  [{r.status:4}] {r.name:<{width}} {r.detail}")
        return "\n".join(lines)


def _version(pkg: str) -> str:
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return "not installed"


def section(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def build_report() -> Report:
    report = Report()

    # -- interpreter ------------------------------------------------------

    @report.check("python")
    def _() -> Result:
        major, minor = sys.version_info[:2]
        detail = f"{platform.python_version()} ({sys.executable})"
        if (major, minor) >= (3, 13):
            return Result(
                "python",
                WARN,
                detail + "  -- 3.13+; video decoding and V-JEPA 2 / VideoMAE "
                "upstream repos lag. 3.11 recommended.",
            )
        if (major, minor) < (3, 10):
            return Result("python", FAIL, detail + "  -- 3.10 minimum required.")
        return Result("python", PASS, detail)

    # -- torch ------------------------------------------------------------

    try:
        import torch
    except ImportError as exc:
        report.add(
            Result("torch", FAIL, f"not installed ({exc}). Install before requirements.txt.")
        )
        return report

    @report.check("torch")
    def _() -> Result:
        cuda_build = getattr(torch.version, "cuda", None) or "cpu-only build"
        return Result(
            "torch",
            PASS,
            f"{torch.__version__}  (CUDA build: {cuda_build})",
            {"version": torch.__version__, "cuda_build": cuda_build},
        )

    @report.check("torchvision", required=False)
    def _() -> Result:
        import torchvision

        return Result("torchvision", PASS, torchvision.__version__)

    # -- CUDA and architectures ------------------------------------------

    @report.check("cuda.available", required=False)
    def _() -> Result:
        if not torch.cuda.is_available():
            return Result(
                "cuda.available",
                WARN,
                "no CUDA device visible -- fine for running tests, not for training.",
            )
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return Result(
            "cuda.available",
            PASS,
            f"{name}  sm_{cap[0]}{cap[1]}  {total:.1f} GiB",
            {"device": name, "capability": f"sm_{cap[0]}{cap[1]}", "vram_gib": round(total, 1)},
        )

    @report.check("cuda.arch_list")
    def _() -> Result:
        """The check that decides whether one wheel can serve both machines.

        CUDA guarantees binary compatibility across *minor* versions within the
        same major compute capability, and only upward: a device sm_XY can run
        a cubin built for sm_XZ when Z <= Y. So sm_86 kernels run on an sm_89
        Ada card, but nothing in the 7.5 or 8.x families will run on an sm_70
        Volta card. PTX entries (compute_XX) can additionally be JIT-compiled
        upward at load time.
        """
        if not getattr(torch.version, "cuda", None):
            return Result("cuda.arch_list", SKIP, "CPU-only build.")

        entries = torch.cuda.get_arch_list()
        cubins, ptx = [], []
        for entry in entries:
            if entry.startswith("sm_"):
                cubins.append(entry[3:])
            elif entry.startswith("compute_"):
                ptx.append(entry[8:])

        def satisfied(target: str) -> tuple[bool, str]:
            t_major, t_minor = int(target[0]), int(target[1:])
            for code in cubins:
                if int(code[0]) == t_major and int(code[1:]) <= t_minor:
                    return True, f"sm_{code}"
            for code in ptx:
                if (int(code[0]), int(code[1:])) <= (t_major, t_minor):
                    return True, f"compute_{code} (JIT)"
            return False, ""

        available, missing = [], {}
        for arch, who in REQUIRED_ARCHS.items():
            target = arch[3:]
            ok, via = satisfied(target)
            if ok:
                available.append(f"{arch} via {via}")
            else:
                missing[arch] = who

        detail = " ".join(entries)
        if missing:
            note = ", ".join(f"{a} -- {who}" for a, who in missing.items())
            return Result(
                "cuda.arch_list",
                FAIL,
                f"{detail}\n         UNSUPPORTED: {note}.\n"
                f"         This build cannot serve both machines; pin them separately.",
                {"archs": entries, "missing": sorted(missing), "satisfied": available},
            )
        return Result(
            "cuda.arch_list",
            PASS,
            f"{detail}\n         satisfied: {'; '.join(available)}",
            {"archs": entries, "satisfied": available},
        )

    # -- precision --------------------------------------------------------

    @report.check("precision.bf16", required=False)
    def _() -> Result:
        """bf16 must NOT be relied on: absent on V100 (sm_70)."""
        if not torch.cuda.is_available():
            return Result("precision.bf16", SKIP, "no CUDA device.")
        supported = torch.cuda.is_bf16_supported()
        cap = torch.cuda.get_device_capability(0)
        if supported and cap[0] < 8:
            return Result("precision.bf16", WARN, "reported supported but emulated below sm_80.")
        if supported:
            return Result(
                "precision.bf16",
                WARN,
                "available here, ABSENT on Kaya V100. Write fp16 + GradScaler so "
                "code developed locally runs on the cluster.",
            )
        return Result("precision.bf16", PASS, "unavailable, as expected on sm_70. Use fp16.")

    @report.check("precision.fp16_amp")
    def _() -> Result:
        """Smoke-test the actual training path: autocast + GradScaler."""
        if not torch.cuda.is_available():
            return Result("precision.fp16_amp", SKIP, "no CUDA device.")
        layer = torch.nn.Linear(64, 8).cuda()
        opt = torch.optim.AdamW(layer.parameters(), lr=1e-3)
        try:
            scaler = torch.amp.GradScaler("cuda")
            autocast = torch.amp.autocast("cuda", dtype=torch.float16)
        except (AttributeError, TypeError):  # older torch API
            scaler = torch.cuda.amp.GradScaler()
            autocast = torch.cuda.amp.autocast(dtype=torch.float16)
        with autocast:
            loss = layer(torch.randn(4, 64, device="cuda")).square().mean()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        return Result("precision.fp16_amp", PASS, f"autocast + GradScaler OK (scale={scaler.get_scale():.0f})")

    @report.check("attention.sdpa", required=False)
    def _() -> Result:
        """FlashAttention-2 needs sm_80+; SDPA's memory-efficient kernel works on V100."""
        if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            return Result("attention.sdpa", FAIL, "scaled_dot_product_attention missing.")
        if not torch.cuda.is_available():
            return Result("attention.sdpa", SKIP, "present, untested without CUDA.")
        q = torch.randn(1, 4, 128, 64, device="cuda", dtype=torch.float16)
        torch.nn.functional.scaled_dot_product_attention(q, q, q)
        cap = torch.cuda.get_device_capability(0)
        note = "" if cap[0] >= 8 else "  (FlashAttention-2 unavailable below sm_80, as expected)"
        return Result("attention.sdpa", PASS, f"functional{note}")

    # -- model hubs -------------------------------------------------------

    @report.check("transformers", required=False)
    def _() -> Result:
        try:
            import transformers
        except ImportError:
            return Result("transformers", WARN, "not installed; needed for MAE/VideoMAE/V-JEPA 2/DINOv3.")

        wanted = {
            "vit_mae": "MAE (image)",
            "videomae": "VideoMAE (video)",
            "vjepa2": "V-JEPA 2 (video)",
            "dinov2": "DINOv2 (image)",
            "dinov3": "DINOv3 (image)",
        }
        found, absent = [], []
        for mod, label in wanted.items():
            try:
                importlib.import_module(f"transformers.models.{mod}")
                found.append(label)
            except ImportError:
                # DINOv3 ships as dinov3_vit / dinov3_convnext in some releases
                alt = any(
                    _try_import(f"transformers.models.{mod}{suffix}")
                    for suffix in ("_vit", "_convnext")
                )
                (found if alt else absent).append(label)

        detail = f"{transformers.__version__}  available: {', '.join(found) or 'none'}"
        if absent:
            return Result(
                "transformers",
                WARN,
                detail + f"\n         MISSING: {', '.join(absent)} -- upgrade transformers.",
                {"version": transformers.__version__, "missing": absent},
            )
        return Result("transformers", PASS, detail, {"version": transformers.__version__})

    @report.check("huggingface_hub", required=False)
    def _() -> Result:
        import huggingface_hub
        from huggingface_hub import HfFolder

        token = HfFolder.get_token()
        note = "token present" if token else "NO TOKEN -- gated DINOv3 download will fail"
        status = PASS if token else WARN
        return Result("huggingface_hub", status, f"{huggingface_hub.__version__}  ({note})")

    # -- video decoding ---------------------------------------------------

    @report.check("video.decoder", required=False)
    def _() -> Result:
        for name in ("decord", "av"):
            try:
                mod = importlib.import_module(name)
            except ImportError:
                continue
            ver = getattr(mod, "__version__", _version(name))
            return Result("video.decoder", PASS, f"{name} {ver}")
        return Result(
            "video.decoder",
            WARN,
            "neither decord nor av installed; required for SAGES clip extraction.",
        )

    # -- remaining dependencies ------------------------------------------

    for pkg, required in [
        ("numpy", True),
        ("pandas", True),
        ("scikit-learn", True),
        ("pillow", True),
        ("pyyaml", True),
        ("matplotlib", False),
        ("einops", False),
        ("webdataset", False),
        ("timm", False),
        ("tqdm", False),
        ("pytest", True),
    ]:
        _add_package_check(report, pkg, required)

    # -- repository code --------------------------------------------------

    @report.check("repo.base_encoder")
    def _() -> Result:
        """End-to-end check that this environment can run the project's own code."""
        from models.encoders.base_encoder import (
            BaseEncoder,
            EncoderOutput,
            PreprocessSpec,
            TokenLayout,
        )

        class _Probe(BaseEncoder):
            modality = "image"

            def __init__(self) -> None:
                super().__init__(freeze=True)
                self.proj = torch.nn.Linear(3 * 16 * 16, 32)
                self._finalise_init()

            @property
            def preprocess_spec(self):
                return PreprocessSpec(image_size=32, mean=(0.5,) * 3, std=(0.5,) * 3)

            @property
            def token_layout(self):
                return TokenLayout(grid=(2, 2), dim=32)

            def _forward_tokens(self, x):
                b = x.shape[0]
                patches = (
                    x.unfold(2, 16, 16)
                    .unfold(3, 16, 16)
                    .permute(0, 2, 3, 1, 4, 5)
                    .reshape(b, 4, -1)
                )
                return EncoderOutput(tokens=self.proj(patches))

        enc = _Probe()
        out = enc.extract(torch.randn(2, 3, 32, 32))
        assert out.tokens.shape == (2, 4, 32), out.tokens.shape
        assert out.tokens.dtype == torch.float16
        return Result("repo.base_encoder", PASS, "interface imports and extracts correctly")

    return report


def _try_import(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except ImportError:
        return False


def _add_package_check(report: Report, pkg: str, required: bool) -> None:
    @report.check(pkg, required=required)
    def _() -> Result:
        ver = _version(pkg)
        if ver == "not installed":
            return Result(pkg, FAIL if required else WARN, "not installed")
        return Result(pkg, PASS, ver)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=str, default=None, help="write report to this path")
    args = parser.parse_args()

    section("cvs-domain-ssl environment check")
    print(f"  host: {platform.node()}   platform: {platform.platform()}")

    report = build_report()

    section("results")
    print(report.render())

    section("summary")
    counts = {s: sum(1 for r in report.results if r.status == s) for s in (PASS, WARN, FAIL, SKIP)}
    print(f"  {counts[PASS]} passed, {counts[WARN]} warnings, {counts[FAIL]} failed, {counts[SKIP]} skipped")

    if report.failed:
        print("\n  Blocking issues:")
        for r in report.failed:
            print(f"    - {r.name}: {r.detail.splitlines()[0]}")
    if report.warned:
        print("\n  Non-blocking, review before running experiments:")
        for r in report.warned:
            print(f"    - {r.name}: {r.detail.splitlines()[0]}")

    if args.json:
        payload = {
            "host": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "results": [
                {"name": r.name, "status": r.status, "detail": r.detail, "data": r.data}
                for r in report.results
            ],
        }
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\n  Report written to {args.json}")

    print()
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())