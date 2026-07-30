# Environment

## 1. Recovered Kaya Configuration

Reconstructed from `scripts/*.sbatch` and `scripts/backup_research_artifacts.sh`.

| Item | Value |
|---|---|
| Conda environment | `/group/pmc085/hlionar/conda_envs/vmamba` (prefix env) |
| Slurm account | `pmc085` |
| Partition | `gpu`, `--gres=gpu:1` |
| Typical resources | `--cpus-per-task=8`, `--mem=48G`, `--time=04:00:00` |
| Repository | `/group/pmc085/hlionar/cvs-domain-ssl` |
| Outputs | `/group/pmc085/hlionar/outputs/cvs-domain-ssl` |
| Logs | `/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.{out,err}` |
| SAGES dataset | `/group/pmc085/hlionar/datasets/SAGES_CVS_Challenge_2024` |
| V-JEPA upstream | `/group/pmc085/hlionar/external/jepa` |
| Backups | `/group/pmc085/hlionar/backups/cvs-domain-ssl` |
| Environment modules | none — no `module load` appears in any script |

Every job script prints `torch.__version__`, CUDA availability, and the device
name at start-up. **The exact versions used for exp001–exp006 are therefore
recorded in the `.out` logs**, which the backup script collects. If a backup
archive was copied off Kaya before the shutdown, the versions can be recovered
now rather than after access resumes:

```bash
tar -xzf cvs-domain-ssl-backup-<STAMP>.tar.gz
grep -h "Torch:\|CUDA device name:" */outputs/**/*.out | sort -u
```

## 2. Environment Naming and Separation

`vmamba` was built for the VMamba arm of the SwinCVS reproduction and has since
been reused for every job in this project. The name is misleading and the
contents are wrong for what follows.

**Do not rename it in place.** `conda rename` is implemented as clone-then-delete,
and cloning an environment containing compiled CUDA extensions — which VMamba's
`selective_scan` kernels are — frequently produces a broken copy, because
pip-installed console scripts and some build artefacts embed absolute paths.
Renaming would also break every sbatch script in the `swincvs-reproduction`
repository, which still references the old prefix.

Create a new environment instead and leave the old one untouched:

| Environment | Purpose | Status |
|---|---|---|
| `vmamba` | SwinCVS reproduction and verification | Frozen — do not modify or delete |
| `cvsssl` | This project | To be created |

Before creating anything, capture what `vmamba` contains. It is the only
existing evidence of a torch/CUDA combination confirmed to work on Kaya's
V100s, and that record is worth having:

```bash
conda activate /group/pmc085/hlionar/conda_envs/vmamba
conda env export > docs/env_vmamba_snapshot.yml
pip freeze > docs/env_vmamba_pip.txt
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_arch_list())"
```

Commit both files. If the recorded build predates the Volta removal described in
§4, it confirms the version to pin for `cvsssl` on Kaya.

Once `cvsssl` is validated, update the `conda activate` line in every script:

```bash
sed -i 's|conda_envs/vmamba|conda_envs/cvsssl|' scripts/*.sbatch
```

Review the diff before committing — the SwinCVS scripts live in a different
repository and must keep pointing at `vmamba`.

## 3. Python Version

Use **Python 3.11**, not 3.13. Video decoding and the upstream V-JEPA 2 and
VideoMAE repositories lag behind the newest interpreter releases, and build
failures there would surface exactly when experiments need to run.

## 4. PyTorch Installation

PyTorch is not listed in `requirements.txt` because the correct build depends on
the target machine.

| Machine | GPU | Architecture | Compute capability |
|---|---|---|---|
| Local laptop | RTX 4060 Laptop | Ada | `sm_89` |
| Kaya | Tesla V100 | Volta | `sm_70` |

### Finding: use the cu126 build on both machines

PyTorch removed Volta from its CUDA 12.8 builds; the minimum architecture for
CUDA 12.8+ is Turing, because retaining Volta blocked a cuDNN upgrade after
cuDNN itself dropped Volta. CUDA Toolkit 13.0 removed offline compilation and
library support for Volta entirely. Upstream guidance directs sm_70 users to
the CUDA 12.6 wheel.

Verified against a `2.13.0+cu130` build, whose architecture list is:

```
sm_75 sm_80 sm_86 sm_90 sm_100 sm_120
```

No `sm_70`. That build cannot run on Kaya at all.

**The cu126 build serves both machines.** It ships `sm_70` for the V100, and an
`sm_89` Ada card runs `sm_86` kernels under CUDA minor-version binary
compatibility. A single pinned version therefore covers local development and
the cluster, which is what the reproducibility argument requires.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

Then confirm:

```bash
python scripts/check_environment.py
```

The `cuda.arch_list` check must report `PASS`. It applies the compatibility rule
rather than string matching: a device `sm_XY` can run a cubin built for `sm_XZ`
only when `Z <= Y` **and** the major version matches. Nothing in the 7.5 or 8.x
families will run on Volta, so `sm_70` must be present literally.

### Consequences for the code

- `bf16` is unavailable on sm_70 — Volta supports only fp16 and fp32. Use
  `fp16` autocast with `GradScaler` throughout. Code developed on the RTX 4060
  will run `bf16` successfully and then fail on Kaya.
- FlashAttention-2 requires sm_80 and will not build. Use PyTorch's
  `scaled_dot_product_attention`, which selects a memory-efficient kernel
  supported on Volta.
- Any third-party package compiling CUDA extensions must also be built against
  cu126. This includes VMamba's `selective_scan` kernels, which is one reason
  the `vmamba` environment must be left alone rather than upgraded in place.

For local test runs only, where no GPU is needed, the CPU wheel is sufficient
and much smaller:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## 5. Local Setup

```bash
conda create -n cvsssl python=3.11 -y
conda activate cvsssl

# 1. PyTorch first, with the appropriate index URL (see above)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 2. Remaining dependencies
pip install -r requirements.txt

# 3. Verify the environment, including that sm_70 is present for Kaya
PYTHONPATH=. python scripts/check_environment.py

# 4. Verify the repository code
PYTHONPATH=. python -m pytest tests/ -q
```

Expected result: `check_environment.py` exits 0 with no FAIL rows, and 31
passing tests. The test suite runs entirely on synthetic tensors and does not
require a GPU.

In VS Code, select the interpreter with `Ctrl+Shift+P` → *Python: Select
Interpreter* → `cvsssl`, so that test discovery targets the correct environment.

## 6. Kaya Setup

To be performed when access resumes (~4 August 2026). Create the environment on
group storage rather than in `$HOME`, following the existing convention, since
home quotas are small:

```bash
conda create -p /group/pmc085/hlionar/conda_envs/cvsssl python=3.11 -y
conda activate /group/pmc085/hlionar/conda_envs/cvsssl
```

Then install as above. Job scripts must be updated from:

```bash
conda activate /group/pmc085/hlionar/conda_envs/vmamba
```

to:

```bash
conda activate /group/pmc085/hlionar/conda_envs/cvsssl
```

Note that V100s are `sm_70` and therefore support neither `bf16` nor
FlashAttention-2. Use `fp16` autocast with `GradScaler`, and PyTorch's built-in
`scaled_dot_product_attention` for the memory-efficient kernel.

## 7. Locking

Once an environment is validated on both machines, freeze it:

```bash
pip freeze > requirements.lock.txt
```

Commit the lock file. `requirements.txt` carries floors for readability;
`requirements.lock.txt` is the reproducible record cited in the dissertation. If
the architecture check in §4 forces divergent builds, produce
`requirements.lock.local.txt` and `requirements.lock.kaya.txt` instead, and
record the reason here.

## 8. Measured Performance

Recorded as measured, since `winter_break_summary.md` §7.3 requires GPU hours
and peak memory per run, and these figures cannot be reconstructed later without
rebuilding the environment.

### Feature extraction, VideoMAE ViT-B

16 frames at 224 px, fp16 autocast, `torch.inference_mode`, synthetic inputs
(`--smoke --random-init`), RTX 4060 Laptop, 8 GiB.

| batch | steady state (clips/s) | peak allocated (GiB) | peak reserved (GiB) |
|---:|---:|---:|---:|
| 8 | 43.6 | 0.74 | 0.92 |
| 16 | 41.8 | 1.16 | 1.46 |
| 32 | 39.4 | 1.98 | 2.53 |
| 64 | 39.3 | 3.63 | 4.68 |
| 128 | 10.0 | 6.93 | 8.99 |

Peak allocation is exactly linear in batch size across all five points:

```
peak_alloc(GiB) = 0.33 + 0.0516 * batch_size
```

that is, **52.8 MiB per clip marginal, 0.33 GiB fixed** (model weights and CUDA
context).

### Interpretation

**Throughput saturates at batch size 8.** Rates are flat from 8 to 64 and
marginally decline. Larger batches buy nothing on this hardware; batch size is
not a tuning target for extraction.

**Batch 128 did not raise OOM, it spilled.** Reserved memory of 8.99 GiB exceeds
the device's 8.0 GiB. Under WSL2 the driver falls back to host memory over PCIe
instead of failing, producing a four-fold slowdown rather than an exception.
Consequence for method: on this machine, capacity must be checked by watching
reported peak memory, not by increasing batch size until something crashes. On
Kaya, which is native Linux, the same overrun fails outright.

**Extraction is not a bottleneck.** SAGES train is 10,080 clips, about 4.2
minutes of GPU time at 40 clips/s; both video arms across three splits is
roughly half an hour. The real cost is decoding 16 frames per clip from MP4, so
optimisation effort belongs in clip pre-extraction and `--num-workers`, not in
batch size.

### Recommended settings

| Machine | `--batch-size` | Rationale |
|---|---:|---|
| RTX 4060 (8 GiB) | 32 | Saturated well below this; leaves headroom |
| Kaya V100 (32 GiB) | 32 | Memory ceiling is ~536, but throughput saturates far earlier |

These figures are for inference. They do not transfer to SSL pretraining, where
gradients, optimiser state and the EMA target encoder dominate.