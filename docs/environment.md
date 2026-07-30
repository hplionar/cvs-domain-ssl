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

## 2. Environment Separation

`vmamba` was built for the VMamba arm of the SwinCVS reproduction and has since
been reused for every job in this project.

**It must not be extended further.** VMamba depends on compiled CUDA extensions
built against a specific torch version. Installing V-JEPA 2, VideoMAE, or
DINOv3 dependencies into the same environment risks breaking those kernels, and
VMamba's own pin may block a torch version the newer checkpoints require. The
two are separate deliverables and take separate environments:

| Environment | Purpose | Status |
|---|---|---|
| `vmamba` | SwinCVS reproduction and verification | Frozen — do not modify |
| `cvsssl` | This project | To be created |

## 3. Python Version

Use **Python 3.11**, not 3.13. Video decoding and the upstream V-JEPA 2 and
VideoMAE repositories lag behind the newest interpreter releases, and build
failures there would surface exactly when experiments need to run.

## 4. PyTorch Installation

PyTorch is not listed in `requirements.txt` because the correct build depends on
the target machine. Install it first, then apply the requirements file.

| Machine | GPU | Architecture | Compute capability |
|---|---|---|---|
| Local laptop | RTX 4060 Laptop | Ada | `sm_89` |
| Kaya | Tesla V100 | Volta | `sm_70` |

A single wheel can serve both provided it ships kernels for both
architectures. **Verify this rather than assuming it**, because recent PyTorch
releases have progressively dropped older compute capabilities:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.get_arch_list())"
```

Both `sm_70` and `sm_89` must appear in the output. If `sm_70` is absent, the
Kaya build must be pinned to an earlier release, and the two machines will
require separate lock files.

Install command (adjust the CUDA suffix to match the driver reported by
`nvidia-smi`):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

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
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 2. Confirm the GPU is visible and the architecture is supported
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_arch_list())"

# 3. Remaining dependencies
pip install -r requirements.txt

# 4. Verify
PYTHONPATH=. python -m pytest tests/ -q
```

Expected result: 31 passing tests. The suite runs entirely on synthetic tensors
and does not require a GPU.

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