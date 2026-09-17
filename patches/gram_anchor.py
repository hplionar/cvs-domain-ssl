"""Add Gram anchoring (DINOv3, Siméoni et al. 2025) to train/pretrain_dino.py.

Off unless train.gram_weight > 0; the flag-off path is unchanged.
"""
from pathlib import Path

p = Path("train/pretrain_dino.py")
src = p.read_text()

def rep(old, new):
    global src
    n = src.count(old)
    assert n == 1, f"expected 1 match, found {n}:\n{old[:80]}"
    src = src.replace(old, new)

# ---- module docstring ---------------------------------------------------
rep('''`data.colour_jitter: true` to restore the reference recipe as its own arm.
"""''',
'''`data.colour_jitter: true` to restore the reference recipe as its own arm.

Gram anchoring
--------------
The loss above reads the CLS token only; the 196 patch tokens receive no
gradient of their own and move only through shared weights. `train.gram_weight`
adds DINOv3's Gram term: the Gram matrix of l2-normalised patch tokens on the
global crops is pulled toward that of a frozen copy of the *initial* backbone
(the anchor -- not the EMA teacher, which drifts with the student). The term is
zero at step 0 and grows with drift, so it is a leash whose tension the weight
sets. Too loose reproduces the unanchored arm; too tight reproduces the base
encoder; a run is informative only if the DINO loss still falls while the Gram
term is non-zero, and both are logged for that reason.
"""''')

# ---- DINOModel.forward: optionally return global patch tokens ----------
rep('''    def forward(self, views: list[torch.Tensor]) -> torch.Tensor:
        """Run a list of views, concatenating outputs in the given order.

        Views of differing resolution cannot be batched into one tensor, so they
        are grouped by size. Order is preserved so the loss can identify which
        rows correspond to which view index.
        """
        sizes = [v.shape[-1] for v in views]
        order = sorted(range(len(views)), key=lambda i: sizes[i])

        outputs: list[torch.Tensor | None] = [None] * len(views)
        i = 0
''',
'''    def forward(
        self, views: list[torch.Tensor], *, return_patches: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run a list of views, concatenating outputs in the given order.

        Views of differing resolution cannot be batched into one tensor, so they
        are grouped by size. Order is preserved so the loss can identify which
        rows correspond to which view index.

        With ``return_patches`` the patch tokens of the largest-resolution group
        (the global views) are also returned, shape ``[n_global * B, P, D]`` in
        view order, for the Gram anchor. They come out of the same forward pass
        as the pooled output, so requesting them costs no extra compute.
        """
        sizes = [v.shape[-1] for v in views]
        order = sorted(range(len(views)), key=lambda i: sizes[i])
        global_size = max(sizes)

        outputs: list[torch.Tensor | None] = [None] * len(views)
        patches: torch.Tensor | None = None
        i = 0
''')

rep('''            projected = self.head(pooled)

            per_view = projected.shape[0] // len(group)
''',
'''            projected = self.head(pooled)
            if return_patches and size == global_size:
                # The last n_patch tokens are the patch grid whatever prefix
                # (CLS, registers) the architecture puts before it. Checked
                # against the grid size rather than assumed.
                patch = int(self.backbone.config.patch_size)
                n_patch = (size // patch) ** 2
                if hidden.shape[1] <= n_patch:
                    raise RuntimeError(
                        f"{hidden.shape[1]} tokens for a {size}px input at patch "
                        f"{patch}: expected {n_patch} patches plus a CLS token."
                    )
                patches = hidden[:, -n_patch:]

            per_view = projected.shape[0] // len(group)
''')

rep('''        return torch.cat(outputs, dim=0)  # type: ignore[arg-type]


class DINOLoss(nn.Module):''',
'''        out = torch.cat(outputs, dim=0)  # type: ignore[arg-type]
        if return_patches:
            if patches is None:
                raise RuntimeError("return_patches: no global-resolution group was run")
            return out, patches
        return out


def gram_loss(student_patches: torch.Tensor, anchor_patches: torch.Tensor) -> torch.Tensor:
    """Squared Frobenius distance between student and anchor patch Gram matrices.

    Tokens are l2-normalised, so each Gram entry is the cosine between two
    patches of the same image and the matrix is the image's internal geometry,
    invariant to any global rotation of feature space. The P x P difference is
    summed and divided by P -- a per-token rather than per-pair scale -- then
    averaged over the batch. DINOv3 uses the raw Frobenius sum with weight 1,
    which on 196 tokens equals this quantity at weight 196.

    Float32 regardless of autocast: the squared differences are small and fp16
    would floor them.
    """
    s = F.normalize(student_patches.float(), dim=-1)
    a = F.normalize(anchor_patches.float(), dim=-1)
    gram_s = torch.bmm(s, s.transpose(1, 2))
    gram_a = torch.bmm(a, a.transpose(1, 2))
    return (gram_s - gram_a).pow(2).sum(dim=(1, 2)).div(s.shape[1]).mean()


class DINOLoss(nn.Module):''')

# ---- build_models: frozen anchor -----------------------------------------
rep('''    for param in teacher.parameters():
        param.requires_grad = False

    freeze_last = int(config["model"].get("freeze_last_blocks", 0))
''',
'''    for param in teacher.parameters():
        param.requires_grad = False

    # Gram anchor: the pretrained backbone, frozen, never updated. Built here,
    # before any checkpoint resume, so it is the pristine weights on every
    # run; it is deterministic from the checkpoint name and is not saved.
    anchor = None
    if float(config.get("train", {}).get("gram_weight", 0.0)) > 0:
        anchor = copy.deepcopy(student.backbone)
        for param in anchor.parameters():
            param.requires_grad = False
        anchor.eval()
        anchor = anchor.to(device)

    freeze_last = int(config["model"].get("freeze_last_blocks", 0))
''')

rep('''    return student.to(device), teacher.to(device), dim, out_dim''',
    '''    return student.to(device), teacher.to(device), anchor, dim, out_dim''')

# ---- main ----------------------------------------------------------------
rep('''    student, teacher, dim, out_dim = build_models(config, device)''',
    '''    student, teacher, anchor, dim, out_dim = build_models(config, device)''')

rep('''    base_lr = float(train_cfg["lr"])
''',
'''    base_lr = float(train_cfg["lr"])
    gram_weight = float(train_cfg.get("gram_weight", 0.0))
    gram_start = int(train_cfg.get("gram_start_step", 0))
''')

rep('''    print(f"centre mom.   {train_cfg.get('centre_momentum', 0.9)}")
''',
'''    print(f"centre mom.   {train_cfg.get('centre_momentum', 0.9)}")
    if anchor is not None:
        print(f"gram anchor   weight {gram_weight} from step {gram_start}, "
              f"frozen copy of the checkpoint")
''')

rep('''        running, seen = 0.0, 0

        for views in loader:''',
'''        running, running_dino, running_gram, seen = 0.0, 0.0, 0.0, 0

        for views in loader:''')

rep('''            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                with torch.no_grad():
                    teacher_out = teacher(views[:2])   # global views only
                student_out = student(views)           # all views
                loss = criterion(student_out, teacher_out,
                                 num_views=len(views), teacher_temp=teacher_temp)
''',
'''            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                with torch.no_grad():
                    teacher_out = teacher(views[:2])   # global views only
                if anchor is not None:
                    student_out, student_patches = student(views, return_patches=True)
                else:
                    student_out = student(views)       # all views
                loss = criterion(student_out, teacher_out,
                                 num_views=len(views), teacher_temp=teacher_temp)
                dino_loss = loss.detach()
                gram = None
                if anchor is not None and state.step >= gram_start:
                    with torch.no_grad():
                        anchor_patches = anchor(
                            pixel_values=torch.cat(views[:2], dim=0)
                        ).last_hidden_state[:, -student_patches.shape[1]:]
                    gram = gram_loss(student_patches, anchor_patches)
                    loss = loss + gram_weight * gram
''')

rep('''            running += loss.item()
            seen += 1
''',
'''            running += loss.item()
            running_dino += dino_loss.item()
            if gram is not None:
                running_gram += gram.item()
            seen += 1
''')

rep('''                row = {
                    "step": state.step, "epoch": state.epoch,
                    "loss": round(mean_loss, 5), "lr": lr, "wd": round(wd, 5),
''',
'''                row = {
                    "step": state.step, "epoch": state.epoch,
                    "loss": round(mean_loss, 5), "lr": lr, "wd": round(wd, 5),
                    "dino": round(running_dino / max(seen, 1), 5),
                    "gram": round(running_gram / max(seen, 1), 5),
''')

rep('''                print(
                    f"step {state.step:7d}/{total_steps}  loss {mean_loss:.4f}  "
                    f"lr {lr:.2e}  t_temp {teacher_temp:.4f}  ema {momentum:.5f}  "
''',
'''                gram_text = (f"gram {running_gram / max(seen, 1):.4f}  "
                             if anchor is not None else "")
                print(
                    f"step {state.step:7d}/{total_steps}  loss {mean_loss:.4f}  "
                    + gram_text +
                    f"lr {lr:.2e}  t_temp {teacher_temp:.4f}  ema {momentum:.5f}  "
''')

rep('''                state.history.append(row)
                running, seen = 0.0, 0
''',
'''                state.history.append(row)
                running, running_dino, running_gram, seen = 0.0, 0.0, 0.0, 0
''')

p.write_text(src)
print("patched", p)
