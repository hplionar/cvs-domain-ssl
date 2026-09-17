"""Gram term from an eval-mode student pass: DINOv3 rescales RoPE coordinates
at random in training mode, so train-mode student tokens and eval-mode anchor
tokens differ by that draw alone (gram 0.03 at lr 0). Reverts return_patches."""
from pathlib import Path

p = Path("train/pretrain_dino.py")
src = p.read_text()

def rep(old, new):
    global src
    n = src.count(old)
    assert n == 1, f"expected 1 match, found {n}:\n{old[:80]}"
    src = src.replace(old, new)

rep('''    def forward(
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
''',
'''    def forward(self, views: list[torch.Tensor]) -> torch.Tensor:
        """Run a list of views, concatenating outputs in the given order.

        Views of differing resolution cannot be batched into one tensor, so they
        are grouped by size. Order is preserved so the loss can identify which
        rows correspond to which view index.
        """
        sizes = [v.shape[-1] for v in views]
        order = sorted(range(len(views)), key=lambda i: sizes[i])

        outputs: list[torch.Tensor | None] = [None] * len(views)
        i = 0
''')

rep('''            projected = self.head(pooled)
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
''',
'''            projected = self.head(pooled)

            per_view = projected.shape[0] // len(group)
''')

rep('''        out = torch.cat(outputs, dim=0)  # type: ignore[arg-type]
        if return_patches:
            if patches is None:
                raise RuntimeError("return_patches: no global-resolution group was run")
            return out, patches
        return out


def gram_loss(''',
'''        return torch.cat(outputs, dim=0)  # type: ignore[arg-type]


def gram_loss(''')

rep('''                if anchor is not None:
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
''',
'''                student_out = student(views)           # all views
                loss = criterion(student_out, teacher_out,
                                 num_views=len(views), teacher_temp=teacher_temp)
                dino_loss = loss.detach()
                gram = None
                if anchor is not None and state.step >= gram_start:
                    # A second, eval-mode pass of the student over the global
                    # crops. DINOv3's forward rescales its RoPE coordinates at
                    # random in training mode (pos_embed_rescale), so the
                    # train-mode tokens above and the eval-mode anchor differ by
                    # that draw alone -- 0.03 measured with lr 0. Geometry is
                    # compared deterministic against deterministic; the DINO
                    # loss keeps its train-mode pass, augmentation included, so
                    # the recipe matches the unanchored arm. The last n_patch
                    # tokens are the patch grid whatever prefix (CLS, registers)
                    # precedes it.
                    globals_ = torch.cat(views[:2], dim=0)
                    n_patch = (globals_.shape[-1] // int(student.backbone.config.patch_size)) ** 2
                    student.backbone.eval()
                    student_patches = student.backbone(
                        pixel_values=globals_).last_hidden_state[:, -n_patch:]
                    student.backbone.train()
                    with torch.no_grad():
                        anchor_patches = anchor(
                            pixel_values=globals_).last_hidden_state[:, -n_patch:]
                    gram = gram_loss(student_patches, anchor_patches)
                    loss = loss + gram_weight * gram
''')

rep('''encoder; a run is informative only if the DINO loss still falls while the Gram
term is non-zero, and both are logged for that reason.
"""''',
'''encoder; a run is informative only if the DINO loss still falls while the Gram
term is non-zero, and both are logged for that reason. The term is computed
from a second, eval-mode pass of the student: the HF DINOv3 forward rescales
its RoPE coordinates at random in training mode, which every DINOv3 arm in this
project trains with, and which would otherwise put a floor of about 0.03 under
the term regardless of drift.
"""''')

p.write_text(src)
print("patched", p)
