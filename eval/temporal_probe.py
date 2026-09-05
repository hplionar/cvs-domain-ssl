#!/usr/bin/env python3
"""Does temporal context help, when the aggregator can use order?

The project's evidence on temporal context is currently weak in two ways. The
comparison between image and video encoders varies architecture, capacity, input
resolution and token count alongside the input level, and on the official test
split the two groups are not even separated. And the one direct test -- mean
pooling DINOv2 features over a five-frame sparse clip -- lost 0.09 mAP, but mean
pooling cannot distinguish "the criterion was satisfied five seconds ago" from
"it is satisfied now". It averages.

This script supplies the missing control: the same frozen features, the same
frames, the same labels, with only the temporal window varying, and an
aggregator that can use order.

**Windows are trailing, not centred.** Frames t-k+1 through t, labelled by frame
t. Centred windows would use future frames, which the SAGES challenge forbade at
inference and which makes any resulting figure non-comparable to the
leaderboard. Trailing windows are causal and the numbers stand beside published
ones.

**Aggregation is over mean-pooled frame features, not patch grids.** Eighteen
frames of 196 tokens at 768 dimensions is 2.7 million values per sequence, and
any recurrent model would have to pool spatially in any case. Pooling first
makes k = 1 exactly the existing frozen probe, so the only thing that changes
between k = 1 and k > 1 is the temporal aggregation.

**Why an LSTM.** The surgical temporal-modelling literature -- TeCNO, TUNeS,
LoViT, Trans-SVNet -- targets phase recognition over one- to two-hour videos,
where LSTMs are criticised for failing to capture long-range dependencies and
where elapsed time alone is a powerful feature. Neither applies here: SAGES
clips are ninety seconds, the longest window is eighteen steps, and CVS is a
state assessed at an instant rather than a phase in a sequence. The LSTM is
chosen because SwinCVS pairs a frozen backbone with a five-frame LSTM and
reaches 0.6553 mAP on Endoscapes in this project's own reproduction, making
k = 5 a close comparison against a published system under the same protocol.

**A caveat to state in the write-up.** TUNeS reports that temporal models
perform better on feature extractors trained with long temporal context. These
features were extracted from encoders that never saw a sequence, so a null
result is partly explained by that. The counter is that SwinCVS Frozen is in the
same position -- ImageNet SwinV2, no temporal pretraining -- and its LSTM works,
so the configuration is not disqualified in principle.

Usage:
    python eval/temporal_probe.py \\
        --train-features ../cache/dinov3_b/sages/train \\
        --val-features   ../cache/dinov3_b/sages/val \\
        --test-features  ../cache/dinov3_b/sages_official/test \\
        --manifest       metadata/sages_frames_internal_split.csv \\
        --test-manifest  metadata/sages_frames_official_test.csv \\
        --windows 1 3 5 9 18 \\
        --output-dir ../outputs/temporal_probe/dinov3_b
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from eval.metrics import compute_multilabel_metrics_from_logits
from train.train_probe_cached import CachedFeatures, set_seed

CRITERIA = ("c1", "c2", "c3")


class TrailingWindows(Dataset):
    """Sequences of k consecutive frames ending at the labelled frame.

    Frames earlier than the start of the video are supplied by repeating the
    first available frame rather than by zero padding. A zero vector is not a
    plausible feature and the head would learn to detect the padding instead of
    the content; repeating the earliest frame degrades gracefully to the k = 1
    case at the start of each video, which is the behaviour wanted.
    """

    def __init__(self, features: torch.Tensor, targets: torch.Tensor,
                 order: np.ndarray, window: int) -> None:
        self.features = features
        self.targets = targets
        self.order = order          # [N, k] row indices, trailing, clamped
        self.window = window

    def __len__(self) -> int:
        return self.order.shape[0]

    def __getitem__(self, i: int):
        return self.features[self.order[i]], self.targets[self.order[i, -1]]


def build_order(cache: CachedFeatures, meta: pd.DataFrame, window: int) -> np.ndarray:
    """[N, k] row indices for every labelled frame's trailing window.

    Ordering within a video comes from the manifest's sequence_index rather than
    from parsing sample_id, so the script does not depend on a naming
    convention that could change.
    """
    if "sequence_index" not in meta.columns:
        raise SystemExit("The manifest has no sequence_index column; window order "
                         "cannot be established.")
    indexed = meta.set_index("sample_id")
    ids = [cache.sample_ids[i] for i in cache.rows]
    missing = [s for s in ids if s not in indexed.index]
    if missing:
        raise SystemExit(f"{len(missing)} sample_ids absent from the manifest, "
                         f"e.g. {missing[:3]}")
    rows = indexed.loc[ids]
    seq = rows["sequence_index"].to_numpy(dtype=int)
    vid = rows["video_id"].to_numpy()

    order = np.empty((len(ids), window), dtype=np.int64)
    for v in np.unique(vid):
        mask = np.flatnonzero(vid == v)
        # Position within the video, ascending in time.
        local = mask[np.argsort(seq[mask])]
        for pos, row in enumerate(local):
            # Trailing window, clamped at the start of the video.
            idx = [local[max(0, pos - (window - 1 - j))] for j in range(window)]
            order[row] = idx
    return order


class LSTMHead(nn.Module):
    """LayerNorm, LSTM, linear on the final hidden state.

    The LayerNorm matches the frozen probe's head so that k = 1 is that head
    with a recurrent layer of one step, and the comparison isolates the
    aggregation rather than the normalisation.
    """

    def __init__(self, dim: int, hidden: int, dropout: float, layers: int = 1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.lstm = nn.LSTM(dim, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, 3)

    def forward(self, x):
        out, _ = self.lstm(self.norm(x))
        # The final step is the labelled frame; earlier steps are context.
        return self.fc(self.drop(out[:, -1]))


class TransformerHead(nn.Module):
    """LayerNorm, a transformer encoder layer, linear on the labelled step.

    The recurrent head processes the window in order and reads its final state;
    this one attends over the window and reads the position of the labelled
    frame. The difference that matters is that attention can weight a frame five
    steps back as heavily as the one immediately preceding, where a recurrence
    must pass information through every intervening step.

    Learned positional embeddings are added because the window is ordered and
    self-attention is not. Without them the head would be a set function, which
    would confound "does order matter" with "does context matter" -- and the
    recurrent result already shows context contributes something, so the
    question here is whether the way it is combined is what limits the gain.

    Kept deliberately small: one layer, four heads. Eighteen steps of a
    768-dimensional sequence with 10,080 training windows does not support more,
    and a larger head would confound capacity with architecture.
    """

    def __init__(self, dim: int, hidden: int, dropout: float,
                 heads: int = 4, layers: int = 1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.project = nn.Linear(dim, hidden)
        self.position = nn.Parameter(torch.zeros(1, 64, hidden))
        nn.init.trunc_normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=heads, dim_feedforward=2 * hidden,
            dropout=dropout, batch_first=True, norm_first=True,
            activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, 3)

    def forward(self, x):
        h = self.project(self.norm(x))
        h = h + self.position[:, : h.shape[1]]
        h = self.encoder(h)
        # The last position is the labelled frame, as in the recurrent head, so
        # that the two differ only in how the window is combined.
        return self.fc(self.drop(h[:, -1]))


@torch.no_grad()
def infer(model, loader, device):
    model.eval()
    logits, targets = [], []
    for x, y in loader:
        logits.append(model(x.to(device)).float().cpu().numpy())
        targets.append(y.numpy())
    return np.concatenate(logits), np.concatenate(targets)


HEADS = {"lstm": LSTMHead, "transformer": TransformerHead}


def train_one(config, seed, train_ds, val_ds, dim, epochs, patience, device,
              head_name="lstm"):
    set_seed(seed)
    head = HEADS[head_name](dim, int(config["hidden"]),
                            float(config["dropout"])).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=float(config["lr"]),
                            weight_decay=float(config["weight_decay"]))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=0)

    best, best_state, best_epoch, left = -1.0, None, 0, patience
    for epoch in range(1, epochs + 1):
        head.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(head(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()
        logits, targets = infer(head, val_loader, device)
        current = compute_multilabel_metrics_from_logits(targets, logits)["mAP"]
        if not np.isnan(current) and current > best:
            best, best_epoch, left = float(current), epoch, patience
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        else:
            left -= 1
            if patience > 0 and left <= 0:
                break
    return best, best_epoch, best_state


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-features", required=True)
    p.add_argument("--val-features", required=True)
    p.add_argument("--test-features", default=None)
    p.add_argument("--manifest", required=True,
                   help="manifest covering the train and val caches")
    p.add_argument("--test-manifest", default=None,
                   help="manifest covering the test cache, if it differs")
    p.add_argument("--windows", type=int, nargs="+", default=[1, 3, 5, 9, 18])
    p.add_argument("--head", default="lstm", choices=sorted(HEADS),
                   help="how the window is combined. The recurrent head reads "
                        "its final state; the transformer attends over the "
                        "window and reads the labelled position. Both reduce to "
                        "the frozen probe at k = 1.")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    meta = pd.read_csv(args.manifest)
    test_meta = pd.read_csv(args.test_manifest) if args.test_manifest else meta

    train_cache = CachedFeatures(args.train_features)
    val_cache = CachedFeatures(args.val_features)
    test_cache = CachedFeatures(args.test_features) if args.test_features else None

    # Pooled once and reused across every window, so that the comparison between
    # windows cannot differ through the features.
    train_x, train_y = train_cache.pooled(), train_cache.all_targets()
    val_x, val_y = val_cache.pooled(), val_cache.all_targets()
    test_x = test_y = None
    if test_cache is not None:
        test_x, test_y = test_cache.pooled(), test_cache.all_targets()

    dim = train_x.shape[1]
    grid = [{"lr": lr, "weight_decay": wd, "dropout": dr, "hidden": h}
            for lr in (1e-3, 3e-3) for wd in (0.0, 1e-2)
            for dr in (0.0, 0.1) for h in (256,)]

    print(f"encoder     {train_cache.manifest['encoder']['checkpoint_id']}")
    print(f"head        {args.head}")
    print(f"features    {dim}-dim, {len(train_x)} train / {len(val_x)} val"
          + (f" / {len(test_x)} test" if test_x is not None else ""))
    print(f"grid        {len(grid)} configs x {args.seeds} seeds per window")
    print(f"windows     {args.windows}\n")

    results: dict[str, Any] = {"encoder": train_cache.manifest["encoder"],
                               "windows": {}}
    print(f"{'k':>3}{'span':>8}{'val mAP':>10}{'sd':>8}{'test mAP':>10}{'sd':>8}"
          f"   config")
    for k in args.windows:
        tr_order = build_order(train_cache, meta, k)
        va_order = build_order(val_cache, meta, k)
        tr_ds = TrailingWindows(train_x, train_y, tr_order, k)
        va_ds = TrailingWindows(val_x, val_y, va_order, k)

        # Selection on validation, exactly as the frozen probe does, so the two
        # are comparable; the test split is scored once afterwards.
        best_cfg, best_score, best_runs = None, -1.0, None
        for cfg in grid:
            runs = [train_one(cfg, s, tr_ds, va_ds, dim, args.epochs,
                              args.patience, device, args.head)
                    for s in range(args.seeds)]
            mean = float(np.mean([r[0] for r in runs]))
            if mean > best_score:
                best_cfg, best_score, best_runs = cfg, mean, runs

        entry: dict[str, Any] = {
            "config": best_cfg,
            "val_map": best_score,
            "val_sd": float(np.std([r[0] for r in best_runs], ddof=1)),
            "epochs": [r[1] for r in best_runs],
        }

        test_line = ""
        if test_x is not None:
            te_order = build_order(test_cache, test_meta, k)
            te_ds = TrailingWindows(test_x, test_y, te_order, k)
            te_loader = DataLoader(te_ds, batch_size=512, shuffle=False, num_workers=0)
            scores = []
            for _, _, state in best_runs:
                head = HEADS[args.head](dim, int(best_cfg["hidden"]),
                                        float(best_cfg["dropout"])).to(device)
                head.load_state_dict(state)
                logits, targets = infer(head, te_loader, device)
                scores.append(compute_multilabel_metrics_from_logits(targets, logits))
            entry["test"] = {m: {"mean": float(np.mean([s[m] for s in scores])),
                                 "sd": float(np.std([s[m] for s in scores], ddof=1))}
                             for m in scores[0]}
            test_line = f"{entry['test']['mAP']['mean']:>10.4f}{entry['test']['mAP']['sd']:>8.4f}"

        results["windows"][str(k)] = entry
        print(f"{k:>3}{5*(k-1):>7}s{best_score:>10.4f}{entry['val_sd']:>8.4f}"
              f"{test_line or '         -       -'}   {best_cfg}")

    base = results["windows"].get(str(args.windows[0]))
    if base and "test" in base:
        print(f"\nchange from k={args.windows[0]}, test mAP")
        for k in args.windows[1:]:
            e = results["windows"][str(k)]
            if "test" in e:
                d = e["test"]["mAP"]["mean"] - base["test"]["mAP"]["mean"]
                print(f"  k={k:<3} {d:+.4f}")
        print("\n  A monotone decline means temporal context is actively harmful when the")
        print("  label describes an instant. A flat curve means it contributes nothing.")
        print("  Report the shape, not the best window: selecting the argmax over five")
        print("  windows would reintroduce the selection problem the protocol avoids.")
    print(f"\n  SwinCVS pairs a frozen backbone with a 5-frame LSTM and reaches")
    print(f"  0.6553 mAP on Endoscapes in this project's reproduction. That is the")
    print(f"  closest published protocol, though on a different dataset.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results["head"] = args.head
    with open(out_dir / f"temporal_probe_{args.head}.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwritten to {out_dir / f'temporal_probe_{args.head}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
