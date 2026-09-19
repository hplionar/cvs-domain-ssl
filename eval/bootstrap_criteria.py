"""Video-clustered bootstrap intervals for the per-criterion analysis.

For each criterion, three quantities with 95% intervals:
  ceiling      each rater vs the consensus of the other two, on frames where
               those two agree, balanced accuracy, mean over raters
  model BAcc   on unanimous frames (votes 0 or 3) and contested frames (1 or 2)
  fourth-rater model minus rater k, on rater k's frames, paired in each
               replicate; mean over k
Videos are resampled with replacement; when several logits files are given
(one per seed), one is drawn per replicate so seed variance is inside.

Usage:
    python eval/bootstrap_criteria.py --manifest metadata/sages_frames_official_test.csv \
        --logits ../outputs/finetune/dinov3_b_lr3e-4/logits_official_test.npz \
                 ../outputs/finetune/dinov3_b_lr3e-4_seed1/logits_official_test.npz ...
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score as bacc, average_precision_score

p = argparse.ArgumentParser()
p.add_argument("--manifest", required=True)
p.add_argument("--logits", nargs="+", required=True)
p.add_argument("--n-boot", type=int, default=2000)
p.add_argument("--threshold-rule", choices=["max-bacc", "prevalence"], default="max-bacc",
               help="how the validation threshold is chosen per seed and criterion: "
                    "the cut maximising validation balanced accuracy, or the cut at "
                    "which the model's validation positive rate equals validation prevalence")
p.add_argument("--val-logits", nargs="+", default=None,
               help="one logits_val.npz per --logits file, same order; the threshold "
                    "per seed and criterion is the one that maximises validation "
                    "balanced accuracy. Omitted: fixed 0.5.")
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()

meta = pd.read_csv(a.manifest)
probs = np.stack([1 / (1 + np.exp(-np.load(f)["logits"])) for f in a.logits])  # [S, N, 3]

def val_threshold(path):
    d = np.load(path); pv = 1 / (1 + np.exp(-d["logits"])); yv = d["targets"].astype(int)
    out = []
    for c in range(3):
        if a.threshold_rule == "prevalence":
            q = 1.0 - yv[:, c].mean()
            out.append(float(np.quantile(pv[:, c], q)))
        else:
            cands = np.unique(np.round(pv[:, c], 3))
            scores_c = [(bacc(yv[:, c], pv[:, c] > t), t) for t in cands]
            out.append(max(scores_c)[1])
    return np.array(out)

if a.val_logits:
    assert len(a.val_logits) == len(a.logits), "--val-logits must match --logits one for one"
    thr = np.stack([val_threshold(f) for f in a.val_logits])  # [S, 3]
    print(f"thresholds chosen on validation, rule = {a.threshold_rule} (seed x criterion):")
    for s in range(thr.shape[0]):
        print("  seed", s, " ".join(f"C{c+1}={thr[s,c]:.3f}" for c in range(3)))
else:
    thr = np.full((probs.shape[0], 3), 0.5)
    print("threshold: fixed 0.5")
assert probs.shape[1] == len(meta)
videos = meta["video_id"].to_numpy()
uv = np.unique(videos)
rows = {v: np.flatnonzero(videos == v) for v in uv}
rng = np.random.default_rng(a.seed)

def safe_bacc(y, yhat):
    return bacc(y, yhat) if y.min() != y.max() else np.nan

def stats(idx, s):
    out = {}
    pr = probs[s][idx]
    for c in (1, 2, 3):
        y = meta[f"c{c}_consensus"].to_numpy(int)[idx]
        votes = meta[f"c{c}_votes"].to_numpy(int)[idx]
        r = meta[[f"c{c}_rater{k}" for k in (1, 2, 3)]].to_numpy(int)[idx]
        unan = np.isin(votes, [0, 3])
        out[f"C{c} AP unanimous"] = average_precision_score(y[unan], pr[unan, c-1]) if y[unan].min() != y[unan].max() else np.nan
        out[f"C{c} AP contested"] = average_precision_score(y[~unan], pr[~unan, c-1]) if y[~unan].min() != y[~unan].max() else np.nan
        out[f"C{c} model unanimous"] = safe_bacc(y[unan], pr[unan, c-1] > thr[s, c-1])
        out[f"C{c} model contested"] = safe_bacc(y[~unan], pr[~unan, c-1] > thr[s, c-1])
        ceil, diff = [], []
        for k in range(3):
            others = np.delete(r, k, axis=1)
            agree = others[:, 0] == others[:, 1]
            t = others[agree, 0]
            rb = safe_bacc(t, r[agree, k])
            mb = safe_bacc(t, pr[agree, c-1] > thr[s, c-1])
            ceil.append(rb); diff.append(mb - rb)
        out[f"C{c} ceiling"] = np.nanmean(ceil)
        out[f"C{c} model minus rater"] = np.nanmean(diff)
    return out

point = stats(np.arange(len(meta)), 0) if probs.shape[0] == 1 else \
        {k: np.mean([stats(np.arange(len(meta)), s)[k] for s in range(probs.shape[0])]) for k in stats(np.arange(len(meta)), 0)}
draws = {k: [] for k in point}
for _ in range(a.n_boot):
    idx = np.concatenate([rows[v] for v in rng.choice(uv, uv.size, replace=True)])
    s = rng.integers(probs.shape[0])
    for k, v in stats(idx, s).items():
        draws[k].append(v)

print(f"seeds={probs.shape[0]}  videos={uv.size}  replicates={a.n_boot}\n")
print(f"{'quantity':<28}{'point':>8}{'95% CI':>20}")
for k in point:
    d = np.asarray(draws[k]); lo, hi = np.nanpercentile(d, [2.5, 97.5])
    print(f"{k:<28}{point[k]:>8.3f}   [{lo:.3f}, {hi:.3f}]")
