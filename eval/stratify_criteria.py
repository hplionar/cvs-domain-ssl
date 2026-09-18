"""Per-criterion AP on unanimous versus contested frames, and model balanced
accuracy against the rater ceiling, for one fine-tuned run's official-test logits.

Unanimous: all three raters gave the same label (votes 0 or 3).
Contested:  votes 1 or 2; the consensus is a majority, not an agreement.
Ceiling:    each rater scored against the consensus of the other two, on frames
            where those two agree, in balanced accuracy; mean over raters.
Model BAcc is at a 0.5 sigmoid threshold and is threshold-dependent; AP is not.

Usage:
    python eval/stratify_criteria.py \
        --logits ../outputs/finetune/dinov3_b_lr3e-4/logits_official_test.npz \
        --manifest metadata/sages_frames_official_test.csv
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, balanced_accuracy_score

p = argparse.ArgumentParser()
p.add_argument("--logits", required=True)
p.add_argument("--manifest", required=True)
a = p.parse_args()

d = np.load(a.logits)
prob = 1 / (1 + np.exp(-d["logits"]))
meta = pd.read_csv(a.manifest)
assert len(meta) == prob.shape[0], f"{len(meta)} manifest rows vs {prob.shape[0]} logits"

print(f"{'crit':<5}{'stratum':<11}{'frames':>7}{'pos%':>7}{'AP':>8}{'BAcc':>8}")
for c in (1, 2, 3):
    y = meta[f"c{c}_consensus"].to_numpy(int)
    assert np.array_equal(y, d["targets"][:, c-1].astype(int)), "targets differ from manifest"
    votes = meta[f"c{c}_votes"].to_numpy(int)
    unan = np.isin(votes, [0, 3])
    for name, m in (("all", np.ones_like(unan)), ("unanimous", unan), ("contested", ~unan)):
        yy, pp = y[m], prob[m, c-1]
        ap = average_precision_score(yy, pp) if yy.min() != yy.max() else float("nan")
        ba = balanced_accuracy_score(yy, pp > 0.5) if yy.min() != yy.max() else float("nan")
        print(f"C{c:<4}{name:<11}{m.sum():>7}{100*yy.mean():>6.1f}%{ap:>8.4f}{ba:>8.4f}")
    # rater ceiling, same construction as tab_ceiling
    r = meta[[f"c{c}_rater{k}" for k in (1, 2, 3)]].to_numpy(int)
    cs = []
    for k in range(3):
        others = np.delete(r, k, axis=1)
        agree = others[:, 0] == others[:, 1]
        cs.append(balanced_accuracy_score(others[agree, 0], r[agree, k]))
    print(f"C{c:<4}{'ceiling':<11}{'':>7}{'':>7}{'':>8}{np.mean(cs):>8.4f}")
    print()

print("fourth-rater test: model vs rater k on frames where the other two agree")
print(f"{'crit':<5}{'rater':>6}{'frames':>7}{'rater BAcc':>12}{'model BAcc':>12}")
for c in (1, 2, 3):
    r = meta[[f"c{c}_rater{k}" for k in (1, 2, 3)]].to_numpy(int)
    for k in range(3):
        others = np.delete(r, k, axis=1)
        agree = others[:, 0] == others[:, 1]
        t = others[agree, 0]
        rb = balanced_accuracy_score(t, r[agree, k])
        mb = balanced_accuracy_score(t, prob[agree, c-1] > 0.5)
        print(f"C{c:<4}{k+1:>6}{agree.sum():>7}{rb:>12.4f}{mb:>12.4f}")
    print()
