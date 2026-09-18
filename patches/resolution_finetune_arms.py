"""benchmark_resolution.py: load fine-tuned arms from outputs/finetune.

Each arm is <name> plus every <name>_seed* directory, one logits_official_test.npz
per seed, stacked [S, N, 3]. Frame order is the official manifest in file order,
which is what eval/score_finetune.py walks with shuffle=False."""
from pathlib import Path
p = Path("eval/benchmark_resolution.py"); src = p.read_text()
def rep(old, new):
    global src
    n = src.count(old); assert n == 1, f"expected 1 match, found {n}:\n{old[:90]}"
    src = src.replace(old, new)

rep('''def mean_ap(y: np.ndarray, p: np.ndarray) -> float:''',
'''def load_finetune_arm(root: Path, name: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Sigmoid scores per seed [S, N, 3] and targets [N, 3] for a fine-tuned arm.

    Seed runs live in sibling directories <name>_seed<k>; <name> itself is seed 0.
    Targets are read from the first file and checked against every other, which
    catches a run scored against a different manifest.
    """
    dirs = [root / name] + sorted(root.glob(f"{name}_seed*"))
    scores, targets = [], None
    for d in dirs:
        f = d / "logits_official_test.npz"
        if not f.is_file():
            continue
        payload = np.load(f)
        scores.append(1.0 / (1.0 + np.exp(-payload["logits"])))
        t = payload["targets"].astype(int)
        if targets is None:
            targets = t
        elif not np.array_equal(t, targets):
            raise SystemExit(f"{f}: targets differ from {dirs[0]}; not the same split")
    if not scores:
        return None
    return np.stack(scores, axis=0), targets


def mean_ap(y: np.ndarray, p: np.ndarray) -> float:''')

rep('''    p.add_argument("--probe-root", required=True)
    p.add_argument("--cache-root", required=True)
''',
'''    p.add_argument("--probe-root", default=None)
    p.add_argument("--cache-root", default=None)
    p.add_argument("--finetune-root", default=None,
                   help="outputs/finetune; use with --finetune-arms instead of "
                        "--probe-root/--cache-root")
    p.add_argument("--finetune-arms", nargs="+", default=None,
                   help="run names; each is <name> plus every <name>_seed*")
''')

rep('''    probe_root, cache_root = Path(args.probe_root), Path(args.cache_root)
    meta = pd.read_csv(args.manifest).set_index("sample_id")

    scores, sample_ids, targets = {}, None, None
    candidates = (args.arms if args.arms else
                  sorted(d.name for d in probe_root.iterdir() if d.is_dir()))
    for name in candidates:
''',
'''    meta = pd.read_csv(args.manifest).set_index("sample_id")

    scores, sample_ids, targets = {}, None, None
    if args.finetune_root:
        if not args.finetune_arms:
            raise SystemExit("--finetune-root needs --finetune-arms")
        sample_ids = list(meta.index)  # score_finetune walks the manifest in file order
        for name in args.finetune_arms:
            loaded = load_finetune_arm(Path(args.finetune_root), name)
            if loaded is None:
                print(f"  no logits for {name}"); continue
            s, t = loaded
            if s.shape[1] != len(sample_ids):
                raise SystemExit(f"{name}: {s.shape[1]} rows against {len(sample_ids)} manifest rows")
            if targets is None:
                targets = t
            elif not np.array_equal(t, targets):
                raise SystemExit(f"{name}: targets differ from the first arm")
            scores[name] = s
        candidates = []
        probe_root = cache_root = None
    else:
        probe_root, cache_root = Path(args.probe_root), Path(args.cache_root)
        candidates = (args.arms if args.arms else
                      sorted(d.name for d in probe_root.iterdir() if d.is_dir()))
    for name in candidates:
''')

p.write_text(src); print("patched", p)
