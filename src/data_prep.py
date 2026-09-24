"""One-time data preparation:
  1. Parquet -> cached float32 array  signals.npy  (N, 12, 5000)  + labels.npy (N, 11)
  2. Record-level multilabel-stratified train/val/test split -> splits/split.npz

Run:  python -m src.data_prep      (from /data/lead_reduction_failure)
"""
import numpy as np
import pyarrow.parquet as pq

from . import config as C


def build_cache():
    C.CACHE.mkdir(parents=True, exist_ok=True)
    sig_path = C.CACHE / "signals.npy"
    lab_path = C.CACHE / "labels.npy"
    if sig_path.exists() and lab_path.exists():
        print(f"cache exists: {sig_path.name}, {lab_path.name} -> skip")
        return

    pf = pq.ParquetFile(str(C.PARQUET))
    n = pf.metadata.num_rows
    print(f"reading {n} records ...")

    # labels
    lab = pf.read(columns=C.LABEL_COLS).to_pandas().to_numpy().astype(np.int8)
    np.save(lab_path, lab)
    print("labels:", lab.shape, "->", lab_path.name)

    # signals: 60000 cols -> (N, 12, 5000) lead-major
    sig_cols = [f"signal_{i}" for i in range(1, C.N_LEADS * C.N_SAMPLES + 1)]
    arr = pf.read(columns=sig_cols).to_pandas().to_numpy().astype(np.float32)
    arr = arr.reshape(n, C.N_LEADS, C.N_SAMPLES)
    np.save(sig_path, arr)
    print("signals:", arr.shape, f"({arr.nbytes/1e9:.2f} GB)", "->", sig_path.name)


def iterative_stratify(Y, fracs, seed):
    """Iterative stratification (Sechidis et al. 2011) for multilabel data.
    Y: (N, L) binary.  fracs: e.g. (0.7,0.15,0.15).  Returns list of index arrays."""
    rng = np.random.default_rng(seed)
    N, L = Y.shape
    n_sub = len(fracs)
    fracs = np.asarray(fracs, float)
    fracs = fracs / fracs.sum()

    # desired number of samples per subset, overall and per label
    target = fracs * N
    label_target = fracs[:, None] * Y.sum(0)[None, :]   # (n_sub, L)

    subsets = [[] for _ in range(n_sub)]
    sub_count = np.zeros(n_sub)
    sub_label = np.zeros((n_sub, L))

    remaining = set(range(N))
    # order labels by rarity (fewest positives first)
    label_order = np.argsort(Y.sum(0))

    # process rare-label-positive samples first, then the rest
    for lab in list(label_order) + [-1]:
        if lab == -1:
            pool = [i for i in remaining]
        else:
            pool = [i for i in remaining if Y[i, lab] == 1]
        rng.shuffle(pool)
        for i in pool:
            if i not in remaining:
                continue
            if lab == -1:
                # assign to whichever subset is most short on total count
                desire = target - sub_count
                cand = np.flatnonzero(desire == desire.max())
            else:
                desire = label_target[:, lab] - sub_label[:, lab]
                cand = np.flatnonzero(desire == desire.max())
            if len(cand) > 1:
                # tie-break on overall count shortfall
                d2 = (target - sub_count)[cand]
                cand = cand[d2 == d2.max()]
            s = int(rng.choice(cand))
            subsets[s].append(i)
            sub_count[s] += 1
            sub_label[s] += Y[i]
            remaining.discard(i)

    return [np.array(sorted(s), dtype=np.int64) for s in subsets]


def build_split():
    C.SPLITS.mkdir(parents=True, exist_ok=True)
    out = C.SPLITS / "split.npz"
    lab = np.load(C.CACHE / "labels.npy")
    tr, va, te = iterative_stratify(lab.astype(int), C.SPLIT_FRACS, C.SEED)
    np.savez(out, train=tr, val=va, test=te)
    print(f"split -> {out.name}  train={len(tr)} val={len(va)} test={len(te)}")

    # report per-class prevalence per split
    print(f"\n{'class':6s} {'all':>6s} {'train':>7s} {'val':>6s} {'test':>6s}")
    tot = lab.sum(0)
    for j, c in enumerate(C.CLASSES):
        print(f"{c:6s} {int(tot[j]):6d} "
              f"{int(lab[tr,j].sum()):7d} {int(lab[va,j].sum()):6d} {int(lab[te,j].sum()):6d}")
    return tr, va, te


if __name__ == "__main__":
    build_cache()
    build_split()
