"""PTB-XL data preparation for external replication of the blind-spot map.

Maps PTB-XL (SCP codes + heart_axis) onto the SAME 11-class space as Chapman,
reads the 500 Hz records into (N,12,5000), and uses the official patient-disjoint
strat_fold (train=1-8, val=9, test=10).

Run:  python -m src.data_prep_ptbxl        (study-1 cache)
      python -m src.data_prep_ptbxl --v2   (resubmission narrow labels)
"""
import ast
from multiprocessing import Pool

import numpy as np
import pandas as pd
import wfdb

from . import config as C

P = C.ds_paths("ptbxl")


def build_labels(db, scp_map=None):
    scp_map = scp_map or C.PTBXL_SCP_MAP
    N = len(db)
    Y = np.zeros((N, C.N_CLASSES), dtype=np.int8)
    scp_lists = [ast.literal_eval(s) for s in db["scp_codes"]]
    axis = db["heart_axis"].fillna("").to_numpy()
    for j, cls in enumerate(C.CLASSES):
        codes = scp_map[cls]
        for i in range(N):
            if "__AXIS_LAD__" in codes:
                if axis[i] in C.PTBXL_AXIS_LAD:
                    Y[i, j] = 1
            else:
                if any(c in scp_lists[i] for c in codes):
                    Y[i, j] = 1
    return Y


def _read_one(args):
    i, path = args
    rec = wfdb.rdrecord(path)               # physical signal (mV), shape (5000,12)
    sig = rec.p_signal.astype(np.float32)
    # reorder channels to canonical LEADS order
    name2idx = {n.upper(): k for k, n in enumerate(rec.sig_name)}
    order = [name2idx[l.upper()] for l in C.LEADS]
    sig = sig[:, order].T                   # (12,5000)
    sig = np.nan_to_num(sig, nan=0.0)
    return i, sig


def build_cache():
    P.cache.mkdir(parents=True, exist_ok=True)
    sig_path, lab_path = P.cache / "signals.npy", P.cache / "labels.npy"
    db = pd.read_csv(C.PTBXL_ROOT / "ptbxl_database.csv")
    if sig_path.exists() and lab_path.exists():
        print("ptbxl cache exists -> skip"); return db

    Y = build_labels(db)
    np.save(lab_path, Y)
    print("labels:", Y.shape, "positives/class:", dict(zip(C.CLASSES, Y.sum(0).tolist())))

    paths = [str(C.PTBXL_ROOT / f) for f in db["filename_hr"]]
    sig = np.zeros((len(db), C.N_LEADS, C.N_SAMPLES), dtype=np.float32)
    print(f"reading {len(db)} records (500Hz) ...")
    with Pool(16) as pool:
        for k, (i, s) in enumerate(pool.imap_unordered(_read_one, list(enumerate(paths)), chunksize=64)):
            sig[i] = s
            if (k + 1) % 4000 == 0:
                print(f"  {k+1}/{len(db)}")
    np.save(sig_path, sig)
    print("signals:", sig.shape, f"({sig.nbytes/1e9:.2f} GB)")
    return db


def build_split(db):
    P.splits.mkdir(parents=True, exist_ok=True)
    fold = db["strat_fold"].to_numpy()
    tr = np.where(fold <= 8)[0]
    va = np.where(fold == 9)[0]
    te = np.where(fold == 10)[0]
    np.savez(P.splits / "split.npz", train=tr, val=va, test=te)
    lab = np.load(P.cache / "labels.npy")
    print(f"\nsplit (official strat_fold, patient-disjoint): "
          f"train={len(tr)} val={len(va)} test={len(te)}")
    print(f"{'class':6s} {'all':>6s} {'train':>7s} {'val':>6s} {'test':>6s}")
    for j, c in enumerate(C.CLASSES):
        print(f"{c:6s} {int(lab[:,j].sum()):6d} {int(lab[tr,j].sum()):7d} "
              f"{int(lab[va,j].sum()):6d} {int(lab[te,j].sum()):6d}")


def build_v2():
    """Resubmission cache: narrow labels (+ broad for sensitivity) under artifacts_v2,
    signals symlinked to the study-1 cache, same official patient-disjoint split."""
    V = C.ds_paths("ptbxl", "any")      # cache/splits are shared by all archs
    V.cache.mkdir(parents=True, exist_ok=True); V.splits.mkdir(parents=True, exist_ok=True)
    db = pd.read_csv(C.PTBXL_ROOT / "ptbxl_database.csv")
    Yn, Yb = build_labels(db, C.PTBXL_SCP_MAP_NARROW), build_labels(db, C.PTBXL_SCP_MAP)
    assert (Yb == np.load(P.cache / "labels.npy")).all(), "broad labels differ from study-1 cache"
    np.save(V.cache / "labels.npy", Yn)
    np.save(V.cache / "labels_broad.npy", Yb)
    link = V.cache / "signals.npy"
    if not link.exists():
        link.symlink_to(P.cache / "signals.npy")
    sp = np.load(P.splits / "split.npz")
    np.savez(V.splits / "split.npz", **{k: sp[k] for k in sp.files})
    te = sp["test"]
    print(f"{'class':6s} {'narrow':>7s} {'broad':>7s} {'test(n)':>8s}")
    for j, c in enumerate(C.CLASSES):
        print(f"{c:6s} {int(Yn[:, j].sum()):7d} {int(Yb[:, j].sum()):7d} {int(Yn[te, j].sum()):8d}")


if __name__ == "__main__":
    import sys
    if "--v2" in sys.argv:
        build_v2()
    else:
        db = build_cache()
        build_split(db)
