"""Data preparation for WFDB/SNOMED-CT cohorts (Georgia, Chapman, Ningbo, PTB-XL).

Reads .mat/.hea records into (N,12,5000) float32 mV in the canonical lead order,
maps the #Dx SNOMED codes onto the 11-class space (config.SNOMED_MAP, plus the broad
map for sensitivity analysis), and makes a record-level multilabel-stratified
70/15/15 split (no patient identifiers are distributed with these cohorts).
Records that are not 12-lead, 500 Hz, 10 s are excluded and counted.

Georgia: signals from georgia/Georgia/*.mat (checksum-identical to Challenge 2021
v1.0.3), gain and labels from the official 2021 headers in georgia/hea_2021/ (the
2020 headers shipped with the local copy give a 4880/mV gain, ~5x too small).
Chapman: Challenge 2021 chapman_shaoxing folder (= ecg-arrhythmia JS00001-JS10646).
Ningbo: ecg-arrhythmia records above JS10646.  For both, records listed in
artifacts_v2/dedupe/exclusions.csv (duplicates, malformed header) are dropped.
PTB-XL (ptbxl_snomed): Challenge 2021 SNOMED labels; HRnnnnn = ecg_id and the
signals equal PTB-XL v1.0.1, so the study-1 signal cache is symlinked and the
official patient-disjoint strat_fold split (1-8 / 9 / 10) is used.

Run:  python -m src.data_prep_snomed --dataset georgia | chapman | ningbo | ptbxl_snomed
"""
import argparse
import json
import re
from multiprocessing import Pool

import numpy as np
import pandas as pd
import scipy.io as sio

from . import config as C
from .data_prep import iterative_stratify

CH21 = C.ROOT / "challenge2021" / "training"
SOURCES = {
    "georgia": dict(hea_dir=C.ROOT / "georgia" / "hea_2021", mat_dir=C.ROOT / "georgia" / "Georgia"),
    "chapman": dict(hea_dir=CH21 / "chapman_shaoxing", dedupe=True),
    "ningbo": dict(hea_dir=C.ROOT / "ningbo" / "WFDBRecords", dedupe=True,
                   keep=lambda rec: int(rec[2:]) > 10646),
    "ptbxl_snomed": dict(hea_dir=CH21 / "ptb-xl", ptbxl=True),
}


def parse_header(path):
    lines = path.read_text().splitlines()
    rec, n_sig, fs, n_samp = lines[0].split()[:4]
    gains, names = [], []
    for ln in lines[1:1 + int(n_sig)]:
        f = ln.split()
        gains.append(float(re.match(r"[\d.]+", f[2]).group()))
        base = re.search(r"\((-?\d+)\)", f[2])
        if base and int(base.group(1)) != 0:
            raise ValueError(f"{path.name}: non-zero baseline")
        names.append(f[-1])
    dx = set()
    for ln in lines:
        m = re.match(r"#\s*Dx:\s*(.*)", ln)
        if m:
            dx = {c.strip() for c in m.group(1).split(",") if c.strip()}
    return dict(n_sig=int(n_sig), fs=int(float(fs)), n_samp=int(n_samp),
                gains=np.array(gains), names=names, dx=dx)


def _labels(dx, mapping):
    return np.array([int(bool(dx & set(mapping[c]))) for c in C.CLASSES], dtype=np.int8)


def _read_one(args):
    i, mat, h = args
    val = sio.loadmat(mat)["val"].astype(np.float32)              # (12, n) ADC units
    order = [[n.upper() for n in h["names"]].index(l.upper()) for l in C.LEADS]
    return i, (val / h["gains"][:, None].astype(np.float32))[order]


def build(dataset):
    src = SOURCES[dataset]
    P = C.ds_paths(dataset, "any")      # cache/splits are shared by all archs
    P.cache.mkdir(parents=True, exist_ok=True); P.splits.mkdir(parents=True, exist_ok=True)
    heas = sorted(src["hea_dir"].rglob("*.hea"), key=lambda h: h.stem)
    if "keep" in src:
        heas = [h for h in heas if src["keep"](h.stem)]
    drop = {}
    if src.get("dedupe"):
        ex = pd.read_csv(C.ART_V2 / "dedupe" / "exclusions.csv")
        drop = dict(zip(ex.rec, ex.reason))
    keep, ids, paths, excluded = [], [], [], {}
    for hp in heas:
        why = drop.get(hp.stem)
        if why is None:
            h = parse_header(hp)
            why = ("not 12-lead" if h["n_sig"] != 12 else "fs != 500" if h["fs"] != C.FS
                   else "length != 10 s" if h["n_samp"] != C.N_SAMPLES else None)
        if why:
            excluded[why] = excluded.get(why, 0) + 1
            continue
        keep.append(h); ids.append(hp.stem)
        paths.append(src["mat_dir"] / f"{hp.stem}.mat" if "mat_dir" in src else hp.with_suffix(".mat"))
    Y = np.stack([_labels(h["dx"], C.SNOMED_MAP) for h in keep])
    Yb = np.stack([_labels(h["dx"], C.SNOMED_MAP_BROAD) for h in keep])

    if src.get("ptbxl"):
        assert ids == [f"HR{e:05d}" for e in range(1, len(ids) + 1)], "HR ids != ecg_id order"
        legacy = C.ds_paths("ptbxl")
        link = P.cache / "signals.npy"
        if not link.exists():
            link.symlink_to(legacy.cache / "signals.npy")
        sig = np.load(link, mmap_mode="r")
        sp = np.load(legacy.splits / "split.npz")
        tr, va, te = sp["train"], sp["val"], sp["test"]
    else:
        sig = np.zeros((len(keep), C.N_LEADS, C.N_SAMPLES), dtype=np.float32)
        jobs = [(i, pth, h) for i, (pth, h) in enumerate(zip(paths, keep))]
        with Pool(16) as pool:
            for i, s in pool.imap_unordered(_read_one, jobs, chunksize=64):
                sig[i] = s
        np.save(P.cache / "signals.npy", sig)
        tr, va, te = iterative_stratify(Y.astype(int), C.SPLIT_FRACS, C.SEED)
    np.save(P.cache / "labels.npy", Y)
    np.save(P.cache / "labels_broad.npy", Yb)
    np.save(P.cache / "record_ids.npy", np.array(ids))
    np.savez(P.splits / "split.npz", train=tr, val=va, test=te)

    sub = np.asarray(sig[::max(1, len(keep) // 3000)])            # QC on a subsample
    e = sub[:, 2] - (sub[:, 1] - sub[:, 0])                       # Einthoven: III = II - I
    rep = dict(dataset=dataset, n_headers=len(heas), n_kept=len(keep), excluded=excluded,
               einthoven_rms_mV=float(np.sqrt(np.mean(e ** 2))),
               lead_II_rms_mV=float(np.sqrt(np.mean(sub[:, 1] ** 2))),
               split=dict(train=len(tr), val=len(va), test=len(te)),
               positives={c: dict(all=int(Y[:, j].sum()), broad=int(Yb[:, j].sum()),
                                  test=int(Y[te, j].sum())) for j, c in enumerate(C.CLASSES)})
    with open(P.cache / "prep_report.json", "w") as f:
        json.dump(rep, f, indent=2)
    print(json.dumps({k: v for k, v in rep.items() if k != "positives"}, indent=2))
    print(f"{'class':6s} {'all':>6s} {'broad':>6s} {'test':>5s}")
    for c, v in rep["positives"].items():
        print(f"{c:6s} {v['all']:6d} {v['broad']:6d} {v['test']:5d}")


def build_broad(dataset):
    """Sensitivity analysis: '<dataset>_broad' shares the signals (symlink) and the split of
    <dataset> but uses the broad label definition (config.SNOMED_MAP_BROAD)."""
    src, dst = C.ds_paths(dataset, "any"), C.ds_paths(f"{dataset}_broad", "any")
    dst.cache.mkdir(parents=True, exist_ok=True); dst.splits.mkdir(parents=True, exist_ok=True)
    link = dst.cache / "signals.npy"
    if not link.exists():
        link.symlink_to((src.cache / "signals.npy").resolve())
    np.save(dst.cache / "labels.npy", np.load(src.cache / "labels_broad.npy"))
    sp = np.load(src.splits / "split.npz")
    np.savez(dst.splits / "split.npz", **{k: sp[k] for k in sp.files})
    print(f"{dataset}_broad ready")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="georgia", choices=list(SOURCES))
    ap.add_argument("--broad", action="store_true", help="make <dataset>_broad from an existing cache")
    a = ap.parse_args()
    build_broad(a.dataset) if a.broad else build(a.dataset)
