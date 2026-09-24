"""Link the Kaggle Chapman release to the raw PhysioNet ecg-arrhythmia release.

The Kaggle parquet carries no record identifiers and every lead is min-max scaled
to [0, 1], so records are matched on signal: each raw record is scaled the same way,
a coarse lead-II fingerprint finds the nearest Kaggle candidate, and the match is
accepted only if all 12 leads correlate above MATCH_R.  The same pass flags raw
records whose ADC data are byte-identical (duplicates inside the release).

Outputs -> artifacts_v2/dedupe/: raw_index.csv (every raw record: id, path, header
fields, Dx, duplicate group) and kaggle_match.csv (Kaggle row -> raw id, r).

Run:  python -m src.dedupe
"""
import hashlib
from multiprocessing import Pool

import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.neighbors import NearestNeighbors

from . import config as C
from .data_prep_snomed import parse_header

RAW = C.ROOT / "ningbo" / "WFDBRecords"
OUT = C.ART_V2 / "dedupe"
STRIDE = 50          # 5000 -> 100-point fingerprint
MATCH_R = 0.99       # min per-lead correlation over all 12 leads


def _minmax(x):
    lo, hi = x.min(-1, keepdims=True), x.max(-1, keepdims=True)
    return (x - lo) / np.where(hi > lo, hi - lo, 1)


def _scan(hea):
    try:
        h = parse_header(hea)
    except Exception as e:                       # two malformed headers in v1.0.0
        return dict(rec=hea.stem, path=str(hea.with_suffix("")), error=f"header: {e!r}",
                    md5=None, fp=None)
    row = dict(rec=hea.stem, path=str(hea.with_suffix("")), n_sig=h["n_sig"], fs=h["fs"],
               n_samp=h["n_samp"], dx=",".join(sorted(h["dx"])), md5=None, fp=None)
    try:
        val = sio.loadmat(hea.with_suffix(".mat"))["val"]
    except Exception as e:                       # unreadable file -> excluded downstream
        row["error"] = repr(e)
        return row
    row["md5"] = hashlib.md5(np.ascontiguousarray(val).tobytes()).hexdigest()
    if h["n_sig"] == 12 and val.shape[1] == C.N_SAMPLES:
        row["fp"] = _minmax(val[1].astype(np.float32))[::STRIDE]
    return row


def _verify(args):
    k_idx, rec_path = args
    kag = np.load(C.CACHE / "signals.npy", mmap_mode="r")[k_idx]
    raw = _minmax(sio.loadmat(rec_path + ".mat")["val"].astype(np.float32))
    r = [np.corrcoef(kag[i], raw[i])[0, 1] if kag[i].std() > 0 and raw[i].std() > 0 else np.nan
         for i in range(C.N_LEADS)]
    return k_idx, float(np.nanmin(r)), int(np.isnan(r).sum())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    heas = sorted(RAW.rglob("*.hea"))
    print(f"scanning {len(heas)} raw records ...")
    with Pool(30) as pool:
        rows = pool.map(_scan, heas, chunksize=256)
    df = pd.DataFrame(rows)
    df["dup_group"] = df.groupby("md5").ngroup().where(df.duplicated("md5", keep=False))
    print(f"unreadable: {df.md5.isna().sum()} {df.loc[df.md5.isna(), 'rec'].tolist()}   byte-identical duplicates: "
          f"{int(df.dup_group.notna().sum())} records in {df.dup_group.nunique()} groups")

    # nearest raw fingerprint for every Kaggle record
    cand = df[df.fp.notna()].reset_index(drop=True)
    F = np.stack(cand.fp.to_numpy())
    kag = np.load(C.CACHE / "signals.npy", mmap_mode="r")
    Q = np.asarray(kag[:, 1, ::STRIDE])
    dist, nn = NearestNeighbors(n_neighbors=1).fit(F).kneighbors(Q)
    jobs = [(k, cand.path[nn[k, 0]]) for k in range(len(Q))]
    with Pool(30) as pool:
        ver = pool.map(_verify, jobs, chunksize=64)
    m = pd.DataFrame(ver, columns=["kaggle_idx", "min_lead_r", "flat_leads"])
    m["rec"] = [cand.rec[nn[k, 0]] for k in m.kaggle_idx]
    m["fp_dist"] = dist[:, 0]
    m["matched"] = m.min_lead_r >= MATCH_R
    m["rec_num"] = m.rec.str[2:].astype(int)
    m.to_csv(OUT / "kaggle_match.csv", index=False)
    df.drop(columns="fp").to_csv(OUT / "raw_index.csv", index=False)

    mm = m[m.matched]
    print(f"Kaggle records matched (min lead r >= {MATCH_R}): {len(mm)} / {len(m)}")
    print(f"distinct raw ids matched: {mm.rec.nunique()}")
    print(f"matched raw id range: {mm.rec.min()} .. {mm.rec.max()}  "
          f"(<= JS10646: {(mm.rec_num <= 10646).sum()}, > JS10646: {(mm.rec_num > 10646).sum()})")
    print("min-lead r quantiles:", m.min_lead_r.quantile([0, .01, .05, .5]).round(4).to_dict())


CHAPMAN_LAST = 10646     # JS00001-JS10646 = Chapman-Shaoxing (verified identical to the
                         # Challenge 2021 chapman_shaoxing folder); higher ids = Ningbo


def build_exclusions():
    """Records to drop so every signal appears once and the cohorts are disjoint.

    byte-identical pair across cohorts  -> drop the Ningbo copy (Chapman is primary)
    pair inside one cohort, same Dx     -> keep the lower id
    pair inside one cohort, other Dx    -> drop both (label cannot be resolved)
    """
    df = pd.read_csv(OUT / "raw_index.csv")
    df["cohort"] = np.where(df.rec.str[2:].astype(int) <= CHAPMAN_LAST, "chapman", "ningbo")
    # Chapman is read from the Challenge 2021 copy, whose JS01052 header is intact
    bad = df.md5.isna() & (df.cohort == "ningbo")
    rows = [dict(rec=r, cohort=c, reason="malformed header")
            for r, c in df.loc[bad, ["rec", "cohort"]].itertuples(index=False)]
    for _, g in df[df.dup_group.notna()].groupby("dup_group"):
        g = g.sort_values("rec")
        dx_same = g.dx.fillna("").nunique() == 1
        if g.cohort.nunique() > 1:
            drop, why = g[g.cohort == "ningbo"], "duplicate of a Chapman record"
        elif dx_same:
            drop, why = g.iloc[1:], "within-cohort duplicate (same Dx)"
        else:
            drop, why = g, "within-cohort duplicate (conflicting Dx)"
        rows += [dict(rec=r, cohort=c, reason=why, dx_conflict=not dx_same)
                 for r, c in drop[["rec", "cohort"]].itertuples(index=False)]
    ex = pd.DataFrame(rows)
    ex.to_csv(OUT / "exclusions.csv", index=False)
    print(ex.groupby(["cohort", "reason"]).size().to_string())
    return ex


if __name__ == "__main__":
    main()
    build_exclusions()
