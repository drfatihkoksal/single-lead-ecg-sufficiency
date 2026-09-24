"""Blind-spot analysis from trained-model predictions.

Builds:
  - per-(lead,class) AUPRC / AUROC / F1 matrices
  - gap-to-ceiling matrix  gap(L,C) = metric_ALL(C) - metric_L(C)
  - record-level bootstrap CIs on each AUPRC gap cell
  - failure-type taxonomy  (Type A invisibility / Type B confusion)

Run:  python -m src.evaluate
Outputs -> artifacts/metrics/blindspot.npz  + analysis.json
"""
import json

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from . import config as C
from .train import atomic_save

# pre-registered thresholds (concept §5.4 / §6) -----------------------------
GAP_FLAG = 0.15      # AUPRC gap above which a (lead,class) cell is a "blind spot"
AUROC_INVIS = 0.65   # below this on the lead -> Type A invisibility; else Type B
FDR_ALPHA = 0.05     # Benjamini-Hochberg q-value cutoff across all 12x11 cells
N_BOOT = 1000


def _load_preds(dataset="chapman", arch=None, seed=None):
    P = C.ds_paths(dataset, arch, seed)
    preds = {}
    for spec in list(C.LEADS) + ["ALL"]:
        d = np.load(P.metrics / f"pred_{spec}.npz")
        preds[spec] = (d["p"], d["y"])
    return preds


def _auprc(y, p):
    return average_precision_score(y, p) if y.sum() else np.nan


def _auroc(y, p):
    return roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan


def build_matrices(dataset="chapman", arch=None, seed=None):
    preds = _load_preds(dataset, arch, seed)
    leads = list(C.LEADS)
    L, K = len(leads), C.N_CLASSES
    auprc = np.full((L, K), np.nan)
    auroc = np.full((L, K), np.nan)
    pA, yA = preds["ALL"]
    auprc_ceil = np.array([_auprc(yA[:, j], pA[:, j]) for j in range(K)])
    auroc_ceil = np.array([_auroc(yA[:, j], pA[:, j]) for j in range(K)])
    for i, ld in enumerate(leads):
        p, y = preds[ld]
        for j in range(K):
            auprc[i, j] = _auprc(y[:, j], p[:, j])
            auroc[i, j] = _auroc(y[:, j], p[:, j])
    gap = auprc_ceil[None, :] - auprc            # (L,K) gap-to-ceiling in AUPRC
    return dict(leads=leads, auprc=auprc, auroc=auroc,
                auprc_ceil=auprc_ceil, auroc_ceil=auroc_ceil, gap=gap, preds=preds)


def bootstrap_all(preds, leads, seed=C.SEED):
    """Record-level bootstrap over ALL (lead,class) cells in one pass.

    Returns ci_lo, ci_hi, pval matrices (L,K). pval is one-sided for H0: gap<=0,
    computed as (1 + #{boot gap <= 0}) / (N_BOOT + 1). Resample indices are shared
    across leads within a class/iteration, and the ceiling AP is computed once per
    class/iteration, so cells are directly comparable and the pass is cheap."""
    L, K = len(leads), C.N_CLASSES
    pA, yA = preds["ALL"]
    ci_lo = np.full((L, K), np.nan)
    ci_hi = np.full((L, K), np.nan)
    pval = np.full((L, K), np.nan)
    n = pA.shape[0]
    rng = np.random.default_rng(seed)

    for j in range(K):
        y = yA[:, j]
        diffs = np.full((N_BOOT, L), np.nan)
        for b in range(N_BOOT):
            idx = rng.integers(0, n, n)
            yy = y[idx]
            if yy.sum() == 0 or yy.sum() == n:
                continue
            a_ceil = average_precision_score(yy, pA[idx, j])
            for i, ld in enumerate(leads):
                pL = preds[ld][0]
                diffs[b, i] = a_ceil - average_precision_score(yy, pL[idx, j])
        for i in range(L):
            d = diffs[:, i]
            d = d[~np.isnan(d)]
            if len(d) == 0:
                continue
            ci_lo[i, j] = np.percentile(d, 2.5)
            ci_hi[i, j] = np.percentile(d, 97.5)
            pval[i, j] = (1 + np.sum(d <= 0)) / (len(d) + 1)
    return ci_lo, ci_hi, pval


def bh_fdr(pvals):
    """Benjamini-Hochberg adjusted q-values for a flat array of p-values (NaN-safe)."""
    p = np.asarray(pvals, float)
    mask = ~np.isnan(p)
    q = np.full_like(p, np.nan)
    pv = p[mask]
    m = len(pv)
    order = np.argsort(pv)
    ranked = pv[order]
    adj = ranked * m / (np.arange(m) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]   # enforce monotonicity
    out = np.empty(m)
    out[order] = np.clip(adj, 0, 1)
    q[mask] = out
    return q


def failure_type(gap, auroc_lead):
    """Classify a flagged cell. Type A: lead can't detect (low AUROC).
    Type B: lead detects abnormal (AUROC ok) but discrimination/precision lost."""
    if gap < GAP_FLAG:
        return "ok"
    return "A_invisible" if auroc_lead < AUROC_INVIS else "B_confusion"


def main(dataset="chapman", arch=None, seed=None):
    P = C.ds_paths(dataset, arch, seed)
    M = build_matrices(dataset, arch, seed)
    leads, gap, auroc = M["leads"], M["gap"], M["auroc"]
    L, K = gap.shape
    analysis = {"gap_flag": GAP_FLAG, "auroc_invis": AUROC_INVIS,
                "fdr_alpha": FDR_ALPHA, "n_boot": N_BOOT, "cells": [], "universal": []}

    # bootstrap CIs + one-sided p-values for ALL cells, then BH-FDR across all 12x11
    print(f"bootstrapping all {L*K} cells x {N_BOOT} ...")
    ci_lo, ci_hi, pval = bootstrap_all(M["preds"], leads)
    qval = bh_fdr(pval.ravel()).reshape(L, K)

    # a cell is a blind spot if gap>flag AND FDR-significant (q<alpha)
    taxonomy = np.empty((L, K), dtype=object)
    for i, ld in enumerate(leads):
        for j, c in enumerate(C.CLASSES):
            sig = bool(qval[i, j] < FDR_ALPHA)
            t = failure_type(gap[i, j], auroc[i, j]) if sig else "ok"
            taxonomy[i, j] = t
            if t != "ok":
                analysis["cells"].append(dict(
                    lead=ld, cls=c, gap=round(float(gap[i, j]), 3),
                    auroc_lead=round(float(auroc[i, j]), 3),
                    ci=[round(float(ci_lo[i, j]), 3), round(float(ci_hi[i, j]), 3)],
                    p=float(pval[i, j]), q=round(float(qval[i, j]), 4),
                    type=t))

    # universal blind spots: flagged on ALL 12 leads
    for j, c in enumerate(C.CLASSES):
        if all(taxonomy[i, j] != "ok" for i in range(L)):
            analysis["universal"].append(c)

    # blindspot.npz is written last: it marks the evaluation complete (run_grid resume)
    atomic_save(P.metrics / "analysis.json", lambda t: t.write_text(json.dumps(analysis, indent=2)))
    atomic_save(P.metrics / "blindspot.npz", lambda t: np.savez(
        t, leads=np.array(leads), classes=np.array(C.CLASSES),
        auprc=M["auprc"], auroc=M["auroc"], gap=gap,
        auprc_ceil=M["auprc_ceil"], auroc_ceil=M["auroc_ceil"],
        taxonomy=taxonomy.astype(str), ci_lo=ci_lo, ci_hi=ci_hi,
        pval=pval, qval=qval))

    nsig = int(np.sum(qval < FDR_ALPHA))
    print(f"FDR-significant cells (q<{FDR_ALPHA}): {nsig} / {L*K}")
    print(f"blind-spot cells (gap>{GAP_FLAG} & FDR-sig): {len(analysis['cells'])}")
    print(f"universal blind-spot classes: {analysis['universal']}")
    nA = sum(c["type"] == "A_invisible" for c in analysis["cells"])
    nB = sum(c["type"] == "B_confusion" for c in analysis["cells"])
    print(f"Type A (invisible): {nA}   Type B (confusion): {nB}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="chapman", help="chapman | ningbo | ptbxl | georgia")
    ap.add_argument("--arch", default=None)
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    main(a.dataset, a.arch, a.seed)
