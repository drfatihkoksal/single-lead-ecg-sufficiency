"""Combine the seed replicates of the Phase-3 grid into the paper's primary results.

For every cohort x architecture:
  - seed ensemble: test-set probabilities averaged over the complete seeds (a seed
    counts only when all lead specs finished); the primary estimate of each cell
  - per (lead spec, class) cell: AUPRC, AUROC, gap to the 12-lead ceiling, relative
    gap (gap / ceiling), record-level bootstrap 95% CI and one-sided p for the gap
    (evaluate.bootstrap_all), Benjamini-Hochberg q over the 12 x 11 single-lead cells,
    and the across-seed SD of the gap
  - sufficiency (primary framing): with margin d on the AUPRC gap, a cell is
    "sufficient" when the upper 95% CI bound of the gap is below d (the configuration
    retains the twelve-lead information within d), "loss" when the lower bound is above
    d, else "indeterminate"; d = 0.05 primary, 0.02 / 0.10 and relative margins as
    sensitivity; cells whose twelve-lead AUPRC is below LOW_CEILING are flagged, because
    a small gap there says little
  - blind-spot label (secondary, threshold 0.15 as in study 1) and Type A/B
  - classes a cohort cannot supply (config.UNAVAILABLE) are NaN throughout
Then, across the grid:
  - architecture concordance within each cohort, and cohort concordance within each
    architecture (Pearson r of gaps, Cohen kappa of the sufficiency classification;
    all classes / non-rhythm classes)

Outputs -> artifacts_v2/aggregate/
  cells.csv            one row per cohort x arch x spec x class (singles, devices, ALL)
  sufficiency_by_class.csv   per class: number of sufficient / loss / indeterminate leads
  concordance_arch.csv, concordance_cohort.csv, summary.json
  <cohort>/<arch>/ensemble.npz   matrices for figures

Run:  python -m src.aggregate_seeds            (recomputes only when the seed set changed)
      python -m src.aggregate_seeds --force
"""
import argparse
import itertools
import json
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import average_precision_score, roc_auc_score

from . import config as C
from .concordance import MORPH, cohen_kappa
from .evaluate import AUROC_INVIS, FDR_ALPHA, GAP_FLAG, bh_fdr, bootstrap_all
from .run_grid import SEEDS, specs_for
from .train import atomic_save

OUT = C.ART_V2 / "aggregate"
ARCHS = ["seresnet", "inceptiontime", "gbm"]
SPECS = specs_for(devices=True)                       # 12 singles, ALL, device sets
SINGLES = list(C.LEADS)
MORPH_IDX = [C.CLASSES.index(c) for c in MORPH]
MARGIN = 0.05                       # primary sufficiency margin on the absolute AUPRC gap
MARGINS_ABS = [0.02, 0.05, 0.10]
MARGINS_REL = [0.05, 0.10]          # on gap / ceiling
LOW_CEILING = 0.50
# diagnostic groups used for reporting
GROUP = {"SR": "rhythm", "SB": "rhythm", "ST": "rhythm", "AF": "rhythm", "AFL": "rhythm",
         "SVT": "rhythm", "LAD": "axis", "RBBB": "conduction", "LVH": "hypertrophy",
         "TWC": "repolarization", "STTC": "repolarization"}


def sufficiency(lo, hi, margin):
    """sufficient / loss / indeterminate from the gap CI and a margin (NaN -> 'na')."""
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "na"
    return "sufficient" if hi < margin else "loss" if lo > margin else "indeterminate"


def _complete_seeds(ds, arch):
    """Seeds whose every lead spec finished (gbm has no device specs)."""
    need = SINGLES + ["ALL"] if arch == "gbm" else SPECS
    out = []
    for s in SEEDS:
        m = C.ds_paths(ds, arch, s).metrics
        if all((m / f"pred_{sp}.npz").exists() for sp in need):
            out.append(s)
    return out, need


def _auprc(y, p):
    return average_precision_score(y, p) if y.sum() else np.nan


def _auroc(y, p):
    return roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan


def aggregate_one(args):
    ds, arch, force, out_dir = args
    seeds, specs = _complete_seeds(ds, arch)
    if not seeds:
        return None
    out = out_dir / ds / arch / "ensemble.npz"
    if out.exists() and not force:
        old = np.load(out, allow_pickle=True)
        if list(old["seeds"]) == seeds:
            return out
    unavailable = [C.CLASSES.index(c) for c in C.UNAVAILABLE.get(ds, [])]

    # per-seed predictions; the test set and its labels must be identical across seeds
    P = {sp: [] for sp in specs}
    y = None
    for s in seeds:
        m = C.ds_paths(ds, arch, s).metrics
        for sp in specs:
            d = np.load(m / f"pred_{sp}.npz")
            if y is None:
                y, idx = d["y"], d["test_idx"]
            assert (d["test_idx"] == idx).all() and (d["y"] == y).all(), f"{ds}/{arch}/{s}/{sp}"
            P[sp].append(d["p"])
    ens = {sp: (np.mean(P[sp], axis=0), y) for sp in specs}
    K = C.N_CLASSES

    auprc = np.array([[_auprc(y[:, j], ens[sp][0][:, j]) for j in range(K)] for sp in specs])
    auroc = np.array([[_auroc(y[:, j], ens[sp][0][:, j]) for j in range(K)] for sp in specs])
    ceil = auprc[specs.index("ALL")]
    gap = ceil[None, :] - auprc
    rel = gap / ceil[None, :]

    # across-seed spread of the gap (per seed, same cell definition)
    seed_gap = np.stack([
        np.array([[_auprc(y[:, j], P["ALL"][k][:, j]) - _auprc(y[:, j], P[sp][k][:, j])
                   for j in range(K)] for sp in specs])
        for k in range(len(seeds))])
    gap_sd = seed_gap.std(axis=0, ddof=1) if len(seeds) > 1 else np.full(gap.shape, np.nan)

    others = [sp for sp in specs if sp != "ALL"]
    lo, hi, pval = bootstrap_all(ens, others)                  # (len(others), K)
    ci_lo, ci_hi, pv = (np.full(gap.shape, np.nan) for _ in range(3))
    rows_o = [specs.index(sp) for sp in others]
    ci_lo[rows_o], ci_hi[rows_o], pv[rows_o] = lo, hi, pval

    # FDR over the 12 x 11 single-lead family (as in study 1); devices get their own family
    q = np.full(gap.shape, np.nan)
    for fam in (SINGLES, [sp for sp in others if sp not in SINGLES]):
        r = [specs.index(sp) for sp in fam]
        if r:
            q[r] = bh_fdr(pv[r].ravel()).reshape(len(r), K)

    for arr in (auprc, auroc, gap, rel, gap_sd, ci_lo, ci_hi, pv, q):
        arr[:, unavailable] = np.nan
    blind = (gap > GAP_FLAG) & (q < FDR_ALPHA)
    typ = np.where(~blind, "ok", np.where(auroc < AUROC_INVIS, "A_invisible", "B_confusion"))

    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_save(out, lambda t: np.savez(
        t, specs=np.array(specs), classes=np.array(C.CLASSES), seeds=np.array(seeds),
        auprc=auprc, auroc=auroc, ceil=ceil, gap=gap, rel_gap=rel, gap_sd=gap_sd,
        ci_lo=ci_lo, ci_hi=ci_hi, p=pv, q=q, blind=blind, type=typ,
        n_pos_test=y.sum(0)))
    print(f"[{ds}/{arch}] seeds={seeds} blind singles={int(blind[:12].sum())}", flush=True)
    return out


def _load(ds, arch, out_dir=OUT):
    f = out_dir / ds / arch / "ensemble.npz"
    return np.load(f, allow_pickle=True) if f.exists() else None


def gwet_ac1(a, b):
    """Gwet's AC1 for two binary ratings; unlike kappa it stays interpretable when one
    category is rare (the 'kappa paradox', e.g. almost no sufficient morphology cells)."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = np.mean(a == b)
    pi = (a.mean() + b.mean()) / 2
    pe = 2 * pi * (1 - pi)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def _concord(a, b, cols):
    """Over single-lead cells of the given classes: Pearson r of the gaps, and Cohen's
    kappa / agreement of the sufficiency classification (sufficient at MARGIN vs not);
    kappa of the study-1 blind-spot label is kept as a secondary column."""
    ga, gb = a["gap"][:12][:, cols].ravel(), b["gap"][:12][:, cols].ravel()
    ok = np.isfinite(ga) & np.isfinite(gb)
    sa = (a["ci_hi"][:12][:, cols].ravel() < MARGIN)[ok]
    sb = (b["ci_hi"][:12][:, cols].ravel() < MARGIN)[ok]
    ba, bb = a["blind"][:12][:, cols].ravel()[ok], b["blind"][:12][:, cols].ravel()[ok]
    return dict(n=int(ok.sum()), r=float(pearsonr(ga[ok], gb[ok])[0]),
                kappa=float(cohen_kappa(sa, sb)), ac1=float(gwet_ac1(sa, sb)),
                agree=float(np.mean(sa == sb)),
                kappa_blind015=float(cohen_kappa(ba, bb)))


def main(force=False, jobs=4, cohorts=None, archs=None, out_dir=OUT):
    """cohorts / archs / out_dir let sensitivity runs (e.g. the broad label definition,
    cohorts '<name>_broad') be aggregated without touching the primary outputs."""
    cohorts, archs = cohorts or C.COHORTS, archs or ARCHS
    OUT = out_dir
    OUT.mkdir(parents=True, exist_ok=True)
    grid = list(itertools.product(cohorts, archs))
    with Pool(jobs) as pool:
        pool.map(aggregate_one, [(d, a, force, OUT) for d, a in grid])
    E = {(d, a): _load(d, a, OUT) for d, a in grid}
    E = {k: v for k, v in E.items() if v is not None}

    rows = []
    for (d, a), e in E.items():
        for i, sp in enumerate(e["specs"]):
            for j, c in enumerate(e["classes"]):
                rows.append(dict(cohort=d, arch=a, spec=str(sp), cls=str(c),
                                 kind="single" if sp in SINGLES else "ceiling" if sp == "ALL" else "device",
                                 group="morphology" if c in MORPH else "rhythm",
                                 n_seeds=len(e["seeds"]), n_pos_test=int(e["n_pos_test"][j]),
                                 auprc=e["auprc"][i, j], auprc_ceiling=e["ceil"][j],
                                 auroc=e["auroc"][i, j], gap=e["gap"][i, j], rel_gap=e["rel_gap"][i, j],
                                 gap_ci_lo=e["ci_lo"][i, j], gap_ci_hi=e["ci_hi"][i, j],
                                 gap_seed_sd=e["gap_sd"][i, j], q=e["q"][i, j],
                                 blind=bool(e["blind"][i, j]), type=str(e["type"][i, j])))
    cells = pd.DataFrame(rows)
    cells["dx_group"] = cells.cls.map(GROUP)
    cells["low_ceiling"] = cells.auprc_ceiling < LOW_CEILING
    for m in MARGINS_ABS:
        cells[f"suff_abs{m:g}"] = [sufficiency(lo, hi, m) for lo, hi in zip(cells.gap_ci_lo, cells.gap_ci_hi)]
    for m in MARGINS_REL:
        cells[f"suff_rel{m:g}"] = [sufficiency(lo / c, hi / c, m) if c > 0 else "na" for lo, hi, c
                                   in zip(cells.gap_ci_lo, cells.gap_ci_hi, cells.auprc_ceiling)]
    cells["sufficiency"] = cells[f"suff_abs{MARGIN:g}"]
    atomic_save(OUT / "cells.csv", lambda t: cells.to_csv(t, index=False, float_format="%.4f"))

    # per class and cohort x arch: how many single leads are sufficient, and which
    sg = cells[cells.kind == "single"]
    tab = (sg.groupby(["cohort", "arch", "dx_group", "cls"])
             .apply(lambda g: pd.Series(dict(
                 n_sufficient=int((g.sufficiency == "sufficient").sum()),
                 n_loss=int((g.sufficiency == "loss").sum()),
                 n_indeterminate=int((g.sufficiency == "indeterminate").sum()),
                 sufficient_leads=",".join(g.spec[g.sufficiency == "sufficient"]),
                 ceiling=float(g.auprc_ceiling.iloc[0]),
                 low_ceiling=bool(g.low_ceiling.iloc[0]),
                 n_pos_test=int(g.n_pos_test.iloc[0]))), include_groups=False)
             .reset_index())
    atomic_save(OUT / "sufficiency_by_class.csv", lambda t: tab.to_csv(t, index=False, float_format="%.3f"))

    allc = list(range(C.N_CLASSES))
    ca = []
    for d in cohorts:
        for a, b in itertools.combinations([x for x in archs if (d, x) in E], 2):
            for name, cols in (("all", allc), ("morphology", MORPH_IDX)):
                ca.append(dict(cohort=d, arch_a=a, arch_b=b, classes=name, **_concord(E[d, a], E[d, b], cols)))
    cc = []
    for a in archs:
        for d1, d2 in itertools.combinations([d for d in cohorts if (d, a) in E], 2):
            for name, cols in (("all", allc), ("morphology", MORPH_IDX)):
                cc.append(dict(arch=a, cohort_a=d1, cohort_b=d2, classes=name, **_concord(E[d1, a], E[d2, a], cols)))
    ca, cc = pd.DataFrame(ca), pd.DataFrame(cc)
    atomic_save(OUT / "concordance_arch.csv", lambda t: ca.to_csv(t, index=False, float_format="%.3f"))
    atomic_save(OUT / "concordance_cohort.csv", lambda t: cc.to_csv(t, index=False, float_format="%.3f"))

    single = cells[cells.kind == "single"]
    summary = {f"{d}/{a}": dict(
        seeds=[int(s) for s in e["seeds"]],
        ceiling_macro_auprc=float(np.nanmean(e["ceil"])),
        blind_single_cells=int(e["blind"][:12].sum()),
        blind_rhythm=int(single[(single.cohort == d) & (single.arch == a) & single.blind & (single.group == "rhythm")].shape[0]),
        blind_morphology=int(single[(single.cohort == d) & (single.arch == a) & single.blind & (single.group == "morphology")].shape[0]),
        median_gap_seed_sd=float(np.nanmedian(e["gap_sd"][:12])) if len(e["seeds"]) > 1 else None)
        for (d, a), e in E.items()}
    atomic_save(OUT / "summary.json", lambda t: t.write_text(json.dumps(summary, indent=2)))

    print("\n== summary ==")
    for k, v in summary.items():
        print(f"{k:28s} seeds={len(v['seeds'])} ceil={v['ceiling_macro_auprc']:.3f} "
              f"blind={v['blind_single_cells']} (rhythm {v['blind_rhythm']}, morph {v['blind_morphology']}) "
              f"seedSD={v['median_gap_seed_sd']}")
    if len(ca):
        print("\n== architecture concordance (morphology) ==")
        print(ca[ca.classes == "morphology"].to_string(index=False, float_format="%.2f"))
    if len(cc):
        print("\n== cohort concordance (morphology) ==")
        print(cc[cc.classes == "morphology"].to_string(index=False, float_format="%.2f"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--cohorts", nargs="+", default=None)
    ap.add_argument("--archs", nargs="+", default=None)
    ap.add_argument("--out", default=None, help="output dir (default artifacts_v2/aggregate)")
    a = ap.parse_args()
    main(a.force, a.jobs, a.cohorts, a.archs, C.ROOT / a.out if a.out else OUT)
