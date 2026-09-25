"""Additional analyses requested in the internal pre-submission review.

From the seed-ensemble test predictions (rebuilt from the per-seed pred_*.npz files):
  paired     paired bootstrap CI of gap(lead a) - gap(lead b) for selected lead pairs and for the
             information-equivalent reduced sets (I+II vs six limb leads; I, II, V2 vs I, II, III, V2)
  best       bootstrap probability that each single lead has the smallest gap, and the optimism of
             choosing the best lead on the test set (select on one random half, evaluate on the other)
From cells.csv:
  concord    within-class agreement of the lead ranking (Kendall's W across cohorts, Spearman between
             models) and a variance decomposition of the single-lead gaps
  normalized sufficiency on the prevalence-normalized AUPRC scale, (AUPRC - pi) / (1 - pi)
  physiology probability, under a uniform null, that the best lead falls in the criterion-defined lead set
From the signals:
  labels     heart rate from R peaks (lead II) against the sinus bradycardia / tachycardia labels, and
             co-coding of sinus rhythm with other diagnoses

Outputs -> artifacts_v2/review/*.csv and summary.json

Run:  python -m src.review_analyses [--jobs 24]
"""
import argparse
import json
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from . import config as C
from .aggregate_seeds import _complete_seeds

OUT = C.ART_V2 / "review"
AGG = C.ART_V2 / "aggregate"
COHORTS = ["chapman", "ningbo", "georgia", "ptbxl_snomed"]
ARCHS = ["seresnet", "inceptiontime", "gbm"]
LEADS = list(C.LEADS)
N_BOOT, N_SPLIT = 1000, 200
PAIRS = [("I", "II"), ("I", "aVF"), ("I+II", "LIMB6"), ("CH3", "CH4")]
# lead sets named by the diagnostic criteria (not by the results): axis from I, II and aVF;
# RBBB from the right precordial leads and the terminal S wave in I and V6; LVH voltage criteria
# (Sokolow-Lyon: V1, V5, V6; Cornell: aVL, V3)
CRITERION_LEADS = {"LAD": ["I", "II", "aVF"], "RBBB": ["V1", "V2", "I", "V6"],
                   "LVH": ["V1", "V5", "V6", "aVL", "V3"]}


def ap(y, p):
    """Average precision identical to sklearn's (ties handled by distinct thresholds)."""
    o = np.argsort(-p, kind="mergesort")
    ys, ps = y[o], p[o]
    last = np.r_[np.diff(ps) != 0, True]
    tp = np.cumsum(ys)[last]
    if tp[-1] == 0:
        return np.nan
    fp = np.arange(1, len(ys) + 1)[last] - tp
    prec, rec = tp / (tp + fp), tp / tp[-1]
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def load_ens(ds, arch):
    seeds, specs = _complete_seeds(ds, arch)
    P, y = {}, None
    for sp in specs:
        ps = []
        for s in seeds:
            d = np.load(C.ds_paths(ds, arch, s).metrics / f"pred_{sp}.npz")
            y = d["y"] if y is None else y
            ps.append(d["p"])
        P[sp] = np.mean(ps, axis=0)
    return P, y


def job(args):
    ds, arch, j = args
    P, y = load_ens(ds, arch)
    cl = C.CLASSES[j]
    yj = y[:, j]
    if cl in C.UNAVAILABLE.get(ds, []) or yj.sum() == 0:
        return None
    n = len(yj)
    specs = [sp for sp in LEADS + ["I+II", "LIMB6", "CH3", "CH4"] if sp in P]
    rng = np.random.default_rng(C.SEED + j)
    gaps = np.full((N_BOOT, len(specs)), np.nan)
    for b in range(N_BOOT):
        idx = rng.integers(0, n, n)
        yy = yj[idx]
        if yy.sum() == 0:
            continue
        c = ap(yy, P["ALL"][idx, j])
        gaps[b] = [c - ap(yy, P[sp][idx, j]) for sp in specs]
    ok = ~np.isnan(gaps).any(1)
    gaps = gaps[ok]
    col = {sp: i for i, sp in enumerate(specs)}
    res = {"cohort": ds, "arch": arch, "cls": cl, "n_pos": int(yj.sum()), "n": n}
    for a, b in PAIRS:
        if a in col and b in col:
            d = gaps[:, col[a]] - gaps[:, col[b]]
            res[f"d_{a}_{b}"] = float(np.mean(d))
            res[f"d_{a}_{b}_lo"], res[f"d_{a}_{b}_hi"] = (float(v) for v in np.percentile(d, [2.5, 97.5]))
    single = [col[ld] for ld in LEADS]
    best = np.argmin(gaps[:, single], axis=1)
    pbest = np.bincount(best, minlength=12) / len(best)
    for i, ld in enumerate(LEADS):
        res[f"pbest_{ld}"] = float(pbest[i])
    # optimism of test-set selection: choose on half A, evaluate on half B
    opt, sel_same = [], []
    pos, neg = np.where(yj == 1)[0], np.where(yj == 0)[0]
    for _ in range(N_SPLIT):
        a_idx = np.r_[rng.permutation(pos)[: len(pos) // 2], rng.permutation(neg)[: len(neg) // 2]]
        mask = np.zeros(n, bool); mask[a_idx] = True
        if yj[mask].sum() == 0 or yj[~mask].sum() == 0:
            continue
        ga = np.array([ap(yj[mask], P["ALL"][mask, j]) - ap(yj[mask], P[ld][mask, j]) for ld in LEADS])
        k = int(np.nanargmin(ga))
        gb = ap(yj[~mask], P["ALL"][~mask, j]) - ap(yj[~mask], P[LEADS[k]][~mask, j])
        opt.append(gb - ga[k])
    res["optimism_mean"] = float(np.mean(opt)) if opt else np.nan
    return res


def run_bootstrap(jobs):
    todo = [(ds, a, j) for ds in COHORTS for a in ARCHS for j in range(C.N_CLASSES)
            if _complete_seeds(ds, a)[0]]
    with Pool(jobs) as pool:
        rows = [r for r in pool.map(job, todo, chunksize=1) if r]
    return pd.DataFrame(rows)


def concordance(cells):
    s = cells[(cells.kind == "single") & cells.gap.notna()]
    dl = s[s.arch != "gbm"]
    rows = []
    for arch in ("seresnet", "inceptiontime"):
        for cl in C.CLASSES:
            m = dl[(dl.arch == arch) & (dl.cls == cl)].pivot_table(index="spec", columns="cohort", values="gap")
            m = m.dropna(axis=1)
            if m.shape[1] < 2:
                continue
            r = m.rank().values                       # Kendall's W over cohorts (raters) and leads (items)
            k, nn = r.shape[1], r.shape[0]
            S = ((r.sum(1) - r.sum(1).mean()) ** 2).sum()
            W = 12 * S / (k ** 2 * (nn ** 3 - nn))
            rows.append({"what": "kendall_w_cohorts", "arch": arch, "cls": cl, "value": W, "k": k})
    for ds in COHORTS:
        for cl in C.CLASSES:
            m = dl[(dl.cohort == ds) & (dl.cls == cl)].pivot_table(index="spec", columns="arch", values="gap").dropna()
            if len(m) == 12 and m.shape[1] == 2:
                rows.append({"what": "spearman_archs", "arch": ds, "cls": cl,
                             "value": spearmanr(m.iloc[:, 0], m.iloc[:, 1])[0], "k": 2})
    conc = pd.DataFrame(rows)

    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm
    d = dl.rename(columns={"spec": "lead"})[["gap", "cls", "lead", "cohort", "arch"]]
    fit = smf.ols("gap ~ cls * lead + cohort + arch", data=d).fit()     # string columns are categorical
    a = anova_lm(fit, typ=2)
    a["share"] = a["sum_sq"] / a["sum_sq"].sum()
    return conc, a


def normalized(cells):
    s = cells[cells.kind.isin(["single", "device"]) & (cells.arch != "gbm") & cells.gap.notna()].copy()
    n_test = {"chapman": 1535, "ningbo": 5225, "georgia": 1544, "ptbxl_snomed": 2203}
    pi = s.n_pos_test / s.cohort.map(n_test)
    for c in ("gap", "gap_ci_lo", "gap_ci_hi"):
        s[c + "_norm"] = s[c] / (1 - pi)
    s["suff_norm"] = np.where(s.gap_ci_hi_norm < 0.05, "sufficient",
                              np.where(s.gap_ci_lo_norm > 0.05, "loss", "indeterminate"))
    single = s[s.kind == "single"]
    tab = pd.crosstab(single.dx_group, single.suff_norm)
    prim = pd.crosstab(single.dx_group, single.sufficiency)
    return tab, prim


def physiology(cells):
    s = cells[(cells.kind == "single") & (cells.arch == "seresnet") & cells.gap.notna()]
    best = s.loc[s.groupby(["cohort", "cls"]).gap.idxmin()]
    rows, p_all = [], 1.0
    from scipy.stats import binom
    for cl, leads in CRITERION_LEADS.items():
        b = best[best.cls == cl]
        hits = int(b.spec.isin(leads).sum())
        p0 = len(leads) / 12
        p = float(binom.sf(hits - 1, len(b), p0))
        p_all *= p
        rows.append({"cls": cl, "criterion_leads": " ".join(leads), "best_leads": " ".join(b.spec),
                     "hits": hits, "cohorts": len(b), "p_uniform": p})
    return pd.DataFrame(rows), p_all


def label_checks():
    import neurokit2 as nk
    rows, co = [], []
    for ds in COHORTS:
        P = C.ds_paths(ds, "any")
        sig = np.load(P.cache / "signals.npy", mmap_mode="r")
        y = np.load(P.cache / "labels.npy")
        te = np.load(P.splits / "split.npz")["test"]
        hr = np.full(len(te), np.nan)
        for i, r in enumerate(te):
            try:
                x = nk.ecg_clean(np.asarray(sig[r, 1]), sampling_rate=C.FS)
                pk = nk.ecg_peaks(x, sampling_rate=C.FS)[1]["ECG_R_Peaks"]
                if len(pk) >= 4:
                    hr[i] = 60 * C.FS / np.median(np.diff(pk))
            except Exception:
                pass
        yt = y[te]
        k = {c: C.CLASSES.index(c) for c in C.CLASSES}
        irregular = (yt[:, k["AF"]] + yt[:, k["AFL"]]) > 0
        ok = ~np.isnan(hr)
        for cl, cond in (("SB", hr < 60), ("ST", hr > 100)):
            lab = yt[:, k[cl]] == 1
            rows.append({"cohort": ds, "cls": cl, "n_label": int(lab.sum()),
                         "label_agrees_with_hr": float(np.mean(cond[lab & ok])) if (lab & ok).any() else np.nan,
                         "n_hr_rule": int((cond & ok & ~irregular).sum()),
                         "hr_rule_labelled": float(np.mean(lab[cond & ok & ~irregular])) if (cond & ok & ~irregular).any() else np.nan})
        # sinus rhythm co-coding: is SR also coded when another (non-rhythm) diagnosis is present?
        morph = yt[:, [k[c] for c in ("LAD", "RBBB", "LVH", "TWC", "STTC")]].sum(1) > 0
        sinus_rate = ok & (hr >= 60) & (hr <= 100) & ~irregular
        co.append({"cohort": ds,
                   "sr_given_morph": float(yt[sinus_rate & morph, k["SR"]].mean()),
                   "sr_given_no_morph": float(yt[sinus_rate & ~morph, k["SR"]].mean()),
                   "n_morph": int((sinus_rate & morph).sum()), "n_no_morph": int((sinus_rate & ~morph).sum()),
                   "hr_detected": float(ok.mean())})
    return pd.DataFrame(rows), pd.DataFrame(co)


def main(jobs):
    OUT.mkdir(parents=True, exist_ok=True)
    cells = pd.read_csv(AGG / "cells.csv")
    if not (OUT / "bootstrap.csv").exists():
        run_bootstrap(jobs).to_csv(OUT / "bootstrap.csv", index=False)
    conc, anova = concordance(cells)
    conc.to_csv(OUT / "concordance_within_class.csv", index=False)
    anova.to_csv(OUT / "variance_decomposition.csv")
    norm, prim = normalized(cells)
    norm.to_csv(OUT / "normalized_sufficiency.csv")
    phys, p_all = physiology(cells)
    phys.to_csv(OUT / "physiology_null.csv", index=False)
    lab, co = label_checks()
    lab.to_csv(OUT / "label_heart_rate.csv", index=False)
    co.to_csv(OUT / "label_sr_cocoding.csv", index=False)
    summary = {"physiology_joint_p": p_all}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print("done", OUT)


if __name__ == "__main__":
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--jobs", type=int, default=24)
    main(ap_.parse_args().jobs)
