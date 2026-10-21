"""Analyses requested by the CIBM reviewers (CIBM-S-26-08400), for the BSPC submission.

  axis     frontal QRS axis computed from limb leads I and II (net QRS area of the median beat;
           aVF = II - I/2, axis = atan2(2 aVF / sqrt 3, I)) against the left axis deviation label in
           every cohort; for PTB-XL also against the heart_axis field, its missingness by recording
           device, and the PTB-XL LAD gaps re-evaluated on the records whose heart_axis is coded
  recl     sufficiency on the primary two-level interval (seeds and test records resampled;
           aggregate_seeds.two_level_bootstrap) against the record-only interval of the earlier version
  prereg   evaluation of the preregistered predictions for the limb + chest lead sets
           (PREREGISTRATION_lateral_sets.md, P1-P7)
  afop     atrial fibrillation operating points: sensitivity at fixed specificity (0.90, 0.95) for
           leads I, II, aVF and the twelve-lead model, and the lead I - lead II differences
           (AUPRC gap and sensitivity) with two-level bootstrap intervals

Outputs -> artifacts_v2/review2/

Run:  python -m src.review2_analyses [--only axis recl afop prereg] [--jobs 24]
"""
import argparse
import itertools
import json
from multiprocessing import Pool

import numpy as np
import pandas as pd

from . import config as C
from .aggregate_seeds import MARGIN, _complete_seeds, sufficiency, two_level_bootstrap
from .review_analyses import ap

OUT = C.ART_V2 / "review2"
AGG = C.ART_V2 / "aggregate"
COHORTS = ["chapman", "ningbo", "georgia", "ptbxl_snomed"]
DL_ARCHS = ["seresnet", "inceptiontime"]
ARCHS = DL_ARCHS + ["gbm"]
N_BOOT = 1000
LAD_DEG = -30.0                     # left axis deviation: frontal QRS axis below -30 degrees
SPECS_FIX = [0.90, 0.95]            # fixed specificities for the AF operating points


# ---- frontal QRS axis -----------------------------------------------------------------------
def qrs_axis(sig, fs=C.FS):
    """Frontal QRS axis (degrees) of one recording from leads I and II, or NaN.

    Beats are found on the limb-lead vector magnitude (lead II alone is near isoelectric when
    the axis is close to -30 degrees); the median beat of I and aVF is baseline-corrected on the
    PR segment, the QRS is the contiguous run around the peak where the magnitude exceeds 15% of
    its maximum, and the axis follows from the net QRS areas."""
    import neurokit2 as nk
    from scipy.signal import find_peaks
    try:
        I = nk.ecg_clean(np.asarray(sig[0], float), sampling_rate=fs)
        II = nk.ecg_clean(np.asarray(sig[1], float), sampling_rate=fs)
    except Exception:
        return np.nan
    aVF = II - I / 2
    y = 2 * aVF / np.sqrt(3)                      # vertical component of the frontal vector
    mag = np.hypot(I, y)
    if not np.isfinite(mag).all() or mag.max() <= 0:
        return np.nan
    pk, _ = find_peaks(mag, distance=int(0.3 * fs), height=0.4 * np.percentile(mag, 99))
    w0, w1 = int(0.20 * fs), int(0.25 * fs)
    pk = pk[(pk > w0) & (pk < len(mag) - w1)]
    if len(pk) < 3:
        return np.nan
    bI = np.median([I[p - w0:p + w1] for p in pk], axis=0)
    bY = np.median([y[p - w0:p + w1] for p in pk], axis=0)
    base = slice(w0 - int(0.12 * fs), w0 - int(0.07 * fs))       # PR segment, 120-70 ms before the peak
    bI, bY = bI - np.median(bI[base]), bY - np.median(bY[base])
    m = np.hypot(bI, bY)
    thr = 0.15 * m[w0]
    on = w0
    while on > base.stop and m[on - 1] > thr:
        on -= 1
    off = w0
    while off < len(m) - 1 and m[off + 1] > thr:
        off += 1
    if (off - on) / fs > 0.22:                    # no clear QRS boundary
        return np.nan
    return float(np.degrees(np.arctan2(bY[on:off + 1].sum(), bI[on:off + 1].sum())))


def _axis_rows(args):
    ds, rows = args
    sig = np.load(C.ds_paths(ds, "any").cache / "signals.npy", mmap_mode="r")
    return [qrs_axis(sig[r]) for r in rows]


def compute_axes(ds, rows, jobs):
    chunks = np.array_split(np.asarray(rows), jobs * 4)
    with Pool(jobs) as pool:
        out = pool.map(_axis_rows, [(ds, c) for c in chunks])
    return np.concatenate([np.asarray(o, float) for o in out])


def _auc(y, s):
    from sklearn.metrics import roc_auc_score
    ok = np.isfinite(s)
    return roc_auc_score(y[ok], s[ok]) if 0 < y[ok].sum() < ok.sum() else np.nan


def axis_analysis(jobs):
    j = C.CLASSES.index("LAD")
    agree, rec = [], []
    for ds in COHORTS:
        P = C.ds_paths(ds, "any")
        y = np.load(P.cache / "labels.npy")[:, j]
        te = np.load(P.splits / "split.npz")["test"]
        rows = np.arange(len(y)) if ds == "ptbxl_snomed" else te    # PTB-XL: whole cohort for the field table
        ax = compute_axes(ds, rows, jobs)
        df = pd.DataFrame({"cohort": ds, "row": rows, "axis": ax, "lad_label": y[rows],
                           "test": np.isin(rows, te)})
        rec.append(df)
        t = df[df.test]
        ok = t.axis.notna()
        rule = t.axis < LAD_DEG
        lab = t.lad_label == 1
        agree.append({"cohort": ds, "n_test": len(t), "axis_computed": float(ok.mean()),
                      "n_label": int(lab.sum()), "n_rule": int((rule & ok).sum()),
                      "auroc_axis": _auc(lab.values.astype(int), -t.axis.values),
                      "auprc_axis": ap(lab[ok].values.astype(int), -t.axis[ok].values),
                      "label_given_rule": float(lab[rule & ok].mean()),        # PPV of the -30 rule
                      "rule_given_label": float(rule[lab & ok].mean()),        # labelled cases with axis < -30
                      "label_given_no_rule": float(lab[~rule & ok].mean()),
                      "kappa_rule_label": _kappa(rule[ok].values, lab[ok].values)})
    rec = pd.concat(rec, ignore_index=True)
    agree = pd.DataFrame(agree)

    # PTB-XL: computed axis by heart_axis category, missingness by device, and the gaps on coded records
    db = pd.read_csv(C.PTBXL_ROOT / "ptbxl_database.csv")
    ids = np.load(C.ds_paths("ptbxl_snomed", "any").cache / "record_ids.npy", allow_pickle=True)
    p = rec[rec.cohort == "ptbxl_snomed"].copy()
    ecg_id = np.array([int(r[2:]) for r in ids])                    # HRnnnnn = ecg_id
    meta = db.set_index("ecg_id").loc[ecg_id[p.row.values]]
    p["heart_axis"] = meta.heart_axis.fillna("missing").values
    p["device"] = meta.device.str.split().str[0].values           # device family (CS..., AT-...)
    by_field = (p.groupby("heart_axis")
                 .apply(lambda g: pd.Series(dict(n=len(g), median_axis=g.axis.median(),
                                                 q1=g.axis.quantile(.25), q3=g.axis.quantile(.75),
                                                 below_m30=float((g.axis < LAD_DEG).mean()),
                                                 below_0=float((g.axis < 0).mean()),
                                                 lad_label=float(g.lad_label.mean()))), include_groups=False)
                 .reset_index())
    rule = p.axis < LAD_DEG
    by_dev = (p.assign(rule=rule, missing=p.heart_axis == "missing")
               .groupby("device")
               .apply(lambda g: pd.Series(dict(n=len(g), field_missing=float(g.missing.mean()),
                                               axis_below_m30=float(g.rule.mean()),
                                               lad_label=float(g.lad_label.mean()),
                                               label_given_rule=float(g.lad_label[g.rule].mean()) if g.rule.any() else np.nan)),
                      include_groups=False)
               .reset_index().sort_values("n", ascending=False))
    gaps = coded_subset_gaps(p)
    return agree, rec, by_field, by_dev, gaps


def _kappa(a, b):
    from sklearn.metrics import cohen_kappa_score
    return float(cohen_kappa_score(a.astype(int), b.astype(int)))


def coded_subset_gaps(p):
    """PTB-XL LAD gaps on all test records and on those whose heart_axis is coded (missing heart_axis
    was read as 'no left axis deviation'); seed ensembles, two-level bootstrap 95% CI."""
    j = C.CLASSES.index("LAD")
    t = p[p.test].set_index("row")
    rows = []
    for arch in ARCHS:
        if not _complete_seeds("ptbxl_snomed", arch)[0]:
            continue
        P, y, idx = _seed_preds("ptbxl_snomed", arch)
        coded = t.loc[idx, "heart_axis"].values != "missing"
        specs = [s for s in ["I", "II", "aVF", "aVL", "III", "I+II", "LIMB6", "CH3", "CH4"] + C.LATERAL_SPECS
                 if s in P]
        for name, m in (("all", np.ones(len(idx), bool)), ("coded", coded)):
            Pm = {s: P[s][:, m] for s in specs + ["ALL"]}
            lo, hi, _ = two_level_bootstrap(Pm, y[m], specs)
            yy = y[m, j]
            c0 = ap(yy, Pm["ALL"][:, :, j].mean(0))
            for k, s in enumerate(specs):
                a = ap(yy, Pm[s][:, :, j].mean(0))
                rows.append({"arch": arch, "subset": name, "n": int(m.sum()), "n_pos": int(yy.sum()),
                             "spec": s, "auprc_12": c0, "auprc": a, "gap": c0 - a, "gap_lo": lo[k, j],
                             "gap_hi": hi[k, j], "sufficiency": sufficiency(lo[k, j], hi[k, j], MARGIN)})
    return pd.DataFrame(rows)


def _seed_preds(ds, arch):
    seeds, specs = _complete_seeds(ds, arch)
    P, y, idx = {}, None, None
    for sp in specs:
        ps = []
        for s in seeds:
            d = np.load(C.ds_paths(ds, arch, s).metrics / f"pred_{sp}.npz")
            if y is None:
                y, idx = d["y"], d["test_idx"]
            ps.append(d["p"])
        P[sp] = np.stack(ps)                       # (seeds, n, K)
    return P, y, idx


# ---- two-level against record-only intervals --------------------------------------------------
def reclassification():
    """How the sufficiency classification changes when training variability (seeds) is added to the
    record-level bootstrap; CNN cells only (the feature model has one seed)."""
    c = pd.read_csv(AGG / "cells.csv")
    s = c[(c.kind != "ceiling") & (c.arch != "gbm") & (c.sufficiency != "na")].copy()
    s["width"] = s.gap_ci_hi - s.gap_ci_lo
    s["width_rec"] = s.gap_ci_hi_rec - s.gap_ci_lo_rec
    return (s.groupby(["arch", "kind"])
              .apply(lambda g: pd.Series(dict(
                  n=len(g),
                  same=int((g.sufficiency == g.sufficiency_rec).sum()),
                  suff_rec_to_indet=int(((g.sufficiency_rec == "sufficient") & (g.sufficiency == "indeterminate")).sum()),
                  loss_rec_to_indet=int(((g.sufficiency_rec == "loss") & (g.sufficiency == "indeterminate")).sum()),
                  indet_rec_to_other=int(((g.sufficiency_rec == "indeterminate") & (g.sufficiency != "indeterminate")).sum()),
                  n_sufficient_rec=int((g.sufficiency_rec == "sufficient").sum()),
                  n_sufficient=int((g.sufficiency == "sufficient").sum()),
                  median_width_rec=float(g.width_rec.median()), median_width=float(g.width.median()))),
                  include_groups=False)
              .reset_index())


# ---- preregistered predictions for the limb + chest lead sets ---------------------------------
NETS = ["seresnet", "inceptiontime"]


def prereg():
    """P1-P7 of PREREGISTRATION_lateral_sets.md.  Reading of the claims: a positive claim (sufficient,
    smaller gap) holds in a cohort only if it holds for both networks; a negative claim (not
    sufficient, not smaller) only if it holds for neither network.  Classes flagged in a cohort
    (dagger or double dagger) are not counted."""
    c = pd.read_csv(AGG / "cells.csv")
    b = pd.read_csv(C.ART_V2 / "review" / "bootstrap.csv")
    ce = c[c.kind == "ceiling"]
    flagged = {(r.cohort, r.cls) for r in ce.itertuples()
               if C.flag(r.cohort, r.cls, r.low_ceiling, r.n_pos_test) != ""}
    suff = {(r.cohort, r.arch, r.spec, r.cls): r.sufficiency for r in c.itertuples()}

    def ok(ds, cl):
        return (ds, cl) not in flagged and cl not in C.UNAVAILABLE.get(ds, [])

    def is_suff(ds, spec, cl, net):
        return suff.get((ds, net, spec, cl)) == "sufficient"

    def smaller(ds, spec, cl, net):                 # gap(spec) - gap(I+II) entirely below 0
        r = b[(b.cohort == ds) & (b.arch == net) & (b.cls == cl)]
        return bool(len(r)) and float(r[f"d_{spec}_I+II_hi"].iloc[0]) < 0

    def both(f, *a):
        return all(f(*a, n) for n in NETS)

    def neither(f, *a):
        return not any(f(*a, n) for n in NETS)

    rows = []

    def add(pid, claim, cohorts, test, rule, detail_f):
        cs = [ds for ds in cohorts if ok(ds, claim[1])]
        hits = [ds for ds in cs if test(ds)]
        confirmed = rule(len(hits), len(cs))
        rows.append({"prediction": pid, "class": claim[1], "claim": claim[0],
                     "cohorts_evaluated": ", ".join(cs), "cohorts_meeting": ", ".join(hits) or "none",
                     "confirmed": "yes" if confirmed else "no",
                     "detail": "; ".join(f"{ds}: {detail_f(ds)}" for ds in cs)})

    def sdet(spec, cl):
        return lambda ds: "/".join(str(suff.get((ds, n, spec, cl), "na"))[:5] for n in NETS)

    def ddet(spec, cl):
        def f(ds):
            out = []
            for n in NETS:
                r = b[(b.cohort == ds) & (b.arch == n) & (b.cls == cl)]
                k = f"d_{spec}_I+II"
                out.append(f"{r[k].iloc[0]:+.3f} [{r[k + '_lo'].iloc[0]:+.3f}, {r[k + '_hi'].iloc[0]:+.3f}]"
                           if len(r) else "na")
            return " / ".join(out)
        return f

    ALL4 = COHORTS
    ge = lambda k: (lambda h, n: h >= k)
    allc = lambda h, n: h == n and n > 0
    # P1 right bundle branch block
    add("P1a", ("I+V1 sufficient (both networks) in >= 3 cohorts", "RBBB"), ALL4,
        lambda ds: both(is_suff, ds, "I+V1", "RBBB"), ge(3), sdet("I+V1", "RBBB"))
    for sp in ("I+V5", "II+V5"):
        add("P1b", (f"{sp} not sufficient (neither network) in at least two of the four cohorts (= sufficient in at most two)", "RBBB"), ALL4,
            lambda ds, sp=sp: neither(is_suff, ds, sp, "RBBB"), lambda h, n: h >= n - 2, sdet(sp, "RBBB"))
    # P2 left ventricular hypertrophy, P3 T-wave change
    for pid, cl in (("P2", "LVH"), ("P3", "TWC")):
        for sp in ("I+V5", "II+V5"):
            add(f"{pid}a", (f"{sp} gap smaller than I+II (both networks) in all cohorts", cl), ALL4,
                lambda ds, sp=sp, cl=cl: both(smaller, ds, sp, cl), allc, ddet(sp, cl))
            add(f"{pid}b", (f"{sp} sufficient (both networks) in Chapman and Ningbo", cl), ["chapman", "ningbo"],
                lambda ds, sp=sp, cl=cl: both(is_suff, ds, sp, cl), allc, sdet(sp, cl))
            add(f"{pid}c", (f"{sp} not sufficient (neither network) in Georgia and PTB-XL", cl), ["georgia", "ptbxl_snomed"],
                lambda ds, sp=sp, cl=cl: neither(is_suff, ds, sp, cl), allc, sdet(sp, cl))
    add("P2d", ("I+V1 not sufficient (neither network) in Georgia and PTB-XL", "LVH"), ["georgia", "ptbxl_snomed"],
        lambda ds: neither(is_suff, ds, "I+V1", "LVH"), allc, sdet("I+V1", "LVH"))
    add("P3d", ("I+V1 gap not smaller than I+II (neither network) in any cohort", "TWC"), ALL4,
        lambda ds: neither(smaller, ds, "I+V1", "TWC"), allc, ddet("I+V1", "TWC"))
    # P4 left axis deviation
    add("P4a", ("II+V5 sufficient (both networks) in >= 2 of Chapman, Ningbo, Georgia", "LAD"),
        ["chapman", "ningbo", "georgia"], lambda ds: both(is_suff, ds, "II+V5", "LAD"), ge(2), sdet("II+V5", "LAD"))
    for sp in ("I+V5", "I+V1"):
        add("P4b", (f"{sp} not sufficient (neither network) in any cohort", "LAD"), ALL4,
            lambda ds, sp=sp: neither(is_suff, ds, sp, "LAD"), allc, sdet(sp, "LAD"))
    g = pd.read_csv(OUT / "axis_ptbxl_coded_gaps.csv")
    g = g[(g.subset == "coded") & (g.spec == "II+V5") & g.arch.isin(NETS)]
    rows.append({"prediction": "P4c", "class": "LAD", "claim": "II+V5 sufficient (both networks) in PTB-XL records with a coded heart_axis",
                 "cohorts_evaluated": "ptbxl_snomed (coded heart_axis)",
                 "cohorts_meeting": "ptbxl_snomed" if (len(g) == 2 and (g.sufficiency == "sufficient").all()) else "none",
                 "confirmed": "yes" if (len(g) == 2 and (g.sufficiency == "sufficient").all()) else "no",
                 "detail": "; ".join(f"{r.arch}: {r.gap:+.3f} [{r.gap_lo:+.3f}, {r.gap_hi:+.3f}] {r.sufficiency}" for r in g.itertuples())})
    # P5 atrial fibrillation
    for sp in ("I+V1", "II+V5"):
        add("P5", (f"{sp} sufficient (both networks) in >= 2 of 3 cohorts", "AF"), ALL4,
            lambda ds, sp=sp: both(is_suff, ds, sp, "AF"), ge(2), sdet(sp, "AF"))
    # P6 atrial flutter
    add("P6a", ("I+V1 gap smaller than I+II (both networks) in Chapman", "AFL"), ["chapman"],
        lambda ds: both(smaller, ds, "I+V1", "AFL"), allc, ddet("I+V1", "AFL"))
    add("P6b", ("I+V1 not sufficient (neither network) in any cohort", "AFL"), ALL4,
        lambda ds: neither(is_suff, ds, "I+V1", "AFL"), allc, sdet("I+V1", "AFL"))
    # P7 rate-defined rhythms
    for cl in ("SR", "SB", "ST"):
        for sp in C.LATERAL_SPECS:
            add("P7", (f"{sp} sufficient (both networks) in every cohort without a flag", cl), ALL4,
                lambda ds, sp=sp, cl=cl: both(is_suff, ds, sp, cl), allc, sdet(sp, cl))
    return pd.DataFrame(rows)


# ---- atrial fibrillation operating points ---------------------------------------------------
def sens_at_spec(y, p, spec):
    """Sensitivity at the threshold that gives at least the requested specificity."""
    neg = np.sort(p[y == 0])
    if len(neg) == 0 or y.sum() == 0:
        return np.nan
    thr = neg[min(int(np.ceil(spec * len(neg))), len(neg) - 1)]
    return float(np.mean(p[y == 1] > thr))


def afop_job(args):
    ds, arch = args
    j = C.CLASSES.index("AF")
    P, y, _ = _seed_preds(ds, arch)
    yj = y[:, j]
    S, n = P["ALL"].shape[:2]
    leads = ["ALL", "I", "II", "aVF"]
    ens = {ld: P[ld].mean(0)[:, j] for ld in leads}

    def metrics(yy, p):
        m = {"auprc": {ld: ap(yy, p[ld]) for ld in leads}}
        for sp in SPECS_FIX:
            m[sp] = {ld: sens_at_spec(yy, p[ld], sp) for ld in leads}
        return m

    point = metrics(yj, ens)
    rng = np.random.default_rng(C.SEED)
    B = []
    for _ in range(N_BOOT):
        r = rng.integers(0, n, n)
        if yj[r].sum() == 0:
            continue
        p = {ld: P[ld][rng.integers(0, S, S)][:, r, j].mean(0) for ld in leads}
        B.append(metrics(yj[r], p))
    rows = []

    def ci(f):
        v = np.array([f(b) for b in B], float)
        return np.nanpercentile(v, [2.5, 97.5])

    for ld in leads:
        row = {"cohort": ds, "arch": arch, "lead": ld, "n_pos": int(yj.sum()), "n": n,
               "auprc": point["auprc"][ld]}
        row["auprc_lo"], row["auprc_hi"] = ci(lambda b: b["auprc"][ld])
        if ld != "ALL":
            row["gap"] = point["auprc"]["ALL"] - point["auprc"][ld]
            row["gap_lo"], row["gap_hi"] = ci(lambda b: b["auprc"]["ALL"] - b["auprc"][ld])
        for sp in SPECS_FIX:
            k = f"sens_at_spec{int(sp * 100)}"
            row[k] = point[sp][ld]
            row[k + "_lo"], row[k + "_hi"] = ci(lambda b: b[sp][ld])
        rows.append(row)
    # lead I minus lead II (and minus aVF): positive = lead I worse
    for other in ("II", "aVF"):
        row = {"cohort": ds, "arch": arch, "lead": f"I-{other}", "n_pos": int(yj.sum()), "n": n}
        row["gap"] = point["auprc"][other] - point["auprc"]["I"]
        row["gap_lo"], row["gap_hi"] = ci(lambda b: b["auprc"][other] - b["auprc"]["I"])
        for sp in SPECS_FIX:
            k = f"sens_at_spec{int(sp * 100)}"
            row[k] = point[sp]["I"] - point[sp][other]
            row[k + "_lo"], row[k + "_hi"] = ci(lambda b: b[sp]["I"] - b[sp][other])
        rows.append(row)
    return rows


def afop(jobs):
    todo = [(ds, a) for ds in COHORTS for a in ARCHS
            if "AF" not in C.UNAVAILABLE.get(ds, []) and _complete_seeds(ds, a)[0]]
    with Pool(jobs) as pool:
        return pd.DataFrame(list(itertools.chain.from_iterable(pool.map(afop_job, todo))))


def main(jobs, only):
    OUT.mkdir(parents=True, exist_ok=True)
    only = only or ["axis", "recl", "afop", "prereg"]
    summary = json.loads((OUT / "summary.json").read_text()) if (OUT / "summary.json").exists() else {}
    if "axis" in only:
        agree, rec, by_field, by_dev, gaps = axis_analysis(jobs)
        agree.to_csv(OUT / "axis_label_agreement.csv", index=False, float_format="%.4f")
        rec.to_csv(OUT / "axis_records.csv.gz", index=False, float_format="%.2f")
        by_field.to_csv(OUT / "axis_ptbxl_by_field.csv", index=False, float_format="%.3f")
        by_dev.to_csv(OUT / "axis_ptbxl_by_device.csv", index=False, float_format="%.3f")
        gaps.to_csv(OUT / "axis_ptbxl_coded_gaps.csv", index=False, float_format="%.4f")
        summary["axis"] = {"agreement": agree.to_dict("records")}
    if "recl" in only:
        recl = reclassification()
        recl.to_csv(OUT / "reclassification.csv", index=False, float_format="%.4f")
        summary["reclassification"] = recl.to_dict("records")
    if "afop" in only:
        af = afop(jobs)
        af.to_csv(OUT / "af_operating_points.csv", index=False, float_format="%.4f")
    if "prereg" in only:
        pr = prereg()
        pr.to_csv(OUT / "prereg_lateral.csv", index=False)
        summary["prereg"] = {"confirmed": int((pr.confirmed == "yes").sum()), "total": len(pr)}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print("done", only)


if __name__ == "__main__":
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--jobs", type=int, default=24)
    ap_.add_argument("--only", nargs="+", default=None, help="axis recl afop prereg")
    a_ = ap_.parse_args()
    main(a_.jobs, a_.only)
