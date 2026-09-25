"""Supplementary material for the resubmission, generated from the data caches and
artifacts_v2/aggregate*/ so that every number is traceable.

  S1  data provenance and harmonization notes
  S2  models, training and analysis details moved from the main text
  Table S1  exclusions by cohort and reason
  Table S2  twelve-lead reference performance per class (AUPRC, AUROC)
  Table S3  sensitivity of the sufficiency classification to the margin
  Table S4  agreement between models and between cohorts (incl. Cohen's kappa)
  Table S5  behaviour of single-lead models on missed diagnoses
  Table S6  broad label definition (SE-ResNet, one seed) vs primary definition
  Tables S7-S10  lead I vs II/aVF for AF, noise floor from equivalent lead sets, within-class ranking
                 agreement and variance decomposition, label consistency (artifacts_v2/review/)
  Table S11  prevalence-normalized sufficiency, test-set selection optimism, physiological lead match
  Figures S1-S2 captions, TRIPOD+AI checklist

Run:  python -m src.supplement_v2     -> submission_cibm/supplementary.md
"""
import numpy as np
import pandas as pd

from . import config as C
from .tables_v2 import NAMES, ORDER, _md, _num

AGG, AGG_B = C.ART_V2 / "aggregate", C.ART_V2 / "aggregate_broad"
OUT = C.ROOT / "submission_cibm" / "supplementary.md"
ARCH = {"seresnet": "SE-ResNet", "inceptiontime": "InceptionTime", "gbm": "Feature-based"}
GROUP_ORDER = ["rhythm", "axis", "conduction", "hypertrophy", "repolarization"]


def s1_text():
    return """## S1. Data provenance and harmonization

*Georgia gains.* The Georgia signal files used here are byte-identical to those of the 2021 Challenge release (verified against the release checksums), but the headers distributed with the 2020 release state a gain of 4880 units per millivolt instead of 1000. Read with the 2020 headers, all amplitudes would be about five times too small; the median peak-to-peak amplitude of lead II was 1.02 mV with the 2021 gain and 0.19 mV with the 2020 gain. Gains and diagnosis codes were therefore taken from the 2021 headers.

*Chapman and Ningbo.* The PhysioNet release of the combined database contains 45 152 recordings. The 10 247 recordings with identifiers up to JS10646 were identical in signal and diagnosis codes to the Chapman folder of the 2021 Challenge release, except one recording (JS01052) whose header in the combined release is malformed and whose Challenge copy was used. The remaining identifiers formed the Ningbo cohort.

*Duplicates.* Hashing the raw sample arrays identified 87 pairs of byte-identical recordings: 71 across Chapman and Ningbo, 13 within Chapman and 3 within Ningbo. In 43 pairs the two copies carried different diagnosis codes. Duplicates across cohorts were removed from Ningbo; within a cohort one copy was kept when the codes agreed and both were removed otherwise.

*Left ventricular hypertrophy.* Chapman and Ningbo record electrocardiographic hypertrophy almost exclusively with the code for left ventricular high voltage (55827005; 1 295 and 4 106 recordings) and rarely with the code for left ventricular hypertrophy (164873001; 15 and 632), whereas Georgia and PTB-XL use only the latter. Both codes were included in the primary definition.

*Atrial fibrillation in Ningbo.* Ningbo codes every atrial fibrillation and atrial flutter recording as atrial flutter (no recording carries the atrial fibrillation code), so both classes were excluded from all Ningbo analyses.

*ST-T change.* Nonspecific ST-T change is coded as 428750005 in Chapman and Georgia and as 55930002 ("ST changes") in Ningbo and in the 2021 release of PTB-XL, where it corresponds to the PTB-XL statement for non-specific ST changes. Both codes were included.

*PTB-XL.* Record identifiers of the 2021 Challenge release (HRnnnnn) equal the PTB-XL ecg_id, and the signals are identical to PTB-XL version 1.0.1 (maximum absolute difference 0 mV in 32 randomly checked recordings), which allowed the official patient-disjoint folds to be used with the Challenge SNOMED CT labels."""


def s2_text():
    return """## S2. Models, training and analysis details

*Signal handling.* Raw samples were converted to millivolts with the gains in the record headers and arranged in the standard lead order. The limb leads satisfied Einthoven's relation (lead III = lead II − lead I) in every cohort, with a root-mean-square residual of at most 0.001 mV. No filtering was applied to the inputs of the deep learning models. Each input lead was standardized with the mean and standard deviation of that lead, estimated on 2 000 training recordings.

*SE-ResNet.* One-dimensional residual network with squeeze-and-excitation blocks (Hu et al., 2018) on the 500 Hz signal. A stem convolution (kernel 15, stride 2, 32 channels) with batch normalization, rectified linear activation and max pooling was followed by four stages of two residual blocks (32, 64, 128 and 256 channels; kernel 7; stride 2 in the first block of stages 2 to 4). Each block contained two convolutions with batch normalization, dropout of 0.1 and a squeeze-and-excitation module with reduction ratio 8. Global average pooling, dropout of 0.2 and a linear layer produced eleven outputs (2.2 million parameters).

*InceptionTime.* As in the PTB-XL benchmark (Strodthoff et al., 2021), the input was average-pooled to 100 Hz. Six inception modules (Ismail Fawaz et al., 2020), each with a 32-channel bottleneck convolution (omitted for single-channel input), three parallel convolutions with kernel sizes 39, 19 and 9 and a max-pooling branch (32 filters each), were followed by batch normalization and rectified linear activation, with a residual connection every third module. Global average pooling and a linear layer produced the outputs (0.4 million parameters).

*Training.* The loss was binary cross-entropy with a positive-class weight equal to the ratio of negative to positive training recordings of each class. Optimization used AdamW (learning rate 10⁻³, weight decay 10⁻⁴) with cosine annealing over at most 40 epochs, a batch size of 128 and mixed precision. Training recordings were augmented by random amplitude scaling (factor 0.9 to 1.1) and a random circular time shift of up to 100 ms. Training stopped when the mean average precision on the validation set had not improved for eight epochs, and the weights of the best validation epoch were kept. Seeds 1337, 1, 2, 3 and 4 were used.

*Feature-based model.* Each lead was band-pass filtered (0.5 to 40 Hz, third-order Butterworth, zero phase) and R peaks were detected with NeuroKit2. From each lead we derived ten rhythm features (mean, standard deviation, coefficient of variation, minimum and maximum of the RR interval, root mean square of successive differences, proportion of successive differences above 50 ms, number of beats, median absolute successive difference divided by the mean RR interval, and mean correlation of each beat with the median beat), eleven measurements on the median beat (R and S amplitude and their difference, QRS width, ST level 40 and 80 ms after the J point, maximum, minimum, extreme value and area of the T wave, and P-wave amplitude, all relative to the PR segment), and the median beat resampled to 50 Hz (35 values): 56 features per lead. One LightGBM classifier per class was trained with at most 2 000 trees (learning rate 0.03, 31 leaves, row and column subsampling of 0.8 and 0.5), the positive-class weight above, and early stopping on validation average precision after 100 rounds without improvement.

*Bootstrap.* Within each class and resample, the same recordings were used for every configuration and the twelve-lead model; resamples without a positive recording were discarded, and percentile intervals were used.

*Behaviour on missed diagnoses.* A decision threshold was set for each class on the validation set by maximizing the F1 score over thresholds from 0.05 to 0.95. A positive recording was missed when its predicted probability was below the threshold. Abnormal classes were all classes except sinus rhythm; a second proportion considered only non-rhythm classes. The expected proportion was estimated from 1 000 random samples, of equal size, of test recordings negative for the missed class but carrying at least one abnormal label. Cells with fewer than five missed recordings were not analysed. Proportions were computed for each seed with that seed's thresholds and averaged over seeds.

*Implementation.* Python 3.13, PyTorch 2.11, scikit-learn 1.9, LightGBM 4.7, NeuroKit2 0.2.13 and WFDB 4.3, on one NVIDIA RTX 5090 graphics processor."""


def table_s1():
    import json
    rows = []
    for ds in NAMES:
        rep = json.load(open(C.ds_paths(ds, "any").cache / "prep_report.json"))
        ex = rep["excluded"]
        th = lambda v: f"{v:,}".replace(",", " ")
        rows.append({"Cohort": NAMES[ds], "Available": th(rep["n_headers"]),
                     "Duplicate of a Chapman recording": ex.get("duplicate of a Chapman record", 0),
                     "Duplicate within cohort, codes agree": ex.get("within-cohort duplicate (same Dx)", 0),
                     "Duplicate within cohort, codes differ": ex.get("within-cohort duplicate (conflicting Dx)", 0),
                     "Shorter than 10 s": ex.get("length != 10 s", 0),
                     "Malformed header": ex.get("malformed header", 0), "Analysed": th(rep["n_kept"])})
    return _md(pd.DataFrame(rows))


def table_s2():
    c = pd.read_csv(AGG / "cells.csv")
    e = c[c.kind == "ceiling"]
    rows = []
    for cl in ORDER:
        for ds in NAMES:
            r = {"Class": cl, "Cohort": NAMES[ds]}
            for a in ARCH:
                g = e[(e.cohort == ds) & (e.arch == a) & (e.cls == cl)]
                if g.empty:
                    r[ARCH[a]] = "–"
                    continue
                v = g.iloc[0]
                r[ARCH[a]] = "n/a" if np.isnan(v.auprc) else f"{v.auprc:.2f} ({v.auroc:.2f})"
            rows.append(r)
    return _md(pd.DataFrame(rows))


def table_s3():
    c = pd.read_csv(AGG / "cells.csv")
    s = c[(c.kind == "single") & c.arch.isin(["seresnet", "inceptiontime"])].dropna(subset=["gap"])
    cols = [("suff_abs0.02", "Absolute 0.02"), ("suff_abs0.05", "Absolute 0.05 (primary)"),
            ("suff_abs0.1", "Absolute 0.10"), ("suff_rel0.05", "Relative 5%"), ("suff_rel0.1", "Relative 10%")]
    rows = []
    for gname in GROUP_ORDER:
        g = s[s.dx_group == gname]
        r = {"Diagnostic group": gname, "Single-lead cells": len(g)}
        for col, lab in cols:
            r[lab] = int((g[col] == "sufficient").sum())
        rows.append(r)
    return _md(pd.DataFrame(rows))


def table_s4():
    ca = pd.read_csv(AGG / "concordance_arch.csv")
    cc = pd.read_csv(AGG / "concordance_cohort.csv")
    fmt = lambda d: d.assign(**{k: d[k].map(lambda v: _num(f"{v:.2f}")) for k in ("r", "agree", "ac1", "kappa")})
    ca = fmt(ca); cc = fmt(cc)
    a = pd.DataFrame({"Cohort": ca.cohort.map(NAMES), "Models": ca.arch_a.map(ARCH) + " vs " + ca.arch_b.map(ARCH),
                      "Classes": ca.classes.str.replace("morphology", "non-rhythm"), "Cells": ca.n, "Pearson r": ca.r,
                      "Agreement": ca.agree, "Gwet AC1": ca.ac1, "Cohen kappa": ca.kappa})
    b = pd.DataFrame({"Model": cc.arch.map(ARCH), "Cohorts": cc.cohort_a.map(NAMES) + " vs " + cc.cohort_b.map(NAMES),
                      "Classes": cc.classes.str.replace("morphology", "non-rhythm"), "Cells": cc.n, "Pearson r": cc.r,
                      "Agreement": cc.agree, "Gwet AC1": cc.ac1, "Cohen kappa": cc.kappa})
    return ("*Between models within a cohort*\n\n" + _md(a) + "\n\n*Between cohorts within a model*\n\n" + _md(b)
            + "\n\nAgreement, Gwet's AC1 and Cohen's kappa refer to the two-category classification (sufficient versus not). For the non-rhythm classes almost no cell is sufficient, so two-category agreement is high by construction and kappa is close to zero; the main text therefore reports the three-category agreement (sufficient, indeterminate, loss) given in Table S9.")


def table_s5():
    d = pd.read_csv(AGG / "confusion_summary.csv")
    d = pd.DataFrame({"Cohort": d.cohort.map(NAMES), "Model": d.arch.map(ARCH), "Loss cells analysed": d.cells,
                      "Reported as another absent abnormality": (100 * d.any_fp).round(0).astype(int).astype(str) + "%",
                      "Expected (comparison recordings)": (100 * d.any_fp_null).round(0).astype(int).astype(str) + "%",
                      "Excess, percentage points": (100 * d.excess_any).round(1),
                      "Cells above chance (p < 0.05)": d.cells_above_null})
    return _md(d)


def table_s6():
    if not (AGG_B / "cells.csv").exists():
        return "*Pending: the broad-definition run has not finished.*"
    p = pd.read_csv(AGG / "cells.csv"); b = pd.read_csv(AGG_B / "cells.csv")
    p = p[(p.arch == "seresnet") & (p.kind == "single")]
    b = b[(b.arch == "seresnet") & (b.kind == "single")].assign(cohort=lambda x: x.cohort.str.replace("_broad", ""))
    rows = []
    for cl in ["SVT", "RBBB", "TWC", "STTC"]:
        for ds in NAMES:
            gp, gb = p[(p.cohort == ds) & (p.cls == cl)], b[(b.cohort == ds) & (b.cls == cl)]
            m = gp.merge(gb, on="spec", suffixes=("_p", "_b"))
            if m.empty:
                continue
            rows.append({"Class": cl, "Cohort": NAMES[ds],
                         "Twelve-lead AUPRC, primary / broad": f"{m.auprc_ceiling_p.iloc[0]:.2f} / {m.auprc_ceiling_b.iloc[0]:.2f}",
                         "Sufficient single leads, primary / broad": f"{(m.sufficiency_p == 'sufficient').sum()} / {(m.sufficiency_b == 'sufficient').sum()}",
                         "Loss, primary / broad": f"{(m.sufficiency_p == 'loss').sum()} / {(m.sufficiency_b == 'loss').sum()}",
                         "Best lead, primary / broad": f"{gp.loc[gp.gap.idxmin(), 'spec']} / {gb.loc[gb.gap.idxmin(), 'spec']}",
                         "Correlation of gaps": _num(f"{np.corrcoef(m.gap_p, m.gap_b)[0, 1]:.2f}")})
    note = ("The primary estimates are five-seed ensembles and the broad-definition estimates a single seed, so the "
            "broad definition has somewhat wider confidence intervals. Classes whose definition does not change "
            "(the rhythm classes except supraventricular tachycardia, left axis deviation and left ventricular "
            "hypertrophy) are not shown.")
    return _md(pd.DataFrame(rows)) + "\n\n" + note


TRIPOD = [
    ("Title", "1", "Title"), ("Abstract", "2", "Abstract"),
    ("Introduction", "3a", "Section 1"), ("Introduction", "3b", "Section 1; Section 4.4"),
    ("Introduction", "3c", "Not addressed: sociodemographic data were not available consistently across the four public cohorts"),
    ("Introduction", "4", "Section 1, last paragraph"),
    ("Methods: data", "5a", "Section 2.1; Table 1; Supplementary S1"),
    ("Methods: data", "5b", "Not reported here; collection periods are given in the source publications of each database"),
    ("Methods: participants", "6a", "Section 2.1; Table 1"), ("Methods: participants", "6b", "Section 2.1; Table S1"),
    ("Methods: participants", "6c", "Not applicable"),
    ("Methods: data preparation", "7", "Sections 2.1 and 2.3; Supplementary S1 and S2"),
    ("Methods: outcome", "8a", "Section 2.2; Table 2"),
    ("Methods: outcome", "8b", "Labels are those assigned by the source institutions; see Section 4.6"),
    ("Methods: outcome", "8c", "Not applicable (retrospective labels)"),
    ("Methods: predictors", "9a", "Sections 2.3 and 2.4; Supplementary S2"), ("Methods: predictors", "9b", "Sections 2.3 and 2.4; Supplementary S2"),
    ("Methods: predictors", "9c", "Not applicable"),
    ("Methods: sample size", "10", "Section 2.1 (all available recordings used); Section 4.7 (classes with few positives)"),
    ("Methods: missing data", "11", "Section 2.1; Table S1 (no missing signal values; excluded recordings listed)"),
    ("Methods: analytical methods", "12a", "Section 2.3"), ("Methods: analytical methods", "12b", "Section 2.4; Supplementary S2"),
    ("Methods: analytical methods", "12c", "Section 2.5"), ("Methods: analytical methods", "12d", "Sections 2.6 and 3.4 (between cohorts)"),
    ("Methods: analytical methods", "12e", "Sections 2.5 and 2.6"), ("Methods: analytical methods", "12f", "Not applicable"),
    ("Methods: analytical methods", "12g", "Section 2.4 (seed ensemble)"),
    ("Methods: class imbalance", "13", "Section 2.4; Supplementary S2 (class-weighted loss)"),
    ("Methods: fairness", "14", "Not addressed"), ("Methods: model output", "15", "Section 2.6; Supplementary S2"),
    ("Methods: training vs evaluation", "16", "Section 2.1; Section 4.6"),
    ("Methods: ethical approval", "17", "Declarations, ethics statement (public de-identified data)"),
    ("Open science", "18a", "Declarations"), ("Open science", "18b", "Declarations"),
    ("Open science", "18c", "No protocol was prepared"), ("Open science", "18d", "Not registered"),
    ("Open science", "18e", "Data availability statement"), ("Open science", "18f", "Code availability statement"),
    ("Patient and public involvement", "19", "No involvement"),
    ("Results: participants", "20a", "Table 1; Table S1"), ("Results: participants", "20b", "Table 1; Table 2"),
    ("Results: participants", "20c", "Table 2"), ("Results: model development", "21", "Tables 1 and 2"),
    ("Results: model specification", "22", "Section 2.4; Supplementary S2; code repository"),
    ("Results: model performance", "23a", "Section 3; Table 3; Table S2; archived cell-level results (Zenodo)"),
    ("Results: model performance", "23b", "Section 3.4; Table S4"), ("Results: model updating", "24", "Not applicable"),
    ("Discussion: interpretation", "25", "Sections 4.1 to 4.5"), ("Discussion: limitations", "26", "Section 4.7"),
    ("Discussion: usability", "27a", "Section 4.7 (wearable recordings)"), ("Discussion: usability", "27b", "Not applicable"),
    ("Discussion: usability", "27c", "Sections 4.4 and 4.7"),
]


REV = C.ART_V2 / "review"
MODEL = {"seresnet": "SE-ResNet", "inceptiontime": "InceptionTime", "gbm": "Feature-based"}


def _ci(m, lo, hi):
    return _num(f"{m:.3f} [{lo:.3f}, {hi:.3f}]")


def table_s7():
    b = pd.read_csv(REV / "bootstrap.csv"); c = pd.read_csv(AGG / "cells.csv")
    rows = []
    for r in b[b.cls == "AF"].itertuples():
        a = c[(c.cohort == r.cohort) & (c.arch == r.arch) & (c.cls == "AF") & (c.kind == "single")].set_index("spec")
        rows.append({"Cohort": NAMES[r.cohort], "Model": MODEL[r.arch],
                     "AUPRC lead I / II / aVF": " / ".join(f"{a.loc[x, 'auprc']:.2f}" for x in ("I", "II", "aVF")),
                     "Twelve-lead AUPRC": f"{a.loc['I', 'auprc_ceiling']:.2f}",
                     "Gap(I) − gap(II) [95% CI]": _ci(r.d_I_II, r.d_I_II_lo, r.d_I_II_hi),
                     "Gap(I) − gap(aVF) [95% CI]": _ci(r.d_I_aVF, r.d_I_aVF_lo, r.d_I_aVF_hi)})
    rr = pd.read_csv(REV / "af_rr_only.csv")
    rows2 = [{"Cohort": NAMES[r.cohort], "Input": f"RR intervals of lead {r.model[3:-1]} only",
              "AUPRC": f"{r.auprc:.2f}", "Twelve-lead AUPRC (SE-ResNet)": f"{r.auprc_seresnet_12:.2f}",
              "Gap to SE-ResNet twelve-lead [95% CI]": _ci(r.gap_vs_seresnet_12, r.gap_lo, r.gap_hi)}
             for r in rr.itertuples()]
    return _md(pd.DataFrame(rows)) + "\n\n" + _md(pd.DataFrame(rows2))


def table_s8():
    b = pd.read_csv(REV / "bootstrap.csv"); c = pd.read_csv(AGG / "cells.csv")
    d = c[(c.kind == "device") & (c.arch != "gbm") & (c.sufficiency != "na")]
    rows = []
    for a, bb, lab in (("I+II", "LIMB6", "Leads I and II vs six limb leads"),
                       ("CH3", "CH4", "Leads I, II, V2 vs I, II, III, V2")):
        k = f"d_{a}_{bb}"
        x = b.dropna(subset=[k]); x = x[~((x.cohort == "ningbo") & x.cls.isin(["AF", "AFL"]))]
        m = d[d.spec.isin([a, bb])].pivot_table(index=["cohort", "arch", "cls"], columns="spec",
                                                  values="sufficiency", aggfunc="first").dropna()
        rows.append({"Comparison": lab, "Cells": len(x),
                     "Median absolute gap difference": f"{x[k].abs().median():.3f}",
                     "95th percentile of the absolute gap difference": f"{x[k].abs().quantile(.95):.3f}",
                     "Different sufficiency category": f"{int((m[a] != m[bb]).sum())} ({100 * (m[a] != m[bb]).mean():.0f}%)"})
    return _md(pd.DataFrame(rows))


def table_s9():
    k = pd.read_csv(REV / "concordance_within_class.csv")
    w = k[k.what == "kendall_w_cohorts"].pivot_table(index="cls", columns="arch", values="value")
    sp = k[k.what == "spearman_archs"].groupby("cls").value.agg(["min", "max"])
    rows = []
    for cl in ORDER:
        if cl not in w.index:
            continue
        rows.append({"Class": cl, "Kendall's W across cohorts (SE-ResNet / InceptionTime)":
                         f"{w.loc[cl, 'seresnet']:.2f} / {w.loc[cl, 'inceptiontime']:.2f}",
                     "Spearman ρ between architectures (range over cohorts)":
                         _num(f"{sp.loc[cl, 'min']:.2f} to {sp.loc[cl, 'max']:.2f}")})
    v = pd.read_csv(REV / "variance_decomposition.csv", index_col=0)
    names = {"cls": "Class", "lead": "Lead", "cls:lead": "Class × lead", "cls:cohort": "Class × cohort",
             "cohort": "Cohort", "arch": "Architecture", "Residual": "Residual"}
    vd = pd.DataFrame([{"Source": names[i], "Share of variance": f"{100 * v.loc[i, 'share']:.1f}%"}
                       for i in ["cls", "cls:lead", "cls:cohort", "lead", "cohort", "arch", "Residual"]])
    og = pd.read_csv(REV / "ordinal_agreement.csv")
    og["kind"] = np.where(og.comparison == "architectures", "Between architectures (4 cohorts)",
                          "Between cohorts (6 pairs × 2 architectures)")
    agg = og.groupby("kind").agg(ex_lo=("exact", "min"), ex_hi=("exact", "max"), k_lo=("kappa_linear", "min"),
                                 k_hi=("kappa_linear", "max"), a_lo=("ac2_linear", "min"), a_hi=("ac2_linear", "max"))
    oa = pd.DataFrame([{"Comparison": k, "Exact agreement": f"{100 * r.ex_lo:.0f}% to {100 * r.ex_hi:.0f}%",
                        "Linear-weighted κ": f"{r.k_lo:.2f} to {r.k_hi:.2f}", "Gwet's AC2 (linear)": f"{r.a_lo:.2f} to {r.a_hi:.2f}"}
                       for k, r in agg.iterrows()])
    return _md(pd.DataFrame(rows)) + "\n\n" + _md(vd) + "\n\n" + _md(oa)


def table_s10():
    h = pd.read_csv(REV / "label_heart_rate.csv"); co = pd.read_csv(REV / "label_sr_cocoding.csv").set_index("cohort")
    rows = []
    for ds in NAMES:
        g = h[h.cohort == ds].set_index("cls")
        rows.append({"Cohort": NAMES[ds],
                     "SB label with rate < 60/min": f"{100 * g.loc['SB', 'label_agrees_with_hr']:.0f}%",
                     "Regular rate < 60/min with SB label": f"{100 * g.loc['SB', 'hr_rule_labelled']:.0f}%",
                     "ST label with rate > 100/min": f"{100 * g.loc['ST', 'label_agrees_with_hr']:.0f}%",
                     "Regular rate > 100/min with ST label": f"{100 * g.loc['ST', 'hr_rule_labelled']:.0f}%",
                     "SR coded with a morphological diagnosis": f"{100 * co.loc[ds, 'sr_given_morph']:.0f}%",
                     "SR coded without one": f"{100 * co.loc[ds, 'sr_given_no_morph']:.0f}%"})
    return _md(pd.DataFrame(rows))


def table_s11():
    import json
    b = pd.read_csv(REV / "bootstrap.csv")
    n = pd.read_csv(REV / "normalized_sufficiency.csv", index_col=0)
    c = pd.read_csv(AGG / "cells.csv")
    s1 = c[(c.kind == "single") & (c.arch != "gbm")]
    prim = s1.groupby("dx_group").sufficiency.apply(lambda x: int((x == "sufficient").sum()))
    tot = s1[s1.sufficiency != "na"].groupby("dx_group").size()
    rows = [{"Group": g.capitalize(), "Sufficient single-lead cells, primary AUPRC": f"{prim[g]} of {tot[g]}",
             "Sufficient, prevalence-normalized AUPRC": f"{int(n.loc[g, 'sufficient'])} of {tot[g]}"}
            for g in ["rhythm", "axis", "conduction", "hypertrophy", "repolarization"]]
    ph = pd.read_csv(REV / "physiology_null.csv")
    p_all = json.loads((REV / "summary.json").read_text())["physiology_joint_p"]
    rows2 = [{"Class": r.cls, "Leads named by the diagnostic criteria": r.criterion_leads,
              "Best single lead (Chapman, Georgia, Ningbo, PTB-XL)": ", ".join(r.best_leads.split()),
              "Probability under a uniform choice": f"{r.p_uniform:.3f}"} for r in ph.itertuples()]
    opt = b[b.arch != "gbm"].optimism_mean
    txt = (f"Choosing the best lead on the test set underestimated its AUPRC gap by a median of "
           f"{opt.median():.3f} (interquartile range {opt.quantile(.25):.3f} to {opt.quantile(.75):.3f}) when the lead "
           f"was chosen on one random half of the test set and evaluated on the other (200 splits, deep learning models). "
           f"The lead sets were named from the criteria, but the test was defined after the results were known.")
    return _md(pd.DataFrame(rows)) + "\n\n" + _md(pd.DataFrame(rows2)) + "\n\n" + txt


def main():
    parts = [
        "# Supplementary material",
        s1_text(),
        s2_text(),
        "## Table S1. Exclusions by cohort and reason", table_s1(),
        "## Table S2. Twelve-lead reference performance",
        "AUPRC (AUROC) of the twelve-lead model for each class, cohort and model (seed ensembles for the deep learning models). n/a: class not available in the source; –: the feature-based model was trained in Chapman and PTB-XL only.",
        table_s2(),
        "## Table S3. Sensitivity of the sufficiency classification to the margin",
        "Number of sufficient single-lead cells (SE-ResNet and InceptionTime combined, all cohorts) under absolute margins on the AUPRC gap and relative margins on the gap divided by the twelve-lead AUPRC.",
        table_s3(),
        "## Table S4. Agreement between models and between cohorts", table_s4(),
        "## Table S5. Behaviour of single-lead models on missed diagnoses",
        "For single-lead cells classified as a loss: proportion of missed positive recordings that the model reported as another abnormal class absent from the recording, against the same proportion in random abnormal recordings negative for the missed class (Supplementary S2). Because the comparison recordings are themselves abnormal, the excess measures redirection towards a different diagnosis, not the separation of abnormal from normal recordings; it exceeded chance in most Ningbo cells but in a minority of cells in the other cohorts.",
        table_s5(),
        "## Table S6. Broad label definition",
        "Single-lead sufficiency under the primary and the broad label definitions (Table 2) for the classes whose definition differs (SE-ResNet).",
        table_s6(),
        "## Table S7. Lead I against leads II and aVF for atrial fibrillation",
        "Top: absolute AUPRC of the single-lead models and paired bootstrap difference of their AUPRC gaps to the twelve-lead model of the same family (1 000 resamples; a positive value means a larger loss on lead I). Bottom: gradient-boosted trees on the ten RR-interval features of one lead only (five seeds, averaged), compared with the SE-ResNet twelve-lead model. Ningbo does not code atrial fibrillation.",
        table_s7(),
        "## Table S8. Model noise floor from information-equivalent lead sets",
        "Leads I and II determine the six limb leads, and leads I, II and V2 determine leads I, II, III and V2, so differences between these pairs reflect training and optimization rather than information. Deep learning models, all classes and cohorts; gap difference: difference between the AUPRC gaps of the two sets (seed ensembles, mean over paired bootstrap resamples).",
        table_s8(),
        "## Table S9. Agreement of the lead ranking within each class",
        "Top: Kendall's W, agreement of the ranking of the twelve single-lead gaps across the cohorts in which the class is available; Spearman ρ, agreement of the ranking between the two architectures within a cohort. Middle: share of the variance of the single-lead gaps (both architectures, 1 008 cells) explained by each term of a linear model with class, lead, class × lead, class × cohort, cohort and architecture (type II sums of squares). Bottom: agreement of the three-category sufficiency classification (sufficient, indeterminate, loss) over the 60 non-rhythm single-lead cells of each comparison, treating the categories as ordered.",
        table_s9(),
        "## Table S10. Consistency of rhythm labels with the signal",
        "Test recordings. Heart rate from the median RR interval of R peaks detected in lead II (NeuroKit2); regular: not labelled atrial fibrillation or flutter. SB, sinus bradycardia; ST, sinus tachycardia; SR, sinus rhythm; morphological diagnosis: left axis deviation, right bundle branch block, left ventricular hypertrophy, T-wave change or ST-T change. Denominators: columns 2 and 4, recordings carrying the label; columns 3 and 5, regular recordings meeting the rate criterion; columns 6 and 7, regular recordings with a rate of 60 to 100/min with or without a morphological diagnosis.",
        table_s10(),
        "## Table S11. Further robustness checks",
        "Top: sufficiency on the prevalence-normalized AUPRC scale, (AUPRC − π)/(1 − π) with π the test-set prevalence, against the primary scale (deep learning models, margin 0.05 on both scales). Bottom: whether the best single lead (SE-ResNet) falls among the leads used by the diagnostic criteria (axis: I, II, aVF; right bundle branch block: V1, V2 and the terminal S wave in I and V6; left ventricular hypertrophy: Sokolow-Lyon V1, V5, V6 and Cornell aVL, V3), with the binomial probability of at least as many matches if the best lead were chosen at random.",
        table_s11(),
        "## Figures S1 and S2",
        "![](../figures/v2/fig2_sufficiency_inceptiontime.png)\n\n**Figure S1.** Diagnostic sufficiency map for InceptionTime (as Figure 2).\n\n![](../figures/v2/fig2_sufficiency_gbm.png)\n\n**Figure S2.** Diagnostic sufficiency map for the feature-based model in Chapman and PTB-XL (single leads only). In PTB-XL, supraventricular tachycardia (three positive test recordings, twelve-lead AUPRC 0.02) appears sufficient on many leads; this illustrates why classes with a twelve-lead AUPRC below 0.50 are flagged rather than interpreted.",
        "## TRIPOD+AI checklist",
        "Items and numbering follow the TRIPOD+AI statement (Collins et al., BMJ 2024;385:e078378).",
        _md(pd.DataFrame(TRIPOD, columns=["Section", "Item", "Reported in"])),
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n\n".join(parts) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
