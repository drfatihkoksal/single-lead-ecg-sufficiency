"""Supplementary material for the resubmission, generated from the data caches and
artifacts_v2/aggregate*/ so that every number is traceable.

  S1  data provenance and harmonization notes
  Table S1  exclusions by cohort and reason
  Table S2  twelve-lead reference performance per class (AUPRC, AUROC)
  Table S3  sensitivity of the sufficiency classification to the margin
  Table S4  agreement between models and between cohorts (incl. Cohen's kappa)
  Table S5  behaviour of single-lead models on missed diagnoses
  Table S6  broad label definition (SE-ResNet, one seed) vs primary definition
  Figures S1-S2 captions, Supplementary Data 1 description, TRIPOD+AI checklist

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
            + "\n\nCohen's kappa is close to zero for the non-rhythm classes although agreement is high, because almost no non-rhythm cell is sufficient (the prevalence paradox of kappa); Gwet's AC1 is reported in the main text for this reason.")


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
    ("Methods: data preparation", "7", "Sections 2.1 and 2.3; Supplementary S1"),
    ("Methods: outcome", "8a", "Section 2.2; Table 2"),
    ("Methods: outcome", "8b", "Labels are those assigned by the source institutions; see Section 4.7"),
    ("Methods: outcome", "8c", "Not applicable (retrospective labels)"),
    ("Methods: predictors", "9a", "Sections 2.4 and 2.5.3"), ("Methods: predictors", "9b", "Sections 2.3, 2.4 and 2.5.3"),
    ("Methods: predictors", "9c", "Not applicable"),
    ("Methods: sample size", "10", "Section 2.1 (all available recordings used); Section 4.8 (classes with few positives)"),
    ("Methods: missing data", "11", "Section 2.1; Table S1 (no missing signal values; excluded recordings listed)"),
    ("Methods: analytical methods", "12a", "Section 2.3"), ("Methods: analytical methods", "12b", "Sections 2.3 and 2.5"),
    ("Methods: analytical methods", "12c", "Section 2.5"), ("Methods: analytical methods", "12d", "Sections 2.8 and 3.4 (between cohorts)"),
    ("Methods: analytical methods", "12e", "Sections 2.6 to 2.9"), ("Methods: analytical methods", "12f", "Not applicable"),
    ("Methods: analytical methods", "12g", "Section 2.5.2 (seed ensemble)"),
    ("Methods: class imbalance", "13", "Section 2.5.2 (class-weighted loss); Section 2.5.3"),
    ("Methods: fairness", "14", "Not addressed"), ("Methods: model output", "15", "Sections 2.5.2 and 2.9"),
    ("Methods: training vs evaluation", "16", "Section 2.1; Section 4.7"),
    ("Methods: ethical approval", "17", "Section 2.10 (public de-identified data)"),
    ("Open science", "18a", "Declarations"), ("Open science", "18b", "Declarations"),
    ("Open science", "18c", "No protocol was prepared"), ("Open science", "18d", "Not registered"),
    ("Open science", "18e", "Data availability statement"), ("Open science", "18f", "Code availability statement"),
    ("Patient and public involvement", "19", "No involvement"),
    ("Results: participants", "20a", "Table 1; Table S1"), ("Results: participants", "20b", "Table 1; Table 2"),
    ("Results: participants", "20c", "Table 2"), ("Results: model development", "21", "Tables 1 and 2"),
    ("Results: model specification", "22", "Section 2.5; code repository"),
    ("Results: model performance", "23a", "Section 3; Table 3; Table S2; Supplementary Data 1"),
    ("Results: model performance", "23b", "Section 3.4; Table S4"), ("Results: model updating", "24", "Not applicable"),
    ("Discussion: interpretation", "25", "Sections 4.1 to 4.5"), ("Discussion: limitations", "26", "Section 4.8"),
    ("Discussion: usability", "27a", "Section 4.8 (wearable recordings)"), ("Discussion: usability", "27b", "Not applicable"),
    ("Discussion: usability", "27c", "Sections 4.4 and 4.8"),
]


def main():
    parts = [
        "# Supplementary material",
        s1_text(),
        "## Table S1. Exclusions by cohort and reason", table_s1(),
        "## Table S2. Twelve-lead reference performance",
        "AUPRC (AUROC) of the twelve-lead model for each class, cohort and model (seed ensembles for the deep learning models). n/a: class not available in the source; –: the feature-based model was trained in Chapman and PTB-XL only.",
        table_s2(),
        "## Table S3. Sensitivity of the sufficiency classification to the margin",
        "Number of sufficient single-lead cells (SE-ResNet and InceptionTime combined, all cohorts) under absolute margins on the AUPRC gap and relative margins on the gap divided by the twelve-lead AUPRC.",
        table_s3(),
        "## Table S4. Agreement between models and between cohorts", table_s4(),
        "## Table S5. Behaviour of single-lead models on missed diagnoses",
        "For single-lead cells classified as a loss: proportion of missed positive recordings that the model reported as another abnormal class absent from the recording, against the same proportion in random abnormal recordings negative for the missed class (Section 2.9).",
        table_s5(),
        "## Table S6. Broad label definition",
        "Single-lead sufficiency under the primary and the broad label definitions (Table 2) for the classes whose definition differs (SE-ResNet).",
        table_s6(),
        "## Figures S1 and S2",
        "![](../figures/v2/fig2_sufficiency_inceptiontime.png)\n\n**Figure S1.** Diagnostic sufficiency map for InceptionTime (as Figure 2).\n\n![](../figures/v2/fig2_sufficiency_gbm.png)\n\n**Figure S2.** Diagnostic sufficiency map for the feature-based model in Chapman and PTB-XL (single leads only). In PTB-XL, supraventricular tachycardia (three positive test recordings, twelve-lead AUPRC 0.02) appears sufficient on many leads; this illustrates why classes with a twelve-lead AUPRC below 0.50 are flagged rather than interpreted.",
        "## Supplementary Data 1",
        "Supplementary_Data_1.csv (comma-separated): one row per cohort, model, lead configuration and class, with the number of positive test recordings, AUPRC, AUROC, twelve-lead AUPRC, gap, relative gap, bootstrap 95% confidence interval, seed standard deviation, q-value and the sufficiency classification under every margin.",
        "## TRIPOD+AI checklist",
        "Items and numbering follow the TRIPOD+AI statement (Collins et al., BMJ 2024;385:e078378).",
        _md(pd.DataFrame(TRIPOD, columns=["Section", "Item", "Reported in"])),
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n\n".join(parts) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
