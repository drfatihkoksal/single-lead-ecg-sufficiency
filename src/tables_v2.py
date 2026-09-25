"""Manuscript tables for the resubmission, generated from the data caches and
artifacts_v2/aggregate/ so that every number in them is traceable.

Table 1  cohorts: source, country, recordings, partition
Table 2  diagnostic classes: SNOMED CT codes and test-set positives per cohort
Table 3  single-lead sufficiency by class and cohort: best lead (SE-ResNet) with its
         AUPRC gap and 95% CI, number of sufficient single leads per model, and the
         reduced lead sets that were sufficient

Run:  python -m src.tables_v2      -> submission_cibm/tables.md
"""
import json

import numpy as np
import pandas as pd

from . import config as C

AGG = C.ART_V2 / "aggregate"
OUT = C.ROOT / "submission_cibm" / "tables.md"
NAMES = {"chapman": "Chapman", "ningbo": "Ningbo", "georgia": "Georgia", "ptbxl_snomed": "PTB-XL"}
SOURCE = {
    "chapman": ("Shaoxing People's Hospital, China", "PhysioNet/CinC Challenge 2021, v1.0.3", "Record level"),
    "ningbo": ("Ningbo First Hospital, China", "PhysioNet ecg-arrhythmia, v1.0.0", "Record level"),
    "georgia": ("Emory University, United States", "PhysioNet/CinC Challenge 2021, v1.0.3", "Record level"),
    "ptbxl_snomed": ("Physikalisch-Technische Bundesanstalt, Germany",
                     "PhysioNet/CinC Challenge 2021, v1.0.3 (PTB-XL v1.0.1)", "Patient level (official folds)"),
}
CLASS_NAME = {"SR": "Sinus rhythm", "SB": "Sinus bradycardia", "ST": "Sinus tachycardia",
              "AF": "Atrial fibrillation", "AFL": "Atrial flutter", "SVT": "Supraventricular tachycardia",
              "LAD": "Left axis deviation", "RBBB": "Right bundle branch block",
              "LVH": "Left ventricular hypertrophy", "TWC": "T-wave change", "STTC": "ST-T change"}
ORDER = ["SR", "SB", "ST", "AF", "AFL", "SVT", "LAD", "RBBB", "LVH", "TWC", "STTC"]
DEVICE_LABEL = {"I+II": "I+II", "LIMB6": "limb 6", "CH3": "I, II, V2", "CH4": "I, II, III, V2"}


def _num(txt):
    """Typographic minus sign for negative numbers."""
    return txt.replace("-", "\u2212")


def _md(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    out += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(out)


def table1():
    rows = []
    for ds in NAMES:
        rep = json.load(open(C.ds_paths(ds, "any").cache / "prep_report.json"))
        sp = rep["split"]
        inst, rel, part = SOURCE[ds]
        rows.append({"Cohort": NAMES[ds], "Source": inst, "Release": rel,
                     "Recordings (available / analysed)": f"{rep['n_headers']:,} / {rep['n_kept']:,}".replace(",", " "),
                     "Training / validation / test": f"{sp['train']:,} / {sp['val']:,} / {sp['test']:,}".replace(",", " "),
                     "Partition": part})
    return _md(pd.DataFrame(rows))


def table2():
    rows = []
    for cl in ORDER:
        r = {"Class": f"{CLASS_NAME[cl]} ({cl})",
             "Primary SNOMED CT codes": ", ".join(C.SNOMED_MAP[cl]),
             "Added in broad definition": ", ".join(c for c in C.SNOMED_MAP_BROAD[cl] if c not in C.SNOMED_MAP[cl]) or "–"}
        for ds in NAMES:
            P = C.ds_paths(ds, "any")
            y = np.load(P.cache / "labels.npy"); te = np.load(P.splits / "split.npz")["test"]
            j = C.CLASSES.index(cl)
            r[NAMES[ds]] = "n/a" if cl in C.UNAVAILABLE.get(ds, []) else int(y[te, j].sum())
        rows.append(r)
    return _md(pd.DataFrame(rows))


def _pbest():
    f = C.ART_V2 / "review" / "bootstrap.csv"
    if not f.exists():
        return {}
    b = pd.read_csv(f)
    b = b[b.arch == "seresnet"]
    return {(r.cohort, r.cls): {ld: getattr(r, f"pbest_{ld}") for ld in C.LEADS} for r in b.itertuples()}


PBEST = _pbest()


def table3():
    c = pd.read_csv(AGG / "cells.csv")
    s = c[c.kind == "single"].dropna(subset=["gap"])
    d = c[c.kind == "device"]
    rows = []
    for cl in ORDER:
        for ds in NAMES:
            g = s[(s.cohort == ds) & (s.cls == cl)]
            if g.empty:
                rows.append({"Class": cl, "Cohort": NAMES[ds], "Best single lead (P)": "n/a",
                             "AUPRC gap [95% CI]": "", "Sufficient single leads (SE-ResNet / InceptionTime)": "",
                             "Sufficient reduced sets (SE-ResNet)": ""})
                continue
            se = g[g.arch == "seresnet"]
            b = se.loc[se.gap.idxmin()]
            n = {a: int((g[g.arch == a].sufficiency == "sufficient").sum()) for a in ("seresnet", "inceptiontime")}
            dv = d[(d.cohort == ds) & (d.cls == cl) & (d.arch == "seresnet") & (d.sufficiency == "sufficient")]
            flag = C.flag(ds, cl, b.low_ceiling, b.n_pos_test)
            pb = PBEST.get((ds, cl), {}).get(b.spec)
            rows.append({"Class": cl + flag, "Cohort": NAMES[ds],
                         "Best single lead (P)": b.spec + (f" ({pb:.2f})" if pb is not None else ""),
                         "AUPRC gap [95% CI]": _num(f"{b.gap:.3f} [{b.gap_ci_lo:.3f}, {b.gap_ci_hi:.3f}]"),
                         "Sufficient single leads (SE-ResNet / InceptionTime)": f"{n['seresnet']} / {n['inceptiontime']}",
                         "Sufficient reduced sets (SE-ResNet)": "; ".join(DEVICE_LABEL[x] for x in dv.spec) or "none"})
    return _md(pd.DataFrame(rows))


def main():
    parts = [
        "**Table 1.** Cohorts. Recordings were excluded when duplicated, shorter than 10 s or with a malformed header (Section 2.1).",
        table1(),
        "**Table 2.** Diagnostic classes, their SNOMED CT codes and the number of positive recordings in each test set. n/a: not available in the source (Ningbo records every atrial fibrillation and flutter as atrial flutter).",
        table2(),
        "**Table 3.** Single-lead sufficiency by class and cohort. Best single lead: lead with the smallest AUPRC gap to the twelve-lead model (SE-ResNet seed ensemble); P: proportion of bootstrap resamples in which this lead had the smallest gap. Sufficient: upper 95% confidence bound of the gap below 0.05. † twelve-lead AUPRC below 0.50 or fewer than 30 positive test recordings; ‡ label inconsistent with the signal (Section 3.1); interpret with caution.",
        table3(),
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n\n".join(parts) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
