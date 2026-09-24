"""Do single-lead models flag a missed diagnosis as some other abnormality?

Resubmission version of confusion_corrected.py, run on the Phase-3 grid.  For every
single-lead cell classified as "loss" in aggregate_seeds (lower 95% CI bound of the
AUPRC gap above the sufficiency margin),
take the test records truly positive for class C that the lead-L model misses
(p_C below the validation-tuned threshold of that seed) and measure how often the
model fires another class the record does NOT carry:

  any_fp    any other non-sinus-rhythm class (rhythm or morphology)
  morph_fp  another morphology class only

The null for each rate is the same statistic on equally many random C-negative
records that carry at least one abnormal label (N_NULL draws).  Rates are computed
per seed with that seed's own thresholds, then averaged over seeds.  Classes a
cohort cannot supply (config.UNAVAILABLE) are excluded as targets and as flags.

Run:  python -m src.confusion_v2        (after aggregate_seeds)
Outputs -> artifacts_v2/aggregate/confusion_cells.csv, confusion_summary.csv
"""
import itertools

import numpy as np
import pandas as pd

from . import config as C
from .aggregate_seeds import MARGIN, OUT
from .concordance import MORPH
from .train import atomic_save

N_NULL = 1000
MIN_MISSED = 5
SR = C.CLASSES.index("SR")


def _cells(ds, arch, rng):
    f = OUT / ds / arch / "ensemble.npz"
    if not f.exists():
        return []
    e = np.load(f, allow_pickle=True)
    specs = [str(s) for s in e["specs"]]
    skip = {C.CLASSES.index(c) for c in C.UNAVAILABLE.get(ds, [])}
    abn = [j for j in range(C.N_CLASSES) if j != SR and j not in skip]
    morph = [C.CLASSES.index(c) for c in MORPH if C.CLASSES.index(c) not in skip]
    rows = []
    for seed in [int(s) for s in e["seeds"]]:
        P = C.ds_paths(ds, arch, seed)
        for i, L in enumerate(C.LEADS):
            d = np.load(P.metrics / f"pred_{L}.npz")
            p, y = d["p"], d["y"]
            fired = p >= np.load(P.models / f"norm_{L}.npz")["thr"]
            fp = fired & (y == 0)
            abnormal_rec = y[:, abn].sum(1) > 0
            for j in range(C.N_CLASSES):
                if j in skip or not e["ci_lo"][specs.index(L), j] > MARGIN:
                    continue
                missed = np.where((y[:, j] == 1) & ~fired[:, j])[0]
                if len(missed) < MIN_MISSED:
                    continue
                o_abn = [k for k in abn if k != j]
                o_mor = [k for k in morph if k != j]
                stat = lambda idx: (fp[idx][:, o_abn].any(1).mean(), fp[idx][:, o_mor].any(1).mean())
                obs = stat(missed)
                pool = np.where((y[:, j] == 0) & abnormal_rec)[0]
                null = np.array([stat(rng.choice(pool, len(missed), replace=False)) for _ in range(N_NULL)])
                rows.append(dict(cohort=ds, arch=arch, seed=seed, lead=L, cls=C.CLASSES[j],
                                 group="morphology" if C.CLASSES[j] in MORPH else "rhythm",
                                 n_missed=len(missed),
                                 any_fp=obs[0], any_fp_null=null[:, 0].mean(),
                                 any_fp_p=(1 + (null[:, 0] >= obs[0]).sum()) / (N_NULL + 1),
                                 morph_fp=obs[1], morph_fp_null=null[:, 1].mean(),
                                 morph_fp_p=(1 + (null[:, 1] >= obs[1]).sum()) / (N_NULL + 1)))
    return rows


def main():
    rng = np.random.default_rng(C.SEED)
    rows = []
    for ds, arch in itertools.product(C.COHORTS, ["seresnet", "inceptiontime"]):
        rows += _cells(ds, arch, rng)
    df = pd.DataFrame(rows)
    atomic_save(OUT / "confusion_cells.csv", lambda t: df.to_csv(t, index=False, float_format="%.4f"))

    # one row per cell (seeds averaged), then per cohort x arch
    cell = df.groupby(["cohort", "arch", "lead", "cls", "group"], as_index=False).agg(
        n_missed=("n_missed", "mean"), any_fp=("any_fp", "mean"), any_fp_null=("any_fp_null", "mean"),
        morph_fp=("morph_fp", "mean"), morph_fp_null=("morph_fp_null", "mean"),
        any_fp_p=("any_fp_p", "median"))
    cell["excess_any"] = cell.any_fp - cell.any_fp_null
    cell["excess_morph"] = cell.morph_fp - cell.morph_fp_null
    summ = cell.groupby(["cohort", "arch"], as_index=False).agg(
        cells=("lead", "size"), any_fp=("any_fp", "mean"), any_fp_null=("any_fp_null", "mean"),
        excess_any=("excess_any", "mean"), morph_fp=("morph_fp", "mean"),
        morph_fp_null=("morph_fp_null", "mean"), excess_morph=("excess_morph", "mean"),
        cells_above_null=("any_fp_p", lambda s: int((s < 0.05).sum())))
    atomic_save(OUT / "confusion_summary.csv", lambda t: summ.to_csv(t, index=False, float_format="%.3f"))
    print(summ.to_string(index=False, float_format="%.3f"))
    return summ


if __name__ == "__main__":
    main()
