# Single-lead ECG sufficiency across diagnoses

Code and aggregate results for the paper *"For which diagnoses is a single electrocardiogram
lead sufficient? A class-level analysis of single- and reduced-lead deep learning across four
independent cohorts"* (under review).

For eleven diagnostic classes (rhythm, axis, conduction, hypertrophy and repolarization) the
study trains models on each of the twelve standard leads, on four device lead sets and on the
full twelve-lead ECG, and asks whether the loss in area under the precision-recall curve
(AUPRC) relative to the twelve-lead model is small enough for the lead to be considered
sufficient. The analysis is repeated in four cohorts (Chapman, Ningbo, Georgia, PTB-XL), with two
deep learning architectures (SE-ResNet, InceptionTime), a gradient-boosted model on hand-crafted
ECG features, and five random seeds.

A lead configuration is **sufficient** for a class when the upper bound of the 95% bootstrap
confidence interval of its AUPRC gap is below 0.05, **insufficient** (loss) when the lower bound
is above 0.05, and **indeterminate** otherwise; margins of 0.02 and 0.10 (absolute) and 5% and
10% (relative) are reported as sensitivity analyses.

## Layout

```
src/config.py            paths, label space, SNOMED CT mapping, lead sets
src/data_prep_snomed.py  Chapman, Ningbo, Georgia, PTB-XL (SNOMED labels): signal cache + split
src/data_prep_ptbxl.py   PTB-XL signal cache and official strat_fold split
src/data_prep.py         iterative multilabel stratification (shared)
src/dedupe.py            Chapman/Ningbo duplicate detection -> artifacts_v2/dedupe/exclusions.csv
src/model.py             SE-ResNet 1D, InceptionTime 1D (XResNet 1D, supplementary)
src/train.py             training (resumable from checkpoints)
src/evaluate.py          per-run AUPRC/AUROC, paired bootstrap, Benjamini-Hochberg
src/run_grid.py          cohort x architecture x seed x lead-configuration grid (skips finished runs)
src/features_gbm.py      NeuroKit2 delineation features + LightGBM baseline
src/aggregate_seeds.py   5-seed ensembles, gap CIs, sufficiency categories, concordance
src/confusion_v2.py      which other class a model predicts when it misses a class
src/figures_v2.py        figures           -> figures/v2/
src/tables_v2.py         main tables       -> submission_cibm/tables.md
src/supplement_v2.py     supplementary     -> submission_cibm/supplementary.md
run_grid.sh              full grid, detached and self-restarting
post_grid.sh             GBM baseline, aggregation and confusion analysis after the grid
run_broad.sh             broad-label sensitivity analysis
artifacts_v2/aggregate/        per-cell results (cells.csv), sufficiency by class, concordance,
                               summary.json and seed-ensemble test-set predictions
artifacts_v2/aggregate_broad/  the same for the broad label definition
artifacts_v2/dedupe/exclusions.csv  records removed as duplicates or malformed
artifacts_v2/<cohort>/cache/   labels (primary and broad), record identifiers, preparation report
artifacts_v2/<cohort>/splits/  training/validation/test indices into the record identifiers
```

## Data

All four cohorts are public on PhysioNet. Place them next to this code:

| Cohort | Source | Location |
|:--|:--|:--|
| Chapman | PhysioNet/CinC Challenge 2021 v1.0.3, `training/chapman_shaoxing` | `challenge2021/training/chapman_shaoxing/` |
| PTB-XL (SNOMED labels) | PhysioNet/CinC Challenge 2021 v1.0.3, `training/ptb-xl` | `challenge2021/training/ptb-xl/` |
| PTB-XL (signals, folds) | PTB-XL v1.0.1 | `../mi_localisation/ptb-xl-...-1.0.1/` or set `PTBXL_ROOT` |
| Ningbo | ECG arrhythmia database v1.0.0 (records above JS10646) | `ningbo/WFDBRecords/` |
| Georgia | PhysioNet/CinC Challenge 2021 v1.0.3, `training/georgia` | `georgia/Georgia/` (signals), `georgia/hea_2021/` (headers) |

The Chapman records of the ECG arrhythmia database duplicate the Challenge Chapman folder, so
Ningbo uses only the later records, and records that appear in both cohorts are removed using
`artifacts_v2/dedupe/exclusions.csv`. The list is included here; `python -m src.dedupe`
regenerates it but also needs the Kaggle Chapman release used in an earlier study.

## Reproducing

```bash
pip install -r requirements.txt

# signal caches and splits
python -m src.data_prep_ptbxl                  # PTB-XL signal cache (shared by ptbxl_snomed)
for ds in chapman ningbo georgia ptbxl_snomed; do
  python -m src.data_prep_snomed --dataset $ds
  python -m src.data_prep_snomed --dataset $ds --broad
done

# training grid: 4 cohorts x 2 architectures x 5 seeds x 17 lead configurations (resumable)
./run_grid.sh          # log: artifacts_v2/grid.log
./post_grid.sh         # GBM baseline, aggregation, confusion analysis
./run_broad.sh         # broad-label sensitivity analysis (SE-ResNet, one seed)

# tables, figures and supplementary material from the aggregate results
python -m src.tables_v2
python -m src.figures_v2
python -m src.supplement_v2
```

The last three commands need neither the data nor the training grid: they run from the
aggregate results, labels and splits included in this repository. On one RTX 5090 the full grid takes roughly half a day.

## Notes

- Raw data, signal arrays, model checkpoints and per-run predictions are not tracked.
- Splits are record-level and multilabel-stratified (70/15/15) for Chapman, Ningbo and Georgia,
  which do not distribute patient identifiers; PTB-XL uses its official patient-disjoint folds
  (1-8 train, 9 validation, 10 test).

## License

MIT
