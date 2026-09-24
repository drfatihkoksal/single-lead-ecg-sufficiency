"""Resubmission experiment grid: cohort x architecture x seed x lead spec.

Generalizes run_all.py.  Every run writes to artifacts_v2/<ds>/<arch>/seed<k>/ and
is skipped when its test predictions already exist, so the grid is resumable.
Interrupted training resumes from its checkpoint (see train.train_one).
Model weights are kept only for the first seed (weights are not needed for the
analyses and xresnet1d101 is ~130 MB per model).

Run:  python -m src.run_grid --datasets chapman --archs seresnet --seeds 1337
      python -m src.run_grid --datasets chapman ptbxl --archs seresnet inceptiontime \
             xresnet1d101 --seeds 1337 1 2 3 4 --devices
"""
import argparse
import json
import time

from . import config as C
from .evaluate import main as evaluate
from .train import atomic_save, train_one

SEEDS = [1337, 1, 2, 3, 4]


def specs_for(devices):
    specs = list(C.LEADS) + ["ALL"]
    if devices:
        specs += [s for s in C.DEVICE_SPECS if s not in specs]
    return specs


def run(datasets, archs, seeds, devices=False, epochs=C.EPOCHS, skip_eval=False):
    """Seed-outer order: the first seed covers every cohort x architecture before any
    replicate seed starts, so early results are already complete in breadth."""
    t0 = time.time()
    for k, seed in enumerate(seeds):
        for ds in datasets:
            for arch in archs:
                P = C.ds_paths(ds, arch, seed)
                summary = {}
                for spec in specs_for(devices):
                    if not (P.metrics / f"pred_{spec}.npz").exists():
                        print(f"\n=== [{ds}/{arch}/seed{seed}] {spec} ({(time.time()-t0)/60:.1f} min) ===",
                              flush=True)
                        train_one(spec, epochs=epochs, dataset=ds, arch=arch, seed=seed,
                                  save_model=(k == 0))
                    res = json.load(open(P.metrics / f"metrics_{spec}.json"))
                    v = [c["auprc"] for c in res["per_class"].values() if c["auprc"] is not None]
                    summary[spec] = round(sum(v) / len(v), 4)
                atomic_save(P.metrics / "summary_macro_auprc.json",
                            lambda t: t.write_text(json.dumps(summary, indent=2)))
                if not skip_eval and not (P.metrics / "blindspot.npz").exists():
                    evaluate(ds, arch, seed)
    print(f"\nGRID DONE in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["chapman"])
    ap.add_argument("--archs", nargs="+", default=["seresnet"])
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--devices", action="store_true", help="also train device lead sets")
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--skip-eval", action="store_true")
    a = ap.parse_args()
    run(a.datasets, a.archs, a.seeds, a.devices, a.epochs, a.skip_eval)
