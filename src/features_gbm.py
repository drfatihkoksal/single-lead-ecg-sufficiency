"""Non-deep-learning baseline: hand-crafted per-lead ECG features + LightGBM.

If the same lead x class blind-spot map appears with a feature-based classifier
that shares nothing with the CNNs, the map reflects lead content, not a model.

Per (record, lead), from a 0.5-40 Hz band-passed signal:
  - rhythm: RR mean/SD/CV/min/max, RMSSD, pNN50, beat count, successive-RR
    irregularity (AF), beat-to-template correlation (morphology variability)
  - median-beat template (-250..+450 ms around R), resampled to 50 Hz (35 values)
  - template measurements: R and S amplitude, QRS peak-to-peak and width,
    ST level at J+40/J+80 ms, T-wave max/min/extreme/area, P-wave amplitude
A single-lead model sees that lead's vector; the 12-lead ceiling sees all twelve.
Outputs mirror train.py (pred_/metrics_/norm_ files) so evaluate.py works as-is.

Run:  python -m src.features_gbm --dataset chapman --seed 1337
"""
import argparse
import json
from multiprocessing import Pool

import lightgbm as lgb
import neurokit2 as nk
import numpy as np
from scipy.signal import butter, sosfiltfilt
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from . import config as C
from .train import atomic_save, lead_ids_for

FS = C.FS
_SOS = butter(3, [0.5, 40], btype="band", fs=FS, output="sos")
PRE, POST = int(0.25 * FS), int(0.45 * FS)      # template window around R
R = PRE                                         # R index inside the template
ms = lambda t: int(round(t * FS / 1000))

RR_NAMES = ["rr_mean", "rr_sd", "rr_cv", "rr_min", "rr_max", "rmssd", "pnn50",
            "n_beats", "rr_irreg", "beat_corr"]
MEAS_NAMES = ["r_amp", "s_amp", "qrs_p2p", "qrs_width", "st40", "st80",
              "t_max", "t_min", "t_ext", "t_area", "p_amp"]
N_TPL = 35
N_FEAT = len(RR_NAMES) + len(MEAS_NAMES) + N_TPL


def _lead_features(x):
    f = np.full(N_FEAT, np.nan, dtype=np.float32)
    try:
        x = sosfiltfilt(_SOS, x)
        _, info = nk.ecg_peaks(x, sampling_rate=FS)
        rp = np.asarray(info["ECG_R_Peaks"])
    except Exception:
        return f
    rp = rp[(rp >= PRE) & (rp < len(x) - POST)]
    if len(rp) < 3:
        return f
    rr = np.diff(rp) / FS * 1000
    d = np.abs(np.diff(rr))
    beats = np.stack([x[r - PRE:r + POST] for r in rp])
    tpl = np.median(beats, axis=0)
    corr = np.mean([np.corrcoef(b, tpl)[0, 1] for b in beats])
    f[:len(RR_NAMES)] = [rr.mean(), rr.std(), rr.std() / rr.mean(), rr.min(), rr.max(),
                         np.sqrt(np.mean(np.diff(rr) ** 2)) if len(rr) > 1 else np.nan,
                         np.mean(d > 50) if len(d) else np.nan, len(rp),
                         np.median(d) / rr.mean() if len(d) else np.nan, corr]

    base = np.median(tpl[R - ms(120):R - ms(60)])        # PR segment
    t = tpl - base
    qrs = t[R - ms(60):R + ms(60)]
    g = np.abs(np.gradient(t[R - ms(120):R + ms(120)]))
    on = np.where(g > 0.15 * g.max())[0]
    j = R - ms(120) + on[-1] if len(on) else R + ms(40)   # J point estimate
    tw = t[R + ms(150):R + ms(400)]
    f[len(RR_NAMES):len(RR_NAMES) + len(MEAS_NAMES)] = [
        qrs.max(), qrs.min(), qrs.max() - qrs.min(),
        (on[-1] - on[0]) / FS * 1000 if len(on) else np.nan,
        t[min(j + ms(40), len(t) - 1)], t[min(j + ms(80), len(t) - 1)],
        tw.max(), tw.min(), tw[np.argmax(np.abs(tw))], tw.sum() / FS,
        t[R - ms(250):R - ms(120)].max()]
    f[-N_TPL:] = t[::FS // 50][:N_TPL]                    # 50 Hz template
    return f


def _record_features(sig):
    return np.stack([_lead_features(sig[i].astype(np.float64)) for i in range(sig.shape[0])])


def _worker(args):
    path, idx = args
    sig = np.load(path, mmap_mode="r")
    return np.stack([_record_features(sig[r]) for r in idx])


def extract(dataset, n_jobs=30):
    P = C.ds_paths(dataset, "gbm", 0)
    out = C.ART_V2 / dataset / "features" / "features.npy"
    if out.exists():
        return np.load(out)
    path = P.cache / "signals.npy"
    n = np.load(path, mmap_mode="r").shape[0]
    chunks = np.array_split(np.arange(n), n_jobs * 8)
    with Pool(n_jobs) as pool:
        parts = pool.map(_worker, [(path, c) for c in chunks])
    feats = np.concatenate(parts).astype(np.float32)       # (N, 12, N_FEAT)
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_save(out, lambda t: np.save(t, feats))
    print(f"[{dataset}] features {feats.shape}, NaN fraction {np.isnan(feats).mean():.3f}")
    return feats


def train_one(spec, feats, dataset="chapman", seed=C.SEED):
    P = C.ds_paths(dataset, "gbm", seed)
    lab = np.load(P.cache / "labels.npy")
    sp = np.load(P.splits / "split.npz")
    tr, va, te = sp["train"], sp["val"], sp["test"]
    X = feats[:, lead_ids_for(spec), :].reshape(len(feats), -1)
    pv = np.zeros((len(va), C.N_CLASSES)); pt = np.zeros((len(te), C.N_CLASSES))
    for j in range(C.N_CLASSES):
        y = lab[:, j]
        pos = y[tr].sum()
        clf = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.03, num_leaves=31,
                                 min_child_samples=10, subsample=0.8, subsample_freq=1,
                                 colsample_bytree=0.5, reg_lambda=1.0,
                                 scale_pos_weight=(len(tr) - pos) / max(pos, 1),
                                 random_state=seed, n_jobs=16, verbose=-1)
        clf.fit(X[tr], y[tr], eval_X=(X[va],), eval_y=(y[va],), eval_metric="average_precision",
                callbacks=[lgb.early_stopping(100, verbose=False)])
        pv[:, j] = clf.predict_proba(X[va])[:, 1]
        pt[:, j] = clf.predict_proba(X[te])[:, 1]

    # identical threshold tuning and outputs to train.py
    yv, yt = lab[va], lab[te]
    grid = np.linspace(0.05, 0.95, 19)
    thr = np.array([grid[int(np.argmax([f1_score(yv[:, j], (pv[:, j] >= t).astype(int),
                                                 zero_division=0) for t in grid]))]
                    for j in range(C.N_CLASSES)])
    res = {"lead": spec, "thresholds": thr.tolist(), "per_class": {}}
    for j, c in enumerate(C.CLASSES):
        yj, pj = yt[:, j], pt[:, j]
        res["per_class"][c] = {
            "auprc": float(average_precision_score(yj, pj)) if yj.sum() else None,
            "auroc": float(roc_auc_score(yj, pj)) if 0 < yj.sum() < len(yj) else None,
            "f1": float(f1_score(yj, (pj >= thr[j]).astype(int), zero_division=0)),
            "n_pos": int(yj.sum()),
        }
    P.models.mkdir(parents=True, exist_ok=True)
    P.metrics.mkdir(parents=True, exist_ok=True)
    atomic_save(P.models / f"norm_{spec}.npz", lambda t: np.savez(t, thr=thr))
    atomic_save(P.metrics / f"metrics_{spec}.json", lambda t: t.write_text(json.dumps(res, indent=2)))
    atomic_save(P.metrics / f"pred_{spec}.npz",            # completion marker, written last
                lambda t: np.savez(t, p=pt, y=yt, test_idx=te))
    macro = np.mean([v["auprc"] for v in res["per_class"].values() if v["auprc"] is not None])
    print(f"[gbm/{dataset}/{spec}] test macro-AUPRC={macro:.4f}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="chapman")
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--extract-only", action="store_true")
    a = ap.parse_args()
    feats = extract(a.dataset)
    if not a.extract_only:
        from .evaluate import main as evaluate
        P = C.ds_paths(a.dataset, "gbm", a.seed)
        for spec in list(C.LEADS) + ["ALL"]:
            if not (P.metrics / f"pred_{spec}.npz").exists():      # resumable
                train_one(spec, feats, a.dataset, a.seed)
        if not (P.metrics / "blindspot.npz").exists():
            evaluate(a.dataset, "gbm", a.seed)
