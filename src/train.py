"""Train one model for a given lead specification.

A "lead spec" is either a single lead name (e.g. "V1") or "ALL" for the 12-lead
ceiling model. Same protocol/seed everywhere so differences trace to lead content.

Run a single lead:   python -m src.train --lead V1
Ceiling model:       python -m src.train --lead ALL
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

from . import config as C
from .model import build_model


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ECGData(Dataset):
    def __init__(self, sig, lab, idx, lead_ids, mean, std, train=False):
        self.sig, self.lab, self.idx = sig, lab, idx
        self.lead_ids = lead_ids
        self.mean, self.std = mean, std          # (Csel,1)
        self.train = train

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        r = self.idx[i]
        x = self.sig[r][self.lead_ids].astype(np.float32)      # (Csel, 5000)
        x = (x - self.mean) / self.std
        if self.train:
            # light augmentation: amplitude scale + small time shift
            x = x * np.random.uniform(0.9, 1.1)
            s = np.random.randint(-50, 51)
            if s:
                x = np.roll(x, s, axis=1)
        y = self.lab[r].astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)


# checkpoint at most this often (an xresnet1d101 checkpoint is ~0.5 GB)
CKPT_EVERY_S = float(os.environ.get("CKPT_EVERY_S", 120))


def atomic_save(path, write):
    """write(tmp_path) then rename, so a killed run never leaves a half-written file.
    The temp name keeps the suffix because np.savez appends '.npz' otherwise."""
    tmp = path.with_name(f".tmp_{path.name}")
    write(tmp)
    os.replace(tmp, path)


def lead_ids_for(spec):
    if spec == "ALL":
        return list(range(C.N_LEADS))
    names = C.LEAD_SETS.get(spec, spec.split("+"))
    return [C.lead_index(n) for n in names]


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ps, ys = [], []
    for x, y in loader:
        p = torch.sigmoid(model(x.to(device))).cpu().numpy()
        ps.append(p); ys.append(y.numpy())
    return np.concatenate(ps), np.concatenate(ys)


def macro_auprc(y, p):
    scores = []
    for j in range(y.shape[1]):
        if y[:, j].sum() > 0:
            scores.append(average_precision_score(y[:, j], p[:, j]))
    return float(np.mean(scores))


def train_one(spec, epochs=C.EPOCHS, quick=False, dataset="chapman", arch=None, seed=None,
              save_model=True):
    """arch=None reproduces the study-1 run (SE-ResNet, seed 1337, artifacts/ layout);
    any arch writes to artifacts_v2/<dataset>/<arch>/seed<k>/.

    Resumable: v2 runs checkpoint (model, optimizer, scheduler, scaler, early-stop
    state, RNG) to models/ckpt_<spec>.pt and continue from it after an interruption.
    pred_<spec>.npz is written last and marks the run complete."""
    set_seed(C.SEED if seed is None else seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lead_ids = lead_ids_for(spec)
    P = C.ds_paths(dataset, arch, seed)

    sig = np.load(P.cache / "signals.npy", mmap_mode="r")
    lab = np.load(P.cache / "labels.npy")
    sp = np.load(P.splits / "split.npz")
    tr, va, te = sp["train"], sp["val"], sp["test"]

    # per-lead normalization stats from TRAIN only
    sample = sig[tr[:2000]][:, lead_ids, :].astype(np.float32)
    mean = sample.mean(axis=(0, 2), keepdims=True)[0]      # (Csel,1)
    std = sample.std(axis=(0, 2), keepdims=True)[0] + 1e-6

    mk = lambda idx, train: DataLoader(
        ECGData(sig, lab, idx, lead_ids, mean, std, train=train),
        batch_size=C.BATCH_SIZE, shuffle=train, num_workers=8,
        pin_memory=True, drop_last=False, persistent_workers=True)
    dl_tr, dl_va, dl_te = mk(tr, True), mk(va, False), mk(te, False)

    model = build_model(arch or "seresnet", len(lead_ids), C.N_CLASSES).to(device)
    pos = lab[tr].sum(0); neg = len(tr) - pos
    pos_weight = torch.tensor(neg / np.maximum(pos, 1), dtype=torch.float32, device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda")

    best, best_state, bad, start = -1.0, None, 0, 0
    ckpt = P.models / f"ckpt_{spec}.pt" if arch is not None else None
    if ckpt is not None and ckpt.exists():
        ck = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        best, best_state, bad, start = ck["best"], ck["best_state"], ck["bad"], ck["epoch"]
        torch.set_rng_state(ck["rng_torch"].cpu()); torch.cuda.set_rng_state(ck["rng_cuda"].cpu())
        np.random.set_state(ck["rng_np"])
        print(f"[{spec}] resumed from epoch {start} (best={best:.4f})")
    P.models.mkdir(parents=True, exist_ok=True)
    stop = bad >= C.PATIENCE
    t0 = t_ck = time.time()
    for ep in range(start, epochs):
        if stop:
            break
        model.train()
        for x, y in dl_tr:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad()
            with torch.amp.autocast("cuda"):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
        sched.step()
        pv, yv = predict(model, dl_va, device)
        m = macro_auprc(yv, pv)
        if m > best:
            best, best_state, bad = m, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
        print(f"[{spec}] ep{ep+1:02d} val_mAUPRC={m:.4f} best={best:.4f} ({time.time()-t0:.0f}s)")
        stop = bad >= C.PATIENCE
        if ckpt is not None and (time.time() - t_ck > CKPT_EVERY_S or stop or ep + 1 == epochs):
            state = dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                         scaler=scaler.state_dict(), best=best, best_state=best_state, bad=bad,
                         epoch=ep + 1, rng_torch=torch.get_rng_state(),
                         rng_cuda=torch.cuda.get_rng_state(), rng_np=np.random.get_state())
            atomic_save(ckpt, lambda t: torch.save(state, t))
            t_ck = time.time()
        if bad >= C.PATIENCE or quick:
            if quick and ep + 1 >= 2:
                break
            if bad >= C.PATIENCE:
                print(f"[{spec}] early stop"); break

    model.load_state_dict(best_state)

    # tune per-class thresholds on val (maximize F1), then evaluate on test
    pv, yv = predict(model, dl_va, device)
    thr = np.zeros(C.N_CLASSES)
    grid = np.linspace(0.05, 0.95, 19)
    for j in range(C.N_CLASSES):
        f1s = [f1_score(yv[:, j], (pv[:, j] >= t).astype(int), zero_division=0) for t in grid]
        thr[j] = grid[int(np.argmax(f1s))]

    pt, yt = predict(model, dl_te, device)
    res = {"lead": spec, "val_mAUPRC": best, "thresholds": thr.tolist(), "per_class": {}}
    for j, c in enumerate(C.CLASSES):
        yj, pj = yt[:, j], pt[:, j]
        res["per_class"][c] = {
            "auprc": float(average_precision_score(yj, pj)) if yj.sum() else None,
            "auroc": float(roc_auc_score(yj, pj)) if 0 < yj.sum() < len(yj) else None,
            "f1": float(f1_score(yj, (pj >= thr[j]).astype(int), zero_division=0)),
            "n_pos": int(yj.sum()),
        }

    P.metrics.mkdir(parents=True, exist_ok=True)
    if save_model:
        atomic_save(P.models / f"model_{spec}.pt", lambda t: torch.save(best_state, t))
    atomic_save(P.models / f"norm_{spec}.npz", lambda t: np.savez(t, mean=mean, std=std, thr=thr))
    atomic_save(P.metrics / f"metrics_{spec}.json",
                lambda t: t.write_text(json.dumps(res, indent=2)))
    # raw test predictions (bootstrap CIs later); written last = completion marker
    atomic_save(P.metrics / f"pred_{spec}.npz", lambda t: np.savez(t, p=pt, y=yt, test_idx=te))
    if ckpt is not None and ckpt.exists():
        ckpt.unlink()
    macro = np.mean([v["auprc"] for v in res["per_class"].values() if v["auprc"] is not None])
    print(f"[{spec}] DONE test macro-AUPRC={macro:.4f}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lead", required=True, help="lead name (e.g. V1) or ALL")
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--quick", action="store_true", help="2-epoch smoke test")
    ap.add_argument("--dataset", default="chapman", help="chapman | ningbo | ptbxl | georgia")
    ap.add_argument("--arch", default=None, help="omit for the study-1 run; else see model.ARCHS")
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    train_one(a.lead, epochs=a.epochs, quick=a.quick, dataset=a.dataset, arch=a.arch, seed=a.seed)
