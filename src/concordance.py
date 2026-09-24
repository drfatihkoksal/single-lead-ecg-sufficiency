"""Cross-dataset replication: do the SAME (lead x class) cells go blind in PTB-XL
as in Chapman-Shaoxing?  This is the answer to the single-source weakness.

Each dataset has its own 12 single-lead models + own 12-lead ceiling + own
gap matrix (blindspot.npz).  We never merge data; we compare the two independent
maps.  Reports:
  - overall correlation of the two gap matrices
  - per-cell agreement on blind / not-blind (Cohen's kappa)
  - focus on the lead-specific morphology classes (LVH, RBBB, LAD, STTC, TWC)
  - Fig4 side-by-side heatmaps + Fig5 concordance scatter

Run (after both sweeps + `evaluate` on both):  python -m src.concordance
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

from . import config as C

MORPH = ["TWC", "LAD", "STTC", "LVH", "RBBB"]   # the lead-specific classes


def _load(dataset):
    d = np.load(C.ds_paths(dataset).metrics / "blindspot.npz", allow_pickle=True)
    return {k: d[k] for k in d.files}


def cohen_kappa(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n = len(a)
    po = np.mean(a == b)
    pa1, pb1 = a.mean(), b.mean()
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def main():
    ch, pt = _load("chapman"), _load("ptbxl")
    classes = [str(x) for x in ch["classes"]]
    leads = [str(x) for x in ch["leads"]]
    gch, gpt = ch["gap"], pt["gap"]
    bch = ch["taxonomy"] != "ok"        # blind-spot boolean masks
    bpt = pt["taxonomy"] != "ok"

    # flatten valid cells
    flat = np.isfinite(gch) & np.isfinite(gpt)
    x, y = gch[flat], gpt[flat]
    pr, pp = pearsonr(x, y)
    sr, sp = spearmanr(x, y)
    kappa = cohen_kappa(bch[flat], bpt[flat])
    agree = np.mean(bch[flat] == bpt[flat])

    print("=== Cross-dataset concordance (Chapman vs PTB-XL) ===")
    print(f"cells compared: {int(flat.sum())}")
    print(f"gap Pearson r = {pr:.3f} (p={pp:.1e})   Spearman = {sr:.3f}")
    print(f"blind/not-blind agreement = {agree:.2%}   Cohen kappa = {kappa:.3f}")
    print(f"blind cells: Chapman={int(bch.sum())}  PTB-XL={int(bpt.sum())}  "
          f"both={int((bch&bpt).sum())}")

    print("\nper-class: leads blind in each dataset (morphology focus)")
    print(f"{'class':6s} {'Chapman blind':28s} {'PTB-XL blind':28s} overlap")
    rows = []
    for j, c in enumerate(classes):
        lc = [leads[i] for i in range(len(leads)) if bch[i, j]]
        lp = [leads[i] for i in range(len(leads)) if bpt[i, j]]
        ov = sorted(set(lc) & set(lp))
        tag = " *" if c in MORPH else ""
        print(f"{c:6s}{tag:2s} {','.join(lc) or '-':28s} {','.join(lp) or '-':28s} "
              f"{len(ov)}/{len(set(lc)|set(lp))}")
        rows.append((c, lc, lp, ov))

    # ---- Fig 4: side-by-side gap heatmaps ----
    vmax = max(0.3, np.nanmax([np.nanmax(gch), np.nanmax(gpt)]))
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    for ax, g, title in [(axes[0], gch, "Chapman-Shaoxing (n=9,194)"),
                         (axes[1], gpt, "PTB-XL (n=21,837)")]:
        im = ax.imshow(g, aspect="auto", cmap="magma_r", vmin=0, vmax=vmax)
        ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
        ax.set_title(title)
        for i in range(len(leads)):
            for j in range(len(classes)):
                if np.isfinite(g[i, j]):
                    ax.text(j, i, f"{g[i,j]:.2f}", ha="center", va="center",
                            fontsize=5, color="white" if g[i, j] > 0.15 else "black")
    axes[0].set_yticks(range(len(leads)), leads)
    fig.colorbar(im, ax=axes, label="AUPRC gap to 12-lead ceiling", shrink=0.8)
    fig.suptitle("Blind-spot map replicates across two independent datasets", fontsize=14)
    out4 = C.FIGURES / "fig4_crossdataset_heatmaps.png"
    fig.savefig(out4, dpi=300, bbox_inches="tight")
    print("\nwrote", out4)

    # ---- Fig 5: concordance scatter (gap_chapman vs gap_ptbxl per cell) ----
    fig, ax = plt.subplots(figsize=(7, 7))
    morph_col = {"TWC": "#2ca02c", "LAD": "#d62728", "STTC": "#bcbd22",
                 "LVH": "#1f77b4", "RBBB": "#9467bd"}
    cmap = {c: morph_col.get(c, "0.6") for c in classes}
    for j, c in enumerate(classes):
        m = np.isfinite(gch[:, j]) & np.isfinite(gpt[:, j])
        big = c in MORPH
        ax.scatter(gch[m, j], gpt[m, j], s=70 if big else 30,
                   color=cmap[c], edgecolor="k" if big else "none",
                   lw=0.6, alpha=0.9 if big else 0.5,
                   label=c if big else None, zorder=3 if big else 2)
    lim = [min(x.min(), y.min()) - 0.05, vmax + 0.02]
    ax.plot(lim, lim, "k--", lw=1, label="y = x")
    ax.axhline(0.15, color="green", ls=":", lw=1)
    ax.axvline(0.15, color="green", ls=":", lw=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("AUPRC gap — Chapman-Shaoxing")
    ax.set_ylabel("AUPRC gap — PTB-XL")
    ax.set_title(f"Per-cell blind-spot concordance\nPearson r={pr:.2f}, "
                 f"kappa={kappa:.2f} (morphology classes highlighted)")
    ax.legend(loc="upper left", fontsize=8, title="lead-specific classes")
    fig.tight_layout()
    out5 = C.FIGURES / "fig5_concordance_scatter.png"
    fig.savefig(out5, dpi=300)
    print("wrote", out5)

    # save summary
    import json
    summary = dict(pearson_r=pr, pearson_p=pp, spearman_r=sr, kappa=kappa,
                   agreement=float(agree),
                   blind_chapman=int(bch.sum()), blind_ptbxl=int(bpt.sum()),
                   blind_both=int((bch & bpt).sum()),
                   per_class={c: dict(chapman=lc, ptbxl=lp, overlap=ov)
                              for c, lc, lp, ov in rows})
    with open(C.METRICS / "concordance.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("wrote", C.METRICS / "concordance.json")


if __name__ == "__main__":
    main()
