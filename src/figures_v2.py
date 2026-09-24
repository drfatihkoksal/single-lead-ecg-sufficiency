"""Manuscript figures for the resubmission (from artifacts_v2/aggregate/).

Fig. 2  sufficiency map: class x lead configuration, one panel per cohort
        (single leads | reduced lead sets); sufficient / indeterminate / loss
Fig. 3  reproducibility: single-lead AUPRC gaps, SE-ResNet vs InceptionTime and
        deep learning vs feature-based model; correlation between cohorts

Colours follow the dataviz reference palette: diverging blue (sufficient) / red (loss)
with the neutral grey midpoint (indeterminate), validated for colour-vision deficiency;
cells also carry a glyph so the state never depends on colour alone. Cohort identity in
Fig. 3 uses categorical slots 1-4 plus a distinct marker shape per cohort.

Run:  python -m src.figures_v2            -> figures/v2/*.pdf, *.png
      (default --arch all: Fig. 2 for SE-ResNet, and InceptionTime / feature-based
      variants as Supplementary Figs S1 / S2, plus Fig. 3)
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Rectangle
from scipy.stats import pearsonr

from . import config as C

AGG = C.ART_V2 / "aggregate"
OUT = C.FIGURES / "v2"

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
STATE = {"sufficient": ("#2a78d6", "✓"), "indeterminate": ("#f0efec", ""),
         "loss": ("#e34948", "✕"), "na": ("#ffffff", "–")}
COHORT = {"chapman": ("Chapman", "#2a78d6", "o"), "ningbo": ("Ningbo", "#eb6834", "s"),
          "georgia": ("Georgia", "#1baf7a", "^"), "ptbxl_snomed": ("PTB-XL", "#eda100", "D")}
GROUPS = [("Rhythm", ["SR", "SB", "ST", "AF", "AFL", "SVT"]), ("Axis", ["LAD"]),
          ("Conduction", ["RBBB"]), ("Hypertrophy", ["LVH"]), ("Repolarization", ["TWC", "STTC"])]
DEVICES = [("I+II", "I, II"), ("LIMB6", "Limb 6"), ("CH3", "I, II, V2"), ("CH4", "I, II, III, V2")]

plt.rcParams.update({"font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "pdf.fonttype": 42, "svg.fonttype": "none"})


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", OUT / name)


def fig2(arch="seresnet"):
    cells = pd.read_csv(AGG / "cells.csv")
    cells = cells[cells.arch == arch]
    cohorts = [ds for ds in COHORT if (cells.cohort == ds).any()]
    has_dev = (cells.kind == "device").any()
    cols = list(C.LEADS) + ([None] + [d for d, _ in DEVICES] if has_dev else [])
    xlab = list(C.LEADS) + ([""] + [lab for _, lab in DEVICES] if has_dev else [])
    rows, ypos, y = [], [], 0.0
    for gname, classes in GROUPS:
        for cl in classes:
            rows.append((gname, cl)); ypos.append(y); y += 1
        y += 0.45                                          # gap between diagnostic groups
    nr = 2 if len(cohorts) > 2 else 1
    fig, axes = plt.subplots(nr, 2, figsize=(7.1, 3.1 * nr + 0.4 * (nr == 1)))
    for ax, ds in zip(np.atleast_1d(axes).ravel(), cohorts):
        g = cells[cells.cohort == ds]
        st = {(r.spec, r.cls): r.sufficiency for r in g.itertuples()}
        low = set(g[g.low_ceiling & (g.kind == "ceiling")].cls)
        for (gname, cl), yy in zip(rows, ypos):
            for xi, sp in enumerate(cols):
                if sp is None:
                    continue
                s = st.get((sp, cl), "na")
                s = "na" if not isinstance(s, str) else s
                face, glyph = STATE[s]
                ax.add_patch(Rectangle((xi + 0.06, yy + 0.06), 0.88, 0.88, facecolor=face,
                                       edgecolor="#d9d8d4" if s == "na" else "none", lw=0.4))
                if glyph:
                    ax.text(xi + 0.5, yy + 0.53, glyph, ha="center", va="center", fontsize=5.5,
                            color="white" if s in ("sufficient", "loss") else MUTED)
        ax.set_xlim(-0.1, len(cols) + 0.1); ax.set_ylim(y - 0.35, -1.05)
        ax.set_xticks([i + 0.5 for i, c in enumerate(cols) if c is not None])
        ax.set_xticklabels([l for l, c in zip(xlab, cols) if c is not None], rotation=90)
        ax.set_yticks([p + 0.5 for p in ypos])
        ax.set_yticklabels([cl + ("†" if cl in low else "") for _, cl in rows])
        ax.tick_params(length=0)
        for sp_ in ax.spines.values():
            sp_.set_visible(False)
        ax.set_title(COHORT[ds][0], loc="left", color=INK, fontweight="bold", pad=4)
        ax.text(6, -0.45, "Single leads", ha="center", fontsize=6.5, color=INK2)
        if has_dev:
            ax.text(len(C.LEADS) + 3, -0.45, "Reduced lead sets", ha="center", fontsize=6.5, color=INK2)
        yg = 0.0
        for gname, classes in GROUPS:                       # group labels on the right
            ax.text(len(cols) + 0.25, yg + len(classes) / 2, gname, va="center", fontsize=6,
                    color=MUTED)
            yg += len(classes) + 0.45
    handles = [Patch(facecolor=STATE[k][0], edgecolor="#d9d8d4" if k == "na" else "none",
                     label=lab) for k, lab in (("sufficient", "Sufficient (✓)"),
                                              ("indeterminate", "Indeterminate"),
                                              ("loss", "Loss (✕)"), ("na", "Not available"))]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=7,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.03 if nr == 2 else 0.08, 1, 1), w_pad=2.5, h_pad=1.5)
    _save(fig, f"fig2_sufficiency_{arch}")


def fig3():
    cells = pd.read_csv(AGG / "cells.csv")
    s = cells[cells.kind == "single"].dropna(subset=["gap"])
    wide = s.pivot_table(index=["cohort", "spec", "cls"], columns="arch", values="gap").reset_index()
    cc = pd.read_csv(AGG / "concordance_cohort.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.55), gridspec_kw={"width_ratios": [1, 1, 0.95]})

    def scatter(ax, xcol, ycol, cohorts, xlabel, ylabel, title):
        lo, hi = -0.15, 0.8
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=0)
        for ds in cohorts:
            d = wide[(wide.cohort == ds)].dropna(subset=[xcol, ycol])
            name, col, mk = COHORT[ds]
            r = pearsonr(d[xcol], d[ycol])[0]
            ax.scatter(d[xcol], d[ycol], s=11, marker=mk, facecolor=col, edgecolor="white",
                       linewidth=0.3, alpha=0.85, label=f"{name}  r = {r:.2f}")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", color=INK, fontweight="bold")
        ax.grid(color="#ecebe8", lw=0.5); ax.set_axisbelow(True)
        for sp_ in ("top", "right"):
            ax.spines[sp_].set_visible(False)
        ax.legend(frameon=False, fontsize=6, loc="upper left", handletextpad=0.2, borderaxespad=0.1)

    scatter(axes[0], "seresnet", "inceptiontime", list(COHORT), "AUPRC gap, SE-ResNet",
            "AUPRC gap, InceptionTime", "a  Architectures")
    scatter(axes[1], "seresnet", "gbm", ["chapman", "ptbxl_snomed"], "AUPRC gap, SE-ResNet",
            "AUPRC gap, feature-based model", "b  Deep learning vs features")

    ax = axes[2]
    order = list(COHORT)
    m = np.full((4, 4), np.nan)
    sub = cc[(cc.arch == "seresnet") & (cc.classes == "morphology")]
    for r in sub.itertuples():
        i, j = order.index(r.cohort_a), order.index(r.cohort_b)
        m[max(i, j), min(i, j)] = r.r
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blue", ["#eaf2fc", "#2a78d6", "#0d3b73"])
    for i in range(4):
        for j in range(i):
            v = m[i, j]
            ax.add_patch(Rectangle((j + 0.04, i + 0.04), 0.92, 0.92, facecolor=cmap((v - 0.5) / 0.5)))
            ax.text(j + 0.5, i + 0.5, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if v > 0.65 else INK)
    ax.set_xlim(0, 3); ax.set_ylim(4, 1)
    ax.set_xticks([0.5, 1.5, 2.5]); ax.set_xticklabels([COHORT[d][0] for d in order[:3]])
    ax.set_yticks([1.5, 2.5, 3.5]); ax.set_yticklabels([COHORT[d][0] for d in order[1:]])
    ax.tick_params(length=0)
    for sp_ in ax.spines.values():
        sp_.set_visible(False)
    ax.set_aspect("equal")
    ax.set_title("c  Cohorts (non-rhythm, r)", loc="left", color=INK, fontweight="bold")
    fig.tight_layout(w_pad=1.2)
    _save(fig, "fig3_reproducibility")


def fig1():
    """Study design: cohorts -> lead configurations -> models -> sufficiency criterion."""
    from matplotlib.patches import FancyBboxPatch
    fig = plt.figure(figsize=(7.1, 2.9))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 42); ax.axis("off")

    def box(x, y, w, h, title, lines, face="#f6f5f2"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.8",
                                    facecolor=face, edgecolor="#d9d8d4", lw=0.6))
        ax.text(x + 1, y + h - 1.6, title, fontsize=7, fontweight="bold", color=INK, va="top")
        for k, ln in enumerate(lines):
            ax.text(x + 1, y + h - 5.0 - k * 2.7, ln, fontsize=6.3, color=INK2, va="top")

    def arrow(x0, x1, y=21):
        ax.annotate("", xy=(x1, y), xytext=(x0, y),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.8, shrinkA=0, shrinkB=0))

    ax.text(1, 40.5, "Four independent cohorts", fontsize=7.5, fontweight="bold", color=INK, va="top")
    for k, (name, where, n) in enumerate([("Chapman", "China", "10 234"), ("Ningbo", "China", "34 829"),
                                           ("Georgia", "United States", "10 292"),
                                           ("PTB-XL", "Germany", "21 837")]):
        yy = 30.5 - k * 8.6
        col, mk = COHORT[list(COHORT)[k]][1:]
        ax.add_patch(FancyBboxPatch((1, yy), 23, 7, boxstyle="round,pad=0.2,rounding_size=0.8",
                                    facecolor="#f6f5f2", edgecolor="#d9d8d4", lw=0.6))
        ax.scatter([3.2], [yy + 3.5], marker=mk, s=18, color=col, zorder=3)
        ax.text(5.2, yy + 4.6, name, fontsize=7, fontweight="bold", color=INK, va="center")
        ax.text(5.2, yy + 2.0, f"{where}; {n} recordings", fontsize=6, color=INK2, va="center")
    arrow(24.6, 26.6)
    box(27, 1.5, 22, 36.5, "Lead configurations", [
        "12 single leads", "  I, II, III, aVR, aVL, aVF,", "  V1 to V6",
        "4 reduced lead sets", "  I+II; six limb leads;", "  I, II, V2; I, II, III, V2",
        "Twelve-lead reference", "", "11 classes in 5 groups", "  rhythm, axis, conduction,",
        "  hypertrophy, repolarization"])
    arrow(49.6, 52.2)
    box(52.6, 1.5, 20.5, 36.5, "Models", [
        "SE-ResNet", "InceptionTime", "  5 seeds each,", "  seed-ensembled", "",
        "Gradient-boosted trees", "  on hand-crafted features", "  (Chapman, PTB-XL)", "",
        "One protocol for every", "configuration"])
    arrow(73.6, 76)

    ax.text(76.5, 40.5, "Sufficiency criterion", fontsize=7.5, fontweight="bold", color=INK, va="top")
    ax.text(76.5, 37.2, "AUPRC gap to the twelve-lead model", fontsize=6.3, color=INK2, va="top")
    ax.text(76.5, 34.6, "with 95% bootstrap interval", fontsize=6.3, color=INK2, va="top")
    x0, x1 = 78, 98                                   # mini axis: gap 0 .. 0.15
    gx = lambda g: x0 + (g + 0.02) / 0.17 * (x1 - x0)
    ax.plot([gx(0), gx(0)], [8, 30], color="#d9d8d4", lw=0.6)
    ax.plot([gx(0.05), gx(0.05)], [8, 30], color=INK2, lw=0.8, ls=(0, (3, 2)))
    ax.text(gx(0.05), 31, "margin 0.05", fontsize=6, color=INK2, ha="center")
    ax.text(gx(0), 6, "0", fontsize=6, color=MUTED, ha="center")
    ax.text(gx(0.15), 6, "0.15", fontsize=6, color=MUTED, ha="center")
    for yy, (lo, pt, hi), state in [(25, (-0.01, 0.01, 0.035), "sufficient"),
                                     (18, (0.02, 0.05, 0.09), "indeterminate"),
                                     (11, (0.07, 0.10, 0.13), "loss")]:
        c = STATE[state][0] if state != "indeterminate" else MUTED
        ax.plot([gx(lo), gx(hi)], [yy, yy], color=c, lw=1.6, solid_capstyle="round")
        ax.scatter([gx(pt)], [yy], s=14, color=c, zorder=3)
        ax.text(x1 + 0.3, yy, {"sufficient": "Sufficient", "indeterminate": "Indeterminate",
                              "loss": "Loss"}[state], fontsize=6.3, color=INK2, va="center", ha="left")
    _save(fig, "fig1_design")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="all", help="seresnet | inceptiontime | gbm | all")
    a = ap.parse_args()
    if a.arch == "all":
        fig1()
        for arch in ("seresnet", "inceptiontime", "gbm"):
            fig2(arch)
        fig3()
    else:
        fig2(a.arch)
        if a.arch == "seresnet":
            fig3()
