# -*- coding: utf-8 -*-
# figure_s2_dataset_profile.py
# 用法（按你数据路径替换）：
#   python Figure9.py --train_npz /root/autodl-tmp/tok_train_512.npz --dev_npz   /root/autodl-tmp/tok_dev_512.npz --test_npz  /root/autodl-tmp/tok_test_512.npz --topk_aspects 20 --out S2_dataset_profile

'''
Panels:
 (a) Token length distribution (train/dev/test overlaid, density)
 (b) Aspect sentiment composition (Top-K aspects)  [legend at right]
 (c) Rating distribution (1..5)
 (d) Sentiment vs. rating — scatter colored by per-bin counts (log),
     with Pearson r / Spearman ρ shown horizontally at the top.

Outputs: <out>.png (300dpi), <out>.pdf (vector)

Author: your team
'''

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

# ----------------------------
# Utils (no SciPy dependency)
# ----------------------------
def load_npz(path):
    z = np.load(path, mmap_mode="r")
    ids, attn, lab, star = z["ids"], z["attn"], z["lab"], z["star"]
    return ids, attn, lab, star

def to_eff_len(attn_mask_row):
    # number of effective tokens (mask==1)
    return int(np.sum(attn_mask_row.astype(bool)))

def _rankdata_avg(a: np.ndarray) -> np.ndarray:
    """Average ranks for ties (SciPy-free)."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(a) + 1, dtype=float)

    vals, first_idx, counts = np.unique(a[order], return_index=True, return_counts=True)
    last_idx = first_idx + counts
    for f, l in zip(first_idx, last_idx):
        if l - f > 1:
            avg = (ranks[order][f:l].min() + ranks[order][f:l].max()) / 2.0
            ranks[order][f:l] = avg
    return ranks

def spearman_rho(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    rx = _rankdata_avg(x)
    ry = _rankdata_avg(y)
    rx = (rx - rx.mean()) / (rx.std() + 1e-12)
    ry = (ry - ry.mean()) / (ry.std() + 1e-12)
    return float(np.clip((rx * ry).mean(), -1.0, 1.0))

# ----------------------------
# Data aggregation
# ----------------------------
def gather_lengths(ids, attn):
    # Effective token lengths per sample
    return np.array([to_eff_len(a) for a in attn], dtype=int)

def gather_ratings(star):
    # star 已是 1..5（依据你训练脚本里 Dataset 把 /5.0 放在 dataloader 层）
    return star.astype(int)

def gather_aspect_comp(lab, topk=20):
    """
    lab: [N, A] with {-2, -1, 0, 1}, where -2 = NA
    Return:
      top_idx: indices of top-K aspects by coverage count
      comp:    [K, 3] proportions of (neg, neu, pos) in that order
      ticks:   list of aspect ids to draw on x-axis (1-based)
    """
    N, A = lab.shape
    valid = lab != -2
    counts_per_aspect = valid.sum(0)  # how many labeled items per aspect

    # choose top-K aspects by coverage
    order = np.argsort(-counts_per_aspect)
    K = min(topk, A)
    top_idx = order[:K]

    comp = np.zeros((K, 3), dtype=float)  # neg, neu, pos
    for i, a in enumerate(top_idx):
        col = lab[:, a]
        m = col != -2
        if m.any():
            neg = np.sum(col[m] == -1)
            neu = np.sum(col[m] == 0)
            pos = np.sum(col[m] == 1)
            s = neg + neu + pos
            if s > 0:
                comp[i] = [neg / s, neu / s, pos / s]
    ticks = [str(a + 1) for a in top_idx]  # 1-based display
    return top_idx, comp, ticks

def gather_mean_sentiment_per_item(lab):
    """
    Mean aspect sentiment per item ∈ [-1, 1], ignoring NA (-2).
    Return: x (float array), with length = #items having at least one labeled aspect.
    """
    M = []
    for row in lab:
        m = row != -2
        if np.any(m):
            vals = row[m].astype(float)  # already in {-1,0,1}
            M.append(np.mean(vals))
    return np.asarray(M, dtype=float)

# ----------------------------
# Plotting: panel (d)
# ----------------------------
def draw_panel_d(axd, x_sent_mean, y_stars, fig=None, cmap="viridis"):
    """
    (d) Scatter colored by per-bin counts (log). Short colorbar on the right.
    Pearson r & Spearman ρ appear horizontally at the top (outside axes).
    """
    x = np.asarray(x_sent_mean, dtype=float)
    y = np.asarray(y_stars, dtype=float)

    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size == 0:
        axd.text(0.5, 0.5, "no valid points", ha="center", va="center", transform=axd.transAxes)
        return

    # Bin edges (fine but not too dense; good for print)
    xb = np.linspace(-1.0, 1.0, 60)
    yb = np.linspace(1.0, 5.0, 41)
    xi = np.clip(np.digitize(x, xb) - 1, 0, len(xb) - 2)
    yi = np.clip(np.digitize(y, yb) - 1, 0, len(yb) - 2)

    grid = np.zeros((len(xb) - 1, len(yb) - 1), dtype=int)
    np.add.at(grid, (xi, yi), 1)
    counts = grid[xi, yi]

    cmin = max(1, counts.min())
    cmax = max(cmin + 1, counts.max())

    sc = axd.scatter(
        x,
        y,
        c=counts,
        s=8,
        alpha=0.75,
        linewidths=0,
        cmap=cmap,
        norm=mcolors.LogNorm(vmin=cmin, vmax=cmax),
        rasterized=True,
    )

    '''axd.set_title("Sentiment vs. rating", pad=6)'''
    axd.set_xlabel("mean aspect sentiment (-1..1)")
    axd.set_ylabel("rating (1..5)")
    axd.set_xlim(-1.05, 1.05)
    axd.set_ylim(0.9, 5.1)
    axd.set_yticks([1, 2, 3, 4, 5])

    # Statistics at the top (horizontal, outside axes, not covering points)
    pear = float(np.corrcoef(x, y)[0, 1]) if x.size > 1 else np.nan
    spear = spearman_rho(x, y) if x.size > 1 else np.nan
    axd.text(
        0.5,
        1.02,
        f"Pearson r = {pear:.3f}    Spearman ρ = {spear:.3f}",
        transform=axd.transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
        clip_on=False,
    )

    # Short colorbar (restored to reasonable length)
    if fig is None:
        fig = axd.figure
    cbar = fig.colorbar(sc, ax=axd, fraction=0.046, pad=0.04, shrink=0.8)
    cbar.set_label("count")

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_npz", required=True)
    ap.add_argument("--dev_npz", required=True)
    ap.add_argument("--test_npz", required=True)
    ap.add_argument("--topk_aspects", type=int, default=20)
    ap.add_argument("--out", default="Dataset_S2")
    args = ap.parse_args()

    # Matplotlib style tuned for journal figures
    plt.rcParams.update({
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })

    # Load three splits
    tr_ids, tr_attn, tr_lab, tr_star = load_npz(args.train_npz)
    dv_ids, dv_attn, dv_lab, dv_star = load_npz(args.dev_npz)
    te_ids, te_attn, te_lab, te_star = load_npz(args.test_npz)

    # (a) token length distribution
    len_train = gather_lengths(tr_ids, tr_attn)
    len_dev   = gather_lengths(dv_ids, dv_attn)
    len_test  = gather_lengths(te_ids, te_attn)

    # (b) aspect sentiment composition from ALL splits (can choose train only if preferred)
    lab_all = np.concatenate([tr_lab, dv_lab, te_lab], axis=0)
    _, comp, ticks = gather_aspect_comp(lab_all, topk=args.topk_aspects)

    # (c) rating distribution from ALL splits
    stars_all = np.concatenate([tr_star, dv_star, te_star], axis=0)
    stars_all = gather_ratings(stars_all)

    # (d) sentiment vs rating — per item mean sentiment & rating
    # 使用与 (c) 同样的 ALL splits
    mean_sent = gather_mean_sentiment_per_item(lab_all)

    # 为了与评分一一对应，我们仅统计“拥有至少一个已标注方面”的样本；
    # 简单做法：在 ALL 中找每条样本是否有有效方面，并筛同一位置的 star。
    has_valid = np.any(lab_all != -2, axis=1)
    stars_valid = stars_all[has_valid]
    mean_sent_valid = mean_sent  # 同上 gather_mean_sent 只会返回有标注的样本

    # ---------------- Plot layout ----------------
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    (ax_a, ax_b), (ax_c, ax_d) = axes
    fig.suptitle("Dataset profile", fontsize=16, y=0.98)

    # (a) token length distribution
    bins = np.arange(min(len_train.min(), len_dev.min(), len_test.min()),
                     max(len_train.max(), len_dev.max(), len_test.max()) + 5, 10)
    ax_a.hist(len_train, bins=bins, density=True, alpha=0.6, label="train")
    ax_a.hist(len_dev,   bins=bins, density=True, alpha=0.6, label="dev")
    ax_a.hist(len_test,  bins=bins, density=True, alpha=0.6, label="test")
    ax_a.set_xlabel("effective tokens")
    ax_a.set_ylabel("density")
    ax_a.legend(frameon=True)
    ax_a.text(-0.12, 1.02, "(a)", transform=ax_a.transAxes, fontweight="bold")

    # (b) aspect sentiment composition (stacked bar), legend at right
    x = np.arange(comp.shape[0])
    width = 0.85
    neg = comp[:, 0]
    neu = comp[:, 1]
    pos = comp[:, 2]
    ax_b.bar(x, neg, width=width, label="neg")
    ax_b.bar(x, neu, width=width, bottom=neg, label="neu")
    ax_b.bar(x, pos, width=width, bottom=neg + neu, label="pos")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(ticks, rotation=20)
    ax_b.set_ylabel("proportion")
    ax_b.set_xlabel("aspect id")
    # legend at right (original style)
    ax_b.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
    ax_b.text(-0.12, 1.02, "(b)", transform=ax_b.transAxes, fontweight="bold")

    # (c) rating distribution
    counts = [np.sum(stars_all == k) for k in [1, 2, 3, 4, 5]]
    ax_c.bar([1, 2, 3, 4, 5], counts, width=0.8)
    ax_c.set_xlabel("rating (stars)")
    ax_c.set_ylabel("count")
    ax_c.text(-0.12, 1.02, "(c)", transform=ax_c.transAxes, fontweight="bold")

    # (d) scatter w/ per-bin counts + Pearson/Spearman on top
    draw_panel_d(ax_c.figure.axes[-1], mean_sent_valid, stars_valid, fig=fig)  # ax_d == fig.axes[-1]
    ax_d.text(-0.12, 1.02, "(d)", transform=ax_d.transAxes, fontweight="bold")

    # Tight layout (reserve right margin for (b) legend)
    fig.tight_layout(rect=[0, 0, 0.92, 0.96], w_pad=2.0, h_pad=2.2)

    out = Path(args.out)
    fig.savefig(str(out.with_suffix(".png")), dpi=300)
    fig.savefig(str(out.with_suffix(".pdf")))
    print(f"[OK] Saved: {out.with_suffix('.png')} and {out.with_suffix('.pdf')}")

if __name__ == "__main__":
    main()
