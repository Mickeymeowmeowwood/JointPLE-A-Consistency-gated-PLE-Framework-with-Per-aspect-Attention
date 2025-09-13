# fig8_hparam_sensitivity.py
'''python Figure8.py --grid_json /root/autodl-tmp/train/grid20_summary.json --metric test_mae --out_prefix fig8_sensitivity
'''

import json, math, argparse, numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, OrderedDict
from pathlib import Path

# ---- Matplotlib defaults for journal figures ----
plt.rcParams.update({
    "font.size": 9,                    # Springer建议 8–12 pt（这里统一 9 pt）
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 600,                # 导出 600 dpi
    "pdf.fonttype": 42,                # 嵌入可编辑文字（TrueType）
    "ps.fonttype": 42,
})

def load_trials(path):
    obj = json.load(open(path, "r", encoding="utf-8"))
    # grid20_summary.json 是列表(list)就直接用；若是 {"trials":[...]} 也兼容
    trials = obj if isinstance(obj, list) else obj.get("trials", [])
    rows = []
    for t in trials:
        # 有的日志结构可能包一层 "params"
        params = t.get("params", {})
        row = dict(t)
        row.update(params)
        rows.append(row)
    return rows

def is_numeric(x):
    try:
        float(x); return True
    except Exception:
        return False

def agg_by_level(rows, key, metric):
    """
    返回有序字典: level -> (mean, se, n)
    """
    buckets = defaultdict(list)
    for r in rows:
        if key not in r or metric not in r: 
            continue
        v = r[key]
        m = r[metric]
        if m is None: 
            continue
        buckets[v].append(float(m))

    if not buckets:
        return OrderedDict()

    # 统一排序：数值升序；类别按字面
    def _keyord(v):
        return (0, float(v)) if is_numeric(v) else (1, str(v))

    od = OrderedDict()
    for lvl in sorted(buckets.keys(), key=_keyord):
        arr = np.asarray(buckets[lvl], dtype=float)
        mu = float(arr.mean())
        se = float(arr.std(ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else 0.0
        od[lvl] = (mu, se, len(arr))
    return od

def fmt_level(v):
    if is_numeric(v):
        f = float(v)
        # 智能格式化
        if abs(f) >= 10 or f.is_integer():
            return str(int(round(f)))
        return f"{f:.2g}"
    return str(v)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid_json", required=True)
    ap.add_argument("--metric", default="test_mae",
                    help="支持任意列名，如 test_mae/test_rmse/dev_mae 等")
    ap.add_argument("--out_prefix", default="fig8_sensitivity")
    ap.add_argument("--max_panels", type=int, default=6,
                    help="最多展示多少个超参面板")
    ap.add_argument("--max_levels", type=int, default=6,
                    help="每个超参最多展示多少个水平（避免横轴拥挤）")
    ap.add_argument("--drop_keys", nargs="*", default=["name","run_name","seed","trial","id"],
                    help="忽略这些键（高基数/非超参）")
    args = ap.parse_args()

    rows = load_trials(args.grid_json)
    if not rows:
        raise SystemExit("No trials loaded from grid_json.")

    # 候选超参键：出现在 trial 里但不在 drop_keys，且至少有2个不同取值
    # 常见的键优先放前面，确保面板合理排序
    preferred = ["batch_size", "epochs", "optimizer", "w_rp", "lambda_cons", 
                 "lr", "n_layers", "n_task_expert", "n_share_expert"]
    all_keys = set()
    for r in rows:
        all_keys.update(k for k in r.keys())

    cand = [k for k in preferred if k in all_keys and k not in args.drop_keys]
    others = [k for k in sorted(all_keys) if (k not in cand and k not in args.drop_keys)]
    keys = []
    for k in cand + others:
        vals = {r[k] for r in rows if k in r}
        if len(vals) >= 2 and len(vals) <= 20:  # 太多水平的一般不是好画的超参
            keys.append(k)
    if not keys:
        raise SystemExit("No suitable hyperparameters to plot.")

    # 先计算所有观测的全局 y 范围，便于 sharey
    metric_vals = [float(r[args.metric]) for r in rows if args.metric in r and r[args.metric] is not None]
    y_min, y_max = min(metric_vals), max(metric_vals)
    pad = 0.02 * (y_max - y_min if y_max > y_min else max(1e-6, y_max))
    y_lim = (y_min - pad, y_max + pad)

    # 选前 max_panels 个：按“有效水平数（越小越好）+ 试验覆盖数（越多越好）”排序
    scored = []
    for k in keys:
        levels = agg_by_level(rows, k, args.metric)
        if not levels: 
            continue
        n_levels = len(levels)
        n_total = sum(n for (_, _, n) in levels.values())
        score = (n_levels, -n_total)  # 先少后多
        scored.append((score, k, levels))
    scored.sort(key=lambda x: x[0])
    panels = scored[:args.max_panels]

    n = len(panels)
    ncols = 3 if n >= 3 else n
    nrows = int(math.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1*ncols+0.4, 2.6*nrows+0.6), squeeze=False, sharey=True)
    fig.suptitle(f"Hyperparameter Sensitivity (mean ± 95% CI) — metric: {args.metric} (lower is better)",
                 y=0.995)

    for idx, (_, key, levels) in enumerate(panels):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]

        # 若水平太多，只取覆盖最全的前 max_levels
        if len(levels) > args.max_levels:
            # 选择 n 最大的若干水平
            items = sorted(levels.items(), key=lambda kv: kv[1][2], reverse=True)[:args.max_levels]
            levels = OrderedDict(items)

        xs, ys, es, ns, xt = [], [], [], [], []
        for i, (lvl, (mu, se, n_obs)) in enumerate(levels.items()):
            xs.append(i)
            ys.append(mu)
            es.append(1.96*se)
            ns.append(n_obs)
            xt.append(fmt_level(lvl))

        # 折线+误差棒；只有类别序列，无需颜色编码（便于灰度打印）
        ax.errorbar(xs, ys, yerr=es, fmt='o-', linewidth=1.2, markersize=3.5, capsize=2.5)
        ax.set_xlim(-0.5, len(xs)-0.5)
        ax.set_ylim(*y_lim)
        ax.set_title(f"{key} sensitivity")
        ax.set_xlabel(key.replace('_', ' '))
        ax.set_ylabel(args.metric if c == 0 else "")
        ax.set_xticks(xs)
        ax.set_xticklabels(xt, rotation=25, ha='right')

        # 在每个点上方标注 n（样本数），数量不多时很有用
        for i, (x, y, n_obs) in enumerate(zip(xs, ys, ns)):
            ax.text(x, y, f" n={n_obs}", va='bottom', ha='center', fontsize=7)

        # 网格淡化，辅助比较
        ax.grid(axis='y', linestyle=':', linewidth=0.8, alpha=0.5)

    # 删除空白子图
    for k in range(n, nrows*ncols):
        r, c = divmod(k, ncols)
        axes[r][c].axis('off')

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = Path(args.out_prefix)
    fig.savefig(out.with_suffix(".png"))
    fig.savefig(out.with_suffix(".pdf"))
    print(f"[OK] Saved: {out.with_suffix('.png')} and {out.with_suffix('.pdf')}")
    plt.close(fig)

if __name__ == "__main__":
    main()