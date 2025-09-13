# fig4_multiseed_variance.py
'''python Figure7.py --csv /root/autodl-tmp/train/ablation_baseline_*.csv --csv /root/autodl-tmp/train/ablation_no_ple_*.csv --group_col ablation --metrics test_acc test_f1 test_mae test_rp_acc --out_prefix Fig4_seedvar
'''

import argparse, glob, os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

# Springer: sans-serif fonts (Arial/Helvetica if available), 8–12 pt; no figure title in artwork.
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial','Helvetica','DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

CAND_METRICS = ['test_acc','test_f1','test_mae','test_rmse','test_rp_acc',
                'dev_acc','dev_f1','dev_mae','dev_rmse','dev_rp_acc']
ALIASES = {  # tolerant header mapping
    'acc':'test_acc', 'f1':'test_f1', 'mae':'test_mae', 'rp_acc':'test_rp_acc',
    'rp-acc':'test_rp_acc', 'rmse':'test_rmse'
}

def _load_frames(paths):
    frames=[]
    for p in paths:
        for f in glob.glob(p):
            try:
                df = pd.read_csv(f)
                df['__src__'] = os.path.basename(f)
                frames.append(df)
            except Exception:
                pass
    if not frames: raise RuntimeError("No CSV loaded. Check --csv pattern(s).")
    df = pd.concat(frames, ignore_index=True)

    # normalize column names
    cols = {c:ALIASES.get(c.lower(), c) for c in df.columns}
    df = df.rename(columns=cols)
    return df

def _pick_metrics(df, wanted):
    if wanted: return [m for m in wanted if m in df.columns]
    return [m for m in CAND_METRICS if m in df.columns]

def _fmt_group_labels(vals):
    return [str(v) for v in vals]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', action='append', required=True,
                    help='One or more CSV glob patterns; repeatable.')
    ap.add_argument('--group_col', default=None,
                    help='Column to group different runs (e.g., "ablation" or "tag"). '
                         'If omitted, all rows treated as one group "Model".')
    ap.add_argument('--metrics', nargs='*', default=None,
                    help='Which metrics to plot; default: auto-detect common columns.')
    ap.add_argument('--out_prefix', default='Fig4_seedvar')
    args = ap.parse_args()

    df = _load_frames(args.csv)
    metrics = _pick_metrics(df, args.metrics)

    if args.group_col is None or args.group_col not in df.columns:
        df['__group__'] = 'Model'
        group_col = '__group__'
    else:
        group_col = args.group_col

    groups = list(df[group_col].unique())
    groups_sorted = sorted(groups, key=lambda x: str(x))

    for met in metrics:
        # assemble data per group
        data = [df.loc[df[group_col]==g, met].dropna().values for g in groups_sorted]
        labels = _fmt_group_labels(groups_sorted)

        # single-figure, single-metric — complies with "no titles inside figure"
        fig_h = 4.2
        fig_w = max(4.5, 1.4*len(labels))
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        bp = ax.boxplot(data, labels=labels, showmeans=False, notch=False)
        # overlay mean markers
        means = [np.mean(d) if len(d)>0 else np.nan for d in data]
        ax.scatter(range(1, len(labels)+1), means, marker='D', zorder=3)

        ax.set_xlabel(group_col, fontsize=11)
        ax.set_ylabel(met, fontsize=11)
        ax.tick_params(axis='both', labelsize=10)
        ax.grid(True, axis='y', linestyle='--', linewidth=0.6, alpha=0.5)

        fig.tight_layout()
        base = f"{args.out_prefix}_{met}"
        fig.savefig(base + ".pdf")                   # vector (preferred)
        fig.savefig(base + ".png", dpi=600)          # 600 dpi for combo artwork
        plt.close(fig)

    print(f"[OK] Saved {len(metrics)} figures with prefix: {args.out_prefix}")

if __name__ == "__main__":
    main()