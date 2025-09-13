# fig3_ablation_compare.py
# Usage:
#   python Figure6.py --ablation_csv /root/autodl-tmp/train/ablation_summary.csv --paired_csv /root/autodl-tmp/train/paired_tests.csv --out_prefix fig3
import argparse, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _need(colset, key):
    c = [c for c in colset if c.lower() == key.lower()]
    return c[0] if c else None

def bar_with_err(df, name_col, mean_col, std_col, title, ylabel, outfile):
    x = np.arange(len(df))
    y = df[mean_col].values.astype(float)
    e = df[std_col].values.astype(float)
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.bar(x, y, yerr=e, capsize=3)
    ax.set_xticks(x); ax.set_xticklabels(df[name_col].tolist(), rotation=45, ha='right', fontsize=9)
    ax.set_ylabel(ylabel); ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outfile + '.png', dpi=300)
    fig.savefig(outfile + '.pdf')
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ablation_csv', required=True)
    ap.add_argument('--paired_csv', required=True)
    ap.add_argument('--out_prefix', default='fig3')
    args = ap.parse_args()

    abl = pd.read_csv(args.ablation_csv)
    # expected columns like: ablation, test_mae_mean, test_mae_std, test_rp_acc_mean, test_rp_acc_std
    name_col = _need(abl.columns, 'ablation') or 'ablation'

    mae_m = _need(abl.columns, 'test_mae_mean'); mae_s = _need(abl.columns, 'test_mae_std')
    rp_m  = _need(abl.columns, 'test_rp_acc_mean'); rp_s  = _need(abl.columns, 'test_rp_acc_std')
    if mae_m is None or mae_s is None:
        raise ValueError("Columns test_mae_mean/test_mae_std not found in ablation_summary.csv")
    if rp_m is None or rp_s is None:
        # allow missing RP-ACC; only draw MAE
        rp_m, rp_s = None, None

    # sort by baseline first, then others by MAE
    order = [i for i,_ in sorted(enumerate(abl[mae_m].values), key=lambda t: (0 if abl.loc[t[0],name_col]=='baseline' else 1, t[1]))]
    abl = abl.iloc[order].reset_index(drop=True)

    # Fig.3a: MAE
    bar_with_err(abl, name_col, mae_m, mae_s,
                 title='Fig.3a  Ablation study: Test-MAE (mean ± std)',
                 ylabel='Test-MAE (lower is better)',
                 outfile=f'{args.out_prefix}a_ablation_mae')

    # Fig.3b: RP-ACC
    if rp_m is not None:
        bar_with_err(abl, name_col, rp_m, rp_s,
                     title='Fig.3b  Ablation study: Test-RP-ACC (mean ± std)',
                     ylabel='Test-RP-ACC (higher is better)',
                     outfile=f'{args.out_prefix}b_ablation_rpacc')

    # Fig.3c: significance heatmap vs. baseline using p_randomization
    pt = pd.read_csv(args.paired_csv)
    # keep wanted metrics
    wanted = ['test_acc','test_f1','test_mae','test_rp_acc']
    pt = pt[pt['metric'].isin(wanted)].copy()
    # choose one p-value column
    pcol = 'p_randomization' if 'p_randomization' in pt.columns else ('p_sign' if 'p_sign' in pt.columns else 'p_bootstrap')
    # pivot: rows=ablation, cols=metric, values=-log10(p)
    pt['neglogp'] = -np.log10(pt[pcol].clip(lower=1e-12))
    piv = pt.pivot_table(index='ablation', columns='metric', values='neglogp', aggfunc='first').fillna(0.0)
    # sort baseline to top (drop), then sort by MAE column if present
    if 'baseline' in piv.index:
        piv = piv.drop(index='baseline')
    if 'test_mae' in piv.columns:
        piv = piv.sort_values('test_mae', ascending=False)

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    im = ax.imshow(piv.values, aspect='auto')
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns.tolist(), rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index.tolist(), fontsize=9)
    ax.set_title('Fig.3c  Significance vs. baseline  (−log10 p)')
    # mark common cut-offs as horizontal rules in caption; here we only add a colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, shrink=0.9, aspect=28)
    cbar.ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(f'{args.out_prefix}c_signif_heatmap.png', dpi=300)
    fig.savefig(f'{args.out_prefix}c_signif_heatmap.pdf')
    plt.close(fig)

if __name__ == '__main__':
    main()
