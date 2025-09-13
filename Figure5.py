# fig2_hparam_overview.py
# Usage:
#   python Figure5.py --grid_json /root/autodl-tmp/train/grid20_summary.json --out_prefix fig2
import json, argparse, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _guess_col(cols, keys):
    cols_l = [c.lower() for c in cols]
    for k in keys:
        for i,c in enumerate(cols_l):
            if k in c:
                return list(cols)[i]
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid_json', required=True)
    ap.add_argument('--out_prefix', default='fig2')
    args = ap.parse_args()

    with open(args.grid_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    # ---- pick metrics ----
    m_dev_mae  = _guess_col(df.columns, ['dev_best_mae'])
    m_test_mae = _guess_col(df.columns, ['test_mae'])
    m_test_f1  = _guess_col(df.columns, ['test_f1'])
    if m_dev_mae is None:
        raise ValueError("dev_mae column not found in grid json.")
    if m_test_mae is None:
        raise ValueError("test_mae column not found in grid json.")

    # ---- pick 2 hyperparams (prefer W_RP, LAMBDA_CONS) ----
    c_wrp = _guess_col(df.columns, ['w_rp', 'wrp'])
    c_lambda = _guess_col(df.columns, ['lambda_cons', 'lambda', 'cons'])
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cand = [c for c in [c_wrp, c_lambda] if c is not None]
    if len(cand) < 2:
        # fallback: choose first two numeric hyperparams with >1 unique values (excluding metrics)
        metrics_like = {m_dev_mae, m_test_mae, m_test_f1}
        hp = [c for c in num_cols if c not in metrics_like and df[c].nunique()>1]
        if len(hp) < 2:
            raise ValueError("Not enough numeric hyperparameter columns to make a 2D heatmap.")
        cand = hp[:2]
    xcol, ycol = cand[0], cand[1]

    # ---------------- Fig.2a: Dev-MAE heatmap ----------------
    piv = df.pivot_table(index=ycol, columns=xcol, values=m_dev_mae, aggfunc='mean')
    piv = piv.sort_index(ascending=True).sort_index(axis=1, ascending=True)
    mat = piv.values

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(mat, aspect='auto')
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels([str(v) for v in piv.columns], rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels([str(v) for v in piv.index], fontsize=9)
    ax.set_xlabel(xcol); ax.set_ylabel(ycol)
    ax.set_title('Fig.2a  Hyperparameter grid (Dev-MAE)')
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, shrink=0.9, aspect=28)
    cbar.ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(f'{args.out_prefix}a_hparam_heatmap.png', dpi=300)
    fig.savefig(f'{args.out_prefix}a_hparam_heatmap.pdf')
    plt.close(fig)

    # ---------------- Fig.2b: Top-10 by Test-MAE ----------------
    show = df.sort_values(m_test_mae, ascending=True).head(10).copy()
    show = show.reset_index(drop=True)
    fig2, ax2 = plt.subplots(figsize=(8, 4.2))
    ax2.bar(range(len(show)), show[m_test_mae])
    ax2.set_xticks(range(len(show)))
    # compact labels using (hp1,hp2,...) pairs if available
    hp_cols = [c for c in df.columns if c not in {m_dev_mae, m_test_mae, m_test_f1} and pd.api.types.is_numeric_dtype(df[c])]
    labels = []
    for _,row in show.iterrows():
        parts=[]
        for c in [xcol, ycol]:
            if c in row: parts.append(f'{c}={row[c]}')
        labels.append(', '.join(parts) if parts else f'run{_}')
    ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('Test-MAE (lower is better)')
    ax2.set_title('Fig.2b  Top-10 runs by Test-MAE')
    # annotate Test-F1 on bars if available
    if m_test_f1 is not None:
        for i, v in enumerate(show[m_test_mae].values):
            f1v = show.iloc[i][m_test_f1]
            ax2.text(i, v, f' F1={f1v:.3f}', ha='left', va='bottom', fontsize=8, rotation=0)
    fig2.tight_layout()
    fig2.savefig(f'{args.out_prefix}b_top_runs.png', dpi=300)
    fig2.savefig(f'{args.out_prefix}b_top_runs.pdf')
    plt.close(fig2)

if __name__ == '__main__':
    main()
