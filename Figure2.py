# rp_calibration_error.py
# 生成 RP 的“校准曲线（回归版可靠性图）+ 误差直方图”
# 用法：
#   python Figure2.py --ckpt /root/autodl-tmp/0813/best_ple_clean_run1.pt --test_npz /root/autodl-tmp/tok_test_512.npz --bins 10 --out_prefix rp_calib
import argparse, os, numpy as np, torch
import matplotlib.pyplot as plt

import ple11 as base
from transformers import AutoModel

def load_model(ckpt_path: str):
    backbone = AutoModel.from_pretrained(base.Config.MODEL_NAME)
    model = base.JointPLE(backbone, backbone.config.hidden_size,
                          n_layers=2, n_task_expert=2, n_share_expert=2).to(base.Config.DEVICE)
    state = torch.load(ckpt_path, map_location=base.Config.DEVICE)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model

@torch.no_grad()
def collect_rp_preds(model, test_npz: str):
    loader = torch.utils.data.DataLoader(
        base.NPZDataset(test_npz, mode='both'),
        batch_size=32, shuffle=False, num_workers=4,
        pin_memory=True, collate_fn=base.collate_fn
    )
    y_true_1to5, y_pred_1to5 = [], []
    for b in loader:
        b = {k: v.to(base.Config.DEVICE) for k, v in b.items()}
        # 只要 rp_norm；labels/stars 不必传入（预测阶段）
        _, rp_norm, *_ = model(b['input_ids'], b['attention_mask'], None, None)
        # 将 [0,1] 预测映射回 [1,5] 浮点，并裁剪到边界
        pred_stars = torch.clamp(rp_norm * 5.0, 1.0, 5.0).detach().cpu().numpy().reshape(-1)
        true_stars = (b['stars'].detach().cpu().numpy().reshape(-1) * 5.0)  # 真实是[0,1] 需×5
        y_true_1to5.extend(true_stars.tolist())
        y_pred_1to5.extend(pred_stars.tolist())
    y_true = np.array(y_true_1to5, dtype=float)
    y_pred = np.array(y_pred_1to5, dtype=float)
    return y_true, y_pred

def rating_accuracy_rounded(y_true_1to5: np.ndarray, y_pred_float_1to5: np.ndarray) -> float:
    pred_int = np.clip(np.rint(y_pred_float_1to5), 1, 5).astype(int)
    true_int = y_true_1to5.astype(int)
    return float((pred_int == true_int).mean())

def plot_calibration(y_true, y_pred, bins: int, out_prefix: str):
    # 等宽分箱 [1,5]
    edges = np.linspace(1.0, 5.0, bins+1)
    bin_ids = np.digitize(y_pred, edges, right=True)  # 1..bins
    x_mean, y_mean, counts = [], [], []
    for b in range(1, bins+1):
        m = (bin_ids == b)
        if np.any(m):
            x_mean.append(float(y_pred[m].mean()))
            y_mean.append(float(y_true[m].mean()))
            counts.append(int(m.sum()))
        else:
            x_mean.append(np.nan); y_mean.append(np.nan); counts.append(0)
    x_mean = np.array(x_mean); y_mean = np.array(y_mean); counts = np.array(counts)

    # 绘图（单图；散点 + 折线 + y=x 参考线）
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    ax.plot([1,5], [1,5], linestyle='--')                 # 参考线
    ax.plot(x_mean, y_mean, marker='o')
    for i, c in enumerate(counts):
        if not np.isnan(x_mean[i]):
            ax.annotate(str(c), (x_mean[i], y_mean[i]), textcoords='offset points', xytext=(2,2), fontsize=8)
    ax.set_xlim(1,5); ax.set_ylim(1,5)
    ax.set_xlabel('Predicted rating (bin mean)')
    ax.set_ylabel('True rating (bin mean)')
    ax.set_title(f'RP Calibration (bins={bins})')
    fig.tight_layout()
    fig.savefig(f'{out_prefix}_calibration.png', dpi=600)
    fig.savefig(f'{out_prefix}_calibration.pdf')
    plt.close(fig)

    # 同时导出数值
    import pandas as pd
    df = pd.DataFrame({'bin_left': edges[:-1], 'bin_right': edges[1:], 'pred_mean': x_mean, 'true_mean': y_mean, 'count': counts})
    df.to_csv(f'{out_prefix}_calibration_bins.csv', index=False)

def plot_error_hist(y_true, y_pred, out_prefix: str):
    abs_err = np.abs(y_pred - y_true)
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean((y_pred - y_true)**2)))
    rp_acc = rating_accuracy_rounded(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    ax.hist(abs_err, bins=20, density=False)
    ax.set_xlabel('|pred - true| (absolute error)')
    ax.set_ylabel('Count')
    ax.set_title(f'RP Error Histogram (MAE={mae:.4f}, RMSE={rmse:.4f}, RP-ACC={rp_acc:.4f})')
    fig.tight_layout()
    fig.savefig(f'{out_prefix}_error_hist.png', dpi=600)
    fig.savefig(f'{out_prefix}_error_hist.pdf')
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True, help='path to best_ple_clean_run1.pt')
    ap.add_argument('--test_npz', required=True, help='path to tok_test_512.npz')
    ap.add_argument('--bins', type=int, default=10)
    ap.add_argument('--out_prefix', default='rp_calib')
    args = ap.parse_args()

    model = load_model(args.ckpt)
    y_true, y_pred = collect_rp_preds(model, args.test_npz)
    plot_calibration(y_true, y_pred, args.bins, args.out_prefix)
    plot_error_hist(y_true, y_pred, args.out_prefix)
    print('[OK] Saved:', f"{args.out_prefix}_calibration.(png/pdf)", f"{args.out_prefix}_error_hist.(png/pdf)")

if __name__ == '__main__':
    main()
