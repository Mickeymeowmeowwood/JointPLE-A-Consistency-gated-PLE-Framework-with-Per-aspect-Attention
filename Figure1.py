# acsa_confusion.py
# 生成 ACSA 三分类混淆矩阵（标准计数 & 按真实类别归一化）
# 用法：
#   python Figure1.py --ckpt /root/autodl-tmp/0813/best_ple_clean_run1.pt --test_npz /root/autodl-tmp/tok_test_512.npz --out_png acsa_cm.png --out_pdf acsa_cm.pdf
import argparse, os, numpy as np, torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 你的训练代码模块（确保 ple_ablation.py 在 PYTHONPATH 中）
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
def collect_acsa_preds(model, test_npz: str):
    loader = torch.utils.data.DataLoader(
        base.NPZDataset(test_npz, mode='both'),
        batch_size=32, shuffle=False, num_workers=4,
        pin_memory=True, collate_fn=base.collate_fn
    )
    y_true, y_pred = [], []
    for b in loader:
        b = {k: v.to(base.Config.DEVICE) for k, v in b.items()}
        logits, rp_norm, *_ = model(b['input_ids'], b['attention_mask'],
                                    b.get('labels'), None)  # stars不必提供
        labs  = b['labels'].detach().cpu().numpy().reshape(-1)     # [B*A]
        preds = logits.argmax(-1).detach().cpu().numpy().reshape(-1)
        mask = (labs != base.IGNORE_IDX)
        if np.any(mask):
            y_true.extend(labs[mask].tolist())   # 0,1,2（对应 neg/neu/pos）
            y_pred.extend(preds[mask].tolist())
    return np.array(y_true, dtype=int), np.array(y_pred, dtype=int)

def plot_and_save_cm(y_true, y_pred, out_png: str, out_pdf: str):
    labels = [0,1,2]  # neg, neu, pos
    # 1) 标准混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['neg','neu','pos'])
    fig1, ax1 = plt.subplots(figsize=(4.2, 4.0))
    disp.plot(ax=ax1, values_format='d', colorbar=False)
    cbar = fig1.colorbar(disp.im_, ax=ax1,
                     fraction=0.03,   # ← 调小就更窄，例如 0.02~0.04
                     pad=0.02,        # ← 与主图的间距
                     shrink=0.9,      # ← 高度（可选）
                     aspect=30)      
    ax1.set_title('ACSA Confusion Matrix (counts)')
    fig1.tight_layout()
    fig1.savefig(out_png.replace('.png', '_counts.png'), dpi=600)
    fig1.savefig(out_pdf.replace('.pdf', '_counts.pdf'))
    plt.close(fig1)

    # 2) 按真实类别归一化（适合类别不均衡）——每行和为1
    cm_norm = confusion_matrix(y_true, y_pred, labels=labels, normalize='true')
    disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=['neg','neu','pos'])
    fig2, ax2 = plt.subplots(figsize=(4.2, 4.0))
    disp2.plot(ax=ax2, values_format='.2f', colorbar=False)
    cbar = fig2.colorbar(disp2.im_, ax=ax2,
                     fraction=0.03,   # ← 调小就更窄，例如 0.02~0.04
                     pad=0.02,        # ← 与主图的间距
                     shrink=0.9,      # ← 高度（可选）
                     aspect=30)       # ← 纵横比，数值越大越“瘦长”（可选）
    cbar.ax.tick_params(labelsize=8) 
    ax2.set_title('ACSA Confusion Matrix (normalized by true)')
    fig2.tight_layout()
    fig2.savefig(out_png, dpi=600)
    fig2.savefig(out_pdf)
    plt.close(fig2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True, help='path to best_ple_clean_run1.pt')
    ap.add_argument('--test_npz', required=True, help='path to tok_test_512.npz')
    ap.add_argument('--out_png', default='acsa_cm.png')
    ap.add_argument('--out_pdf', default='acsa_cm.pdf')
    args = ap.parse_args()

    model = load_model(args.ckpt)
    y_true, y_pred = collect_acsa_preds(model, args.test_npz)
    plot_and_save_cm(y_true, y_pred, args.out_png, args.out_pdf)
    print('[OK] Saved:', args.out_png, args.out_pdf,
          args.out_png.replace('.png','_counts.png'),
          args.out_pdf.replace('.pdf','_counts.pdf'))

if __name__ == '__main__':
    main()
