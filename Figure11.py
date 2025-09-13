# -*- coding: utf-8 -*-
# Figure11.py  —— OOM-safe 版本：S4 t-SNE/UMAP 可视化
# 用法示例：
# python Figure11.py --ckpt /root/autodl-tmp/0813/best_ple_clean_run1.pt --test_npz /root/autodl-tmp/tok_test_512.npz --mode sentence --max_points 2000 --batch_size 8 --precision bf16 --chunk_ff 1024 --out_prefix S4_embed

import argparse, os, gc, math, inspect
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
try:
    import umap  # umap-learn
    HAVE_UMAP = True
except Exception:
    HAVE_UMAP = False

import ple11 as base
from transformers import AutoModel

plt.rcParams["axes.unicode_minus"] = False

def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).bool()
    x = last_hidden_state.masked_fill(~mask, 0.0)
    len_ = attention_mask.sum(1).clamp(min=1).unsqueeze(-1)
    return (x.sum(1) / len_).float()  # [B, H]

def prepare_model(ckpt_path: str):
    backbone = AutoModel.from_pretrained(base.Config.MODEL_NAME)
    model = base.JointPLE(backbone, backbone.config.hidden_size,
                          n_layers=2, n_task_expert=2, n_share_expert=2).to(base.Config.DEVICE)
    state = torch.load(ckpt_path, map_location=base.Config.DEVICE)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model

def build_loader(test_npz: str, bs: int):
    ds = base.NPZDataset(test_npz, mode='both')
    loader = torch.utils.data.DataLoader(
        ds, batch_size=bs, shuffle=False, collate_fn=base.collate_fn, pin_memory=False
    )
    return loader, ds

def _set_iter_kw(tsne_cls, n_iter_val: int) -> dict:
    """
    Return the correct iteration kwarg for the installed scikit-learn:
    - >=1.5 uses 'max_iter'
    - older releases use 'n_iter'
    """
    params = inspect.signature(tsne_cls.__init__).parameters
    if "max_iter" in params:          # sklearn >= 1.5
        return {"max_iter": int(n_iter_val)}
    elif "n_iter" in params:          # sklearn < 1.5
        return {"n_iter": int(n_iter_val)}
    else:
        # very old builds: fall back to default iterations by returning nothing
        return {}

def safe_tsne(X: np.ndarray,
              seed: int = 42,
              n_iter: int = 1000,
              perplexity_candidates=(5, 10, 20, 30, 40)) -> np.ndarray:
    """
    Memory-safe & version-compatible t-SNE.
    - Auto-chooses a valid perplexity (< n/3)
    - Uses PCA init + learning_rate='auto'
    - Passes 'max_iter' or 'n_iter' depending on sklearn version
    """
    X = np.asarray(X, dtype=np.float32, order="C")
    n = X.shape[0]

    # choose a valid perplexity (must be < n/3 per sklearn's check)
    valid_perps = [p for p in perplexity_candidates if p < max(5, n // 3)]
    if not valid_perps:
        # minimal workable perplexity when n is very small
        valid_perps = [max(2, n // 3 - 1)]
    # pick a middle candidate for robustness
    perp = valid_perps[min(2, len(valid_perps) - 1)]

    iter_kw = _set_iter_kw(TSNE, n_iter)

    tsne = TSNE(
        n_components=2,
        perplexity=float(perp),
        learning_rate="auto",
        init="pca",
        random_state=seed,
        # Barnes-Hut is used automatically when appropriate; no need to set 'method'
        **iter_kw
    )
    return tsne.fit_transform(X)

def safe_umap(X: np.ndarray, rnd: int = 2023):
    if not HAVE_UMAP:
        return None
    n = len(X)
    n_neighbors = max(10, min(50, int(math.sqrt(n))))
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.1,
                        metric="cosine", random_state=rnd, verbose=False)
    return reducer.fit_transform(X)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--test_npz', required=True)
    ap.add_argument('--mode', choices=['sentence', 'token'], default='sentence',
                    help='sentence: 句向量；token: 采样 token 向量')
    ap.add_argument('--max_points', type=int, default=2000)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--precision', choices=['fp32','fp16','bf16'], default='bf16')
    ap.add_argument('--chunk_ff', type=int, default=1024,
                    help='feed-forward 分块大小（降低峰值显存；0 为关闭）')
    ap.add_argument('--out_prefix', default='S4_embed')
    args = ap.parse_args()

    device = base.Config.DEVICE
    model = prepare_model(args.ckpt)

    # 关键：降低前馈层峰值显存（Transformers 支持在 forward 中分块执行 FFN）
    # 该配置会被 BERT 层在 forward 时读取并走 chunking 路径（降低峰值，牺牲少量时间）
    if hasattr(model.backbone, "config"):
        try:
            model.backbone.config.chunk_size_feed_forward = int(args.chunk_ff)
        except Exception:
            pass  # 某些模型不支持就忽略

    # 自动混合精度
    if args.precision == 'fp16':
        amp_dtype = torch.float16
    elif args.precision == 'bf16':
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = None

    loader, ds = build_loader(args.test_npz, args.batch_size)

    feats = []
    y_star = []      # 1..5 评分
    y_sent = []      # 方面情感聚合（-1..1），如果需要

    taken = 0
    torch.cuda.empty_cache()

    # 推理：inference_mode 比 no_grad 更省显存（跳过 autograd 图构建）
    # 参考：PyTorch Autograd mechanics（Inference Mode）与 AMP 文档。:contentReference[oaicite:1]{index=1}
    with torch.inference_mode():
        for b in loader:
            if taken >= args.max_points:
                break
            bb = {k: v.to(device, non_blocking=True) for k, v in b.items()}

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(amp_dtype is not None)):
                out = model.backbone(input_ids=bb['input_ids'], attention_mask=bb['attention_mask'])
                H = out.last_hidden_state  # [B, L, H]

                if args.mode == 'sentence':
                    vec = mean_pool(H, bb['attention_mask'])       # [B, H]
                    keep = min(vec.size(0), args.max_points - taken)
                    feats.append(vec[:keep].cpu().numpy())
                    if 'stars' in bb:
                        y_star.append((bb['stars'][:keep] * 5.0).cpu().numpy())
                else:
                    # token 模式：抽样每个 batch 的有效 token，直到达到 max_points
                    mask = bb['attention_mask'].bool()  # [B, L]
                    B, L = mask.shape
                    hs = H[mask]                         # [N_eff, H]
                    remain = args.max_points - taken
                    if hs.size(0) > remain:
                        idx = torch.randperm(hs.size(0), device=hs.device)[:remain]
                        hs = hs.index_select(0, idx)
                    feats.append(hs.float().cpu().numpy())
                    if 'stars' in bb:
                        # 为 token 模式简化：用该样本对应的评分重复；可按需扩展为方面情感
                        s = (bb['stars'] * 5.0).unsqueeze(1).expand(-1, L)[mask].float()
                        if s.numel() > remain: s = s[:remain]
                        y_star.append(s.cpu().numpy())

            taken = sum(len(x) for x in feats)
            # 释放 & 清理
            del out, H
            if 'vec' in locals(): del vec
            if 'hs' in locals(): del hs
            gc.collect()
            torch.cuda.empty_cache()

    X = np.concatenate(feats, axis=0)
    stars = np.concatenate(y_star, axis=0) if y_star else None
    n = len(X)
    print(f"[S4] collected points: {n}")

    # ===== 可视化（点图；颜色按评分）=====
    def _scatter(ax, Z, title):
        if stars is not None:
            sc = ax.scatter(Z[:,0], Z[:,1], c=stars, s=6, cmap="viridis", alpha=0.9)
            cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("rating (1..5)")
        else:
            ax.scatter(Z[:,0], Z[:,1], s=6, alpha=0.9)
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])

    cols = 2 if HAVE_UMAP else 1
    fig, axes = plt.subplots(1, cols, figsize=(6*cols, 5))
    if cols == 1: axes = [axes]

    Z_tsne = safe_tsne(X)
    _scatter(axes[0], Z_tsne, "t-SNE")

    if HAVE_UMAP:
        Z_umap = safe_umap(X)
        _scatter(axes[1], Z_umap, "UMAP")

    fig.suptitle("S4: Embedding visualization")
    fig.tight_layout()
    fig.savefig(f"{args.out_prefix}.png", dpi=300)
    fig.savefig(f"{args.out_prefix}.pdf")
    print("[OK] saved:", f"{args.out_prefix}.(png/pdf)")

if __name__ == "__main__":
    # 若分配器碎片严重，可在代码层面设置默认 env（不影响外部 export）
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    main()
