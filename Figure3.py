# -*- coding: utf-8 -*-
# attention_heatmap_cn_fix.py
# 目标：更清晰的“按方面注意力”热力图（中文兼容、Top-K×Top-M子集、对数/幂律归一、窄色条）
'''python Figure3.py \
  --ckpt /root/autodl-tmp/0813/best_ple_clean_run1.pt \
  --test_npz /root/autodl-tmp/tok_test_512.npz \
  --sample_index 0 \
  --topk_aspects 6 --topm_tokens 8 --max_total_tokens 40 \
  --norm log --vmin_pct 60 --vmax_pct 99.5 \
  --out_prefix attn_case_fix
'''


import argparse, math, numpy as np, torch, os, glob, shutil
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import font_manager as fm
from matplotlib.font_manager import FontProperties
from pathlib import Path
import ple11 as base   # 如用 ple_ablation.py，改这里
from transformers import AutoModel, AutoTokenizer

# ========= 中文字体“自举” =========
def setup_cjk_font(font_name=None, font_path=None, refresh_cache=True):
    """
    目的：在容器/服务器上确保 Matplotlib 能显示中文。
    做法：
      1) 可选地清理字体缓存（当前进程会重建）；
      2) 主动注册系统里的 CJK 字体文件（Noto CJK 等）或用户提供的 font_path；
      3) 若能解析出常见家族名（如 'Noto Sans CJK SC'），设为全局 sans-serif；
      4) 返回一个 FontProperties，后续显式绑定到标题/刻度以确保中文显示。
    """
    # 1) 清缓存（可避免长期遗留的 fontlist 误判）
    if refresh_cache:
        try:
            cache_dir = mpl.get_cachedir()
            shutil.rmtree(cache_dir, ignore_errors=True)
        except Exception:
            pass

    # 2) 先注册用户明确指定的字体
    fp_return = None
    if font_path and os.path.exists(font_path):
        try:
            fm.fontManager.addfont(font_path)
            fp_return = FontProperties(fname=font_path)
        except Exception:
            pass

    # 3) 主动注册系统常见的 Noto CJK 字体（ttc/ttf/otf）
    #    Ubuntu: /usr/share/fonts/opentype/noto/、/usr/share/fonts/truetype/noto/
    patterns = (
        "/usr/share/fonts/opentype/noto/*CJK*.*",
        "/usr/share/fonts/truetype/noto/*CJK*.*",
        "/usr/share/fonts/*/*Noto*Sans*CJK*.*",
    )
    found_any = False
    for pat in patterns:
        for p in glob.glob(pat):
            try:
                fm.fontManager.addfont(p)
                found_any = True
                if fp_return is None:
                    fp_return = FontProperties(fname=p)  # 选一个作显式 FP
            except Exception:
                continue

    # 4) 尝试选择家族名
    candidates = []
    if font_name:
        candidates.append(font_name)
    candidates += ["Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Sans CJK TC",
                   "Microsoft YaHei", "SimHei", "PingFang SC", "WenQuanYi Zen Hei"]
    chosen = None
    for fam in candidates:
        try:
            # 如果能找到，就说明该家族可用
            fm.findfont(fam, fallback_to_default=False)
            chosen = fam
            break
        except Exception:
            continue

    # 5) 全局设置与负号修正
    mpl.rcParams["axes.unicode_minus"] = False
    if chosen:
        mpl.rcParams["font.family"] = "sans-serif"
        mpl.rcParams["font.sans-serif"] = [chosen]

    return fp_return  # 后续显式绑定（标题/刻度/色条刻度）

# 运行时先做一次字体自举
FP_CJK = setup_cjk_font()

# ========= 原逻辑 =========
SPECIAL = set(['[PAD]','[CLS]','[SEP]'])

def clean_tokens(tokens):
    out=[]
    for t in tokens:
        if t in SPECIAL: continue
        if t.startswith('##'): t = t[2:]
        out.append(t)
    return out

@torch.no_grad()
def compute_attn(model, input_ids, attention_mask):
    # 复现源码：scores = <WQ*h, aspect_queries> / sqrt(H)，在 token 维 softmax
    out = model.backbone(input_ids=input_ids, attention_mask=attention_mask)
    hidden = out.last_hidden_state
    Q = model.wq(hidden)
    scores = torch.einsum('blh,ah->bla', Q, model.aspect_queries) / math.sqrt(model.H)
    mask = (attention_mask == 0).unsqueeze(-1)
    scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
    attn = torch.softmax(scores, dim=1)  # over tokens L
    return attn  # [B, L, A]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--test_npz', required=True)
    ap.add_argument('--sample_index', type=int, default=0)
    ap.add_argument('--topk_aspects', type=int, default=6)
    ap.add_argument('--topm_tokens', type=int, default=8,
                    help='每个方面选取前M个token（按注意力值），随后取并集')
    ap.add_argument('--max_total_tokens', type=int, default=40,
                    help='并集最多展示多少列，超出则按总注意力再筛')
    ap.add_argument('--norm', choices=['log','power','linear'], default='log')
    ap.add_argument('--gamma', type=float, default=0.5, help='power归一化的gamma')
    ap.add_argument('--vmin_pct', type=float, default=60.0, help='下分位剪裁，如60表示第60百分位')
    ap.add_argument('--vmax_pct', type=float, default=99.5, help='上分位剪裁')
    ap.add_argument('--out_prefix', default='attn_case_fix')
    # 可选：显式指定字体
    ap.add_argument('--font_path', default=None, help='显式指定 CJK 字体文件(.ttf/.ttc/.otf)')
    ap.add_argument('--font_name', default=None, help='显式指定系统家族名，如 "Noto Sans CJK SC"')
    ap.add_argument('--no_refresh_font_cache', action='store_true',
                    help='不清理 Matplotlib 字体缓存（默认会清理）')
    args = ap.parse_args()

    # 若传入了自定义字体，重新自举一次
    if args.font_path or args.font_name or args.no_refresh_font_cache:
        global FP_CJK
        FP_CJK = setup_cjk_font(font_name=args.font_name,
                                font_path=args.font_path,
                                refresh_cache=not args.no_refresh_font_cache)

    # 1) 加载模型（推理模式）
    backbone = AutoModel.from_pretrained(base.Config.MODEL_NAME)
    model = base.JointPLE(backbone, backbone.config.hidden_size,
                          n_layers=2, n_task_expert=2, n_share_expert=2).to(base.Config.DEVICE)
    state = torch.load(args.ckpt, map_location=base.Config.DEVICE)
    model.load_state_dict(state, strict=True); model.eval()

    # 2) 数据与分词器
    tok = AutoTokenizer.from_pretrained(base.Config.MODEL_NAME)
    ds = base.NPZDataset(args.test_npz, mode='both')
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, collate_fn=base.collate_fn)

    # 3) 取目标样本，计算注意力
    for i, batch in enumerate(loader):
        if i != args.sample_index: continue
        ib = {k: v.to(base.Config.DEVICE) for k,v in batch.items()}
        attn = compute_attn(model, ib['input_ids'], ib['attention_mask'])[0]  # [L, A]
        ids = ib['input_ids'][0].cpu().tolist()
        mask = ib['attention_mask'][0].cpu().numpy().astype(bool)
        tok_all = tok.convert_ids_to_tokens(ids)

        # 仅保留有效token并去特殊符号
        idx_valid = [j for j,m in enumerate(mask) if m and tok_all[j] not in SPECIAL]
        tokens = clean_tokens([tok_all[j] for j in idx_valid])
        A = attn.shape[-1]
        mat_full = attn[idx_valid, :].cpu().numpy().T   # [A, T_all]

        # 4) 选 Top-K 方面（行）
        K = min(args.topk_aspects, A)
        top_aspects = np.argsort(-mat_full.mean(axis=1))[:K]
        mat = mat_full[top_aspects, :]   # [K, T_all]

        # 5) 每个方面选 Top-M token 的**位置**（列），并集后按位置排序
        cols_set = set()
        for r in range(K):
            cols = np.argsort(-mat[r])[:args.topm_tokens]
            for c in cols: cols_set.add(int(c))
        cols = sorted(list(cols_set))
        # 若列过多，按照“所有选中方面上的总注意力”再筛到 max_total_tokens
        if len(cols) > args.max_total_tokens:
            scores_sum = mat[:, cols].sum(axis=0)  # 每列的总注意力
            order = np.argsort(-scores_sum)[:args.max_total_tokens]
            cols = [cols[i] for i in order]
            cols = sorted(cols)  # 仍按原文顺序显示

        mat = mat[:, cols]                     # [K, T]
        tokens = [tokens[c] for c in cols]

        # 6) 归一化与分位裁剪（增强对比）
        arr = mat.copy()
        eps = 1e-8
        arr[arr < eps] = eps                   # LogNorm 不能接受0
        vmin = np.percentile(arr, args.vmin_pct)
        vmax = np.percentile(arr, args.vmax_pct)
        if args.norm == 'log':
            norm = colors.LogNorm(vmin=max(vmin, eps), vmax=max(vmax, vmin+1e-6))
        elif args.norm == 'power':
            norm = colors.PowerNorm(gamma=args.gamma, vmin=vmin, vmax=max(vmax, vmin+1e-6))
        else:
            norm = colors.Normalize(vmin=vmin, vmax=vmax)

        # 7) 画图（窄色条；横轴仅保留精选token，显著更清楚）
        H = max(2.8, 0.6*K + 1.8)
        W = max(6.0, 0.28*len(tokens) + 2.0)
        fig, ax = plt.subplots(figsize=(W, H))
        im = ax.imshow(arr, aspect='auto', norm=norm)

        # 标题与坐标轴标签（显式中文字体）
        if FP_CJK is not None:
            ax.set_title('Aspect-wise Attention Heatmap (token × aspect)', fontproperties=FP_CJK)
            ax.set_xlabel(f'Chinese tokens (selected {len(tokens)} columns)', fontproperties=FP_CJK)
            ax.set_ylabel(f'Top-{K}', fontproperties=FP_CJK)
        else:
            ax.set_title('Aspect-wise Attention Heatmap (token × aspect)')
            ax.set_xlabel(f'Chinese tokens (selected {len(tokens)} columns)）')
            ax.set_ylabel(f'Top-{K}')

        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=9)
        ax.set_yticks(range(K))
        ax.set_yticklabels([f'Aspect_{i}' for i in top_aspects], fontsize=10)

        # 若找到中文字体，强制刻度使用之
        if FP_CJK is not None:
            for lab in ax.get_xticklabels() + ax.get_yticklabels():
                lab.set_fontproperties(FP_CJK)

        # 色条变窄 + 中文刻度
        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, shrink=0.9, aspect=30)
        cbar.ax.tick_params(labelsize=8)
        if FP_CJK is not None:
            for t in cbar.ax.yaxis.get_ticklabels():
                t.set_fontproperties(FP_CJK)

        fig.tight_layout()
        fig.savefig(f'{args.out_prefix}_attention.png', dpi=600)
        fig.savefig(f'{args.out_prefix}_attention.pdf')
        print('[OK] Saved:', f'{args.out_prefix}_attention.(png/pdf)')
        break

if __name__ == '__main__':
    main()