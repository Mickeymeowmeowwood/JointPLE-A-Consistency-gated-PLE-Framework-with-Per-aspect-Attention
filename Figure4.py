# -*- coding: utf-8 -*-
# Figure4.py — PLE gates heatmap (English labels, narrow colorbar, robust norm)
# Usage:
#   python Figure4.py --ckpt /root/autodl-tmp/0813/best_ple_clean_run1.pt --test_npz /root/autodl-tmp/tok_test_512.npz --out_prefix gates --max_batches 0 --norm log --vmin_pct 5 --vmax_pct 99.5

import argparse, types, os, numpy as np, torch
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import LogFormatter
from matplotlib import font_manager as fm
from matplotlib.font_manager import FontProperties

# ---- DO NOT force missing CJK fonts to avoid warnings ----
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["savefig.bbox"] = "tight"   # prevent label clipping

import ple11 as base
from transformers import AutoModel


# ---------- optional font helpers ----------
def pick_font(user_font_path=None, user_font_name=None):
    """Return a FontProperties that works even if CJK fonts are unavailable."""
    if user_font_path and os.path.exists(user_font_path):
        try:
            fm.fontManager.addfont(user_font_path)   # register once
        except Exception:
            pass
        return FontProperties(fname=user_font_path)

    if user_font_name:
        try:
            path = fm.findfont(user_font_name)
            if os.path.exists(path):
                try:
                    fm.fontManager.addfont(path)
                except Exception:
                    pass
                return FontProperties(fname=path)
        except Exception:
            pass

    # Fallback to default DejaVu Sans
    return FontProperties()


# ---------- patching PLELayer to capture gates ----------
def patch_layer_forward(layer, layer_id, cache):
    import torch as _torch

    def forward(self, ha, hr, hs):
        Ea = _torch.stack([e(ha) for e in self.acsa_experts], 1)   # [B, n_task, D]
        Er = _torch.stack([e(hr) for e in self.rp_experts],   1)   # [B, n_task, D]
        Es = _torch.stack([e(hs) for e in self.share_experts],1)   # [B, n_share, D]

        wa = _torch.softmax(self.gate_acsa(ha), -1).unsqueeze(-1)  # [B, n_task+n_share, 1]
        wr = _torch.softmax(self.gate_rp(hr),   -1).unsqueeze(-1)  # [B, n_task+n_share, 1]
        ws = _torch.softmax(self.gate_share(hs),-1).unsqueeze(-1)  # [B, 2*n_task+n_share, 1]

        # cache weights without the last singleton
        cache["acsa"][layer_id].append(wa.squeeze(-1).detach().cpu().numpy())
        cache["rp"][layer_id].append(wr.squeeze(-1).detach().cpu().numpy())
        cache["share"][layer_id].append(ws.squeeze(-1).detach().cpu().numpy())

        ha_o = (wa * _torch.cat([Ea, Es], 1)).sum(1)              # [B, D]
        hr_o = (wr * _torch.cat([Er, Es], 1)).sum(1)              # [B, D]
        hs_o = (ws * _torch.cat([Ea, Er, Es], 1)).sum(1)          # [B, D]
        return ha_o, hr_o, hs_o

    layer.forward = types.MethodType(forward, layer)


# ---------- small utilities ----------
def _stack_mean(per_layer_list):
    rows = []
    for arrs in per_layer_list:
        if not arrs:
            rows.append(np.zeros((1,), dtype=float))
            continue
        big = np.concatenate(arrs, axis=0)   # [sum_B, D]
        rows.append(big.mean(axis=0))
    return np.stack(rows, axis=0)             # [L, D]


def _build_norm(arr, norm_type="linear", vmin_pct=0.0, vmax_pct=100.0, gamma=0.5):
    # gate weights are in (0,1); support log/power for long-tail contrast
    eps = 1e-8
    arr = np.asarray(arr, dtype=float).copy()
    arr[arr < eps] = eps
    vmin = np.percentile(arr, vmin_pct)
    vmax = np.percentile(arr, vmax_pct)
    vmax = max(vmax, vmin + 1e-12)

    if norm_type == "log":
        return colors.LogNorm(vmin=max(vmin, eps), vmax=vmax)
    elif norm_type == "power":
        return colors.PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)
    else:
        return colors.Normalize(vmin=vmin, vmax=vmax)


def heatmap(mat, xlabels, ylabels, title, outfile,
            norm_type="linear", vmin_pct=0.0, vmax_pct=100.0, gamma=0.5,
            fp=None):
    H = max(3.0, 0.7 * len(ylabels) + 1.6)
    W = max(6.0, 0.45 * len(xlabels) + 1.8)

    fig, ax = plt.subplots(figsize=(W, H))
    norm = _build_norm(mat, norm_type, vmin_pct, vmax_pct, gamma)
    im = ax.imshow(mat, aspect="auto", norm=norm)

    ax.set_title(title, fontproperties=fp)
    ax.set_xlabel("Experts", fontproperties=fp)
    ax.set_ylabel("PLE layers", fontproperties=fp)

    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=9, fontproperties=fp)
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize=10, fontproperties=fp)

    # narrow colorbar + log tick formatter if needed
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, shrink=0.92, aspect=28)
    if isinstance(norm, colors.LogNorm):
        cbar.formatter = LogFormatter(10, labelOnlyBase=False)  # show intermediate ticks
        cbar.update_ticks()
    cbar.ax.tick_params(labelsize=8)

    fig.tight_layout()
    fig.savefig(outfile + ".png", dpi=600)
    fig.savefig(outfile + ".pdf")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--test_npz", required=True)
    ap.add_argument("--out_prefix", default="gates")
    ap.add_argument("--max_batches", type=int, default=0)

    # visualization controls
    ap.add_argument("--norm", choices=["linear", "log", "power"], default="log")
    ap.add_argument("--vmin_pct", type=float, default=5.0)
    ap.add_argument("--vmax_pct", type=float, default=99.5)
    ap.add_argument("--gamma", type=float, default=0.5)

    # optional font
    ap.add_argument("--font_path", default=None, help="Path to a .ttf/.otf/.ttc font")
    ap.add_argument("--font_name", default=None, help="System font name (e.g., 'Noto Sans CJK SC')")
    args = ap.parse_args()

    fp = pick_font(args.font_path, args.font_name)

    # ---- load model
    backbone = AutoModel.from_pretrained(base.Config.MODEL_NAME)
    model = base.JointPLE(
        backbone, backbone.config.hidden_size, n_layers=2, n_task_expert=2, n_share_expert=2
    ).to(base.Config.DEVICE)
    state = torch.load(args.ckpt, map_location=base.Config.DEVICE)
    model.load_state_dict(state, strict=True)
    model.eval()

    L = len(model.layers)
    n_task = len(model.layers[0].acsa_experts)
    n_share = len(model.layers[0].share_experts)

    cache = {
        "acsa": [[] for _ in range(L)],
        "rp": [[] for _ in range(L)],
        "share": [[] for _ in range(L)],
    }
    for li, layer in enumerate(model.layers):
        patch_layer_forward(layer, li, cache)

    loader = torch.utils.data.DataLoader(
        base.NPZDataset(args.test_npz, mode="both"),
        batch_size=32, shuffle=False, collate_fn=base.collate_fn,
    )

    with torch.no_grad():
        for bi, b in enumerate(loader):
            bb = {k: v.to(base.Config.DEVICE) for k, v in b.items()}
            _ = model(bb["input_ids"], bb["attention_mask"], None, None)
            if args.max_batches and (bi + 1) >= args.max_batches:
                break

    # ---- aggregate
    acsa_mat = _stack_mean(cache["acsa"])     # [L, n_task+n_share]
    rp_mat = _stack_mean(cache["rp"])         # [L, n_task+n_share]
    share_mat = _stack_mean(cache["share"])   # [L, 2*n_task+n_share]

    # ---- labels (English)
    acsa_cols = [f"Task expert {i+1}" for i in range(n_task)] + \
                [f"Shared expert {j+1}" for j in range(n_share)]
    rp_cols   = [f"Task expert {i+1}" for i in range(n_task)] + \
                [f"Shared expert {j+1}" for j in range(n_share)]
    share_cols= [f"ACSA expert {i+1}" for i in range(n_task)] + \
                [f"RP expert {i+1}"   for i in range(n_task)] + \
                [f"Shared expert {j+1}" for j in range(n_share)]
    rows = [f"Layer {i+1}" for i in range(L)]

    # ---- draw three heatmaps
    heatmap(
        acsa_mat, acsa_cols, rows,
        "ACSA Gate (layer × expert) – mean weight",
        f"{args.out_prefix}_acsa",
        norm_type=args.norm, vmin_pct=args.vmin_pct, vmax_pct=args.vmax_pct, gamma=args.gamma, fp=fp,
    )
    heatmap(
        rp_mat, rp_cols, rows,
        "RP Gate (layer × expert) – mean weight",
        f"{args.out_prefix}_rp",
        norm_type=args.norm, vmin_pct=args.vmin_pct, vmax_pct=args.vmax_pct, gamma=args.gamma, fp=fp,
    )
    heatmap(
        share_mat, share_cols, rows,
        "Shared Gate (layer × expert) – mean weight",
        f"{args.out_prefix}_share",
        norm_type=args.norm, vmin_pct=args.vmin_pct, vmax_pct=args.vmax_pct, gamma=args.gamma, fp=fp,
    )

    print("[OK] Saved:",
          f"{args.out_prefix}_acsa.(png/pdf), "
          f"{args.out_prefix}_rp.(png/pdf), "
          f"{args.out_prefix}_share.(png/pdf)")

if __name__ == "__main__":
    main()
