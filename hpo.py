# grid_20runs.py  —— 每次显示“当前参数组合”的 20 组网格搜索驱动

import os, time, json, math, random
from dataclasses import dataclass, asdict
from typing import Dict, Any, List
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from torch.cuda.amp import autocast, GradScaler
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

import ple as base

DEVICE = base.Config.DEVICE
SEED   = base.Config.SEED

# -------- utils --------
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def build_loaders(batch_size: int):
    train_loader = DataLoader(
        base.NPZDataset(base.Config.TRAIN_NPZ, 'both'),
        batch_size=batch_size, shuffle=True,
        num_workers=8, pin_memory=True, collate_fn=base.collate_fn
    )
    dev_loader = DataLoader(
        base.NPZDataset(base.Config.DEV_NPZ, 'both'),
        batch_size=batch_size, shuffle=False,
        num_workers=8, pin_memory=True, collate_fn=base.collate_fn
    )
    test_loader = DataLoader(
        base.NPZDataset(base.Config.TEST_NPZ, 'both'),
        batch_size=batch_size, shuffle=False,
        num_workers=8, pin_memory=True, collate_fn=base.collate_fn
    )
    return train_loader, dev_loader, test_loader

@dataclass
class RunCfg:
    name: str
    batch_size: int
    epochs: int
    optimizer: str          # 'adam' | 'adamw'
    weight_decay: float     # only for adamw
    scheduler: str          # 'cosine' | 'linear'
    warmup_ratio: float
    w_rp: float
    lambda_cons: float
    n_layers: int
    n_task_exp: int
    n_share_exp: int
    seed: int = SEED

def make_optimizer(model, cfg: RunCfg):
    if cfg.optimizer.lower() == 'adam':
        return Adam(model.parameters(), lr=base.Config.LR)
    else:
        return AdamW(model.parameters(), lr=base.Config.LR, weight_decay=cfg.weight_decay)

def make_scheduler(optimizer, total_steps: int, cfg: RunCfg):
    num_warmup = max(1, int(total_steps * cfg.warmup_ratio))
    if cfg.scheduler == 'linear':
        return get_linear_schedule_with_warmup(optimizer, num_warmup, total_steps)
    return get_cosine_schedule_with_warmup(optimizer, num_warmup, total_steps, num_cycles=0.5)

# -------- pretty print 当前参数组合 --------
def print_cfg(cfg: RunCfg):
    print("\n>>> 当前参数组合（JSON）")
    print(json.dumps(asdict(cfg), ensure_ascii=False, indent=2))  # 官方建议用 indent 做 pretty-print :contentReference[oaicite:1]{index=1}

def short_cfg(cfg: RunCfg) -> str:
    # 进度条前缀的简短版（单行展示）
    return (f"bs={cfg.batch_size}|ep={cfg.epochs}|opt={cfg.optimizer}|wd={cfg.weight_decay}|"
            f"sch={cfg.scheduler}|wu={cfg.warmup_ratio}|wrp={cfg.w_rp}|lam={cfg.lambda_cons}|"
            f"layers={cfg.n_layers}|nt={cfg.n_task_exp}|ns={cfg.n_share_exp}|seed={cfg.seed}")

# -------- train & eval --------
def train_and_eval(cfg: RunCfg) -> Dict[str, Any]:
    set_seed(cfg.seed)
    train_loader, dev_loader, test_loader = build_loaders(cfg.batch_size)

    backbone = base.AutoModel.from_pretrained(base.Config.MODEL_NAME)
    model = base.JointPLE(backbone, backbone.config.hidden_size,
                          n_layers=cfg.n_layers,
                          n_task_expert=cfg.n_task_exp,
                          n_share_expert=cfg.n_share_exp).to(DEVICE)

    optimizer = make_optimizer(model, cfg)
    total_steps = len(train_loader) * cfg.epochs
    scheduler = make_scheduler(optimizer, total_steps, cfg)
    scaler = GradScaler()

    # —— 每次开跑就打印“当前参数组合” —— #
    print_cfg(cfg)

    best_mae = float('inf'); best_state = None; patience = 0
    for ep in range(1, cfg.epochs + 1):
        model.train()
        # 进度条前缀里也带上当前组合的关键字段（可读性强）：
        desc = f"[{cfg.name}] e{ep}/{cfg.epochs} | {short_cfg(cfg)}"
        pbar = base.tqdm(train_loader, desc=desc)  # tqdm 支持 desc/set_description :contentReference[oaicite:2]{index=2}
        for batch in pbar:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            with autocast():
                _, _, _unused_total, la, lr, lc = model(
                    batch['input_ids'], batch['attention_mask'],
                    batch.get('labels'), batch.get('stars')
                )
                total = la + cfg.w_rp * lr + cfg.lambda_cons * lc
            optimizer.zero_grad()
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); scheduler.step()
            pbar.set_postfix({"L_ac": f"{la.item():.4f}",
                              "L_rp": f"{lr.item():.4f}",
                              "L_cons": f"{lc.item():.4f}"})

        acc_dev, f1_dev, mse_dev, mae_dev, rmse_dev, rpacc_dev = base.evaluate(model, dev_loader)
        print(f"[{cfg.name}][Dev] ACC={acc_dev:.4f}, F1={f1_dev:.4f}, "
              f"MSE={mse_dev:.4f}, MAE={mae_dev:.4f}, RMSE={rmse_dev:.4f}, RP-ACC={rpacc_dev:.4f}")

        if mae_dev + 1e-12 < best_mae:
            best_mae = mae_dev
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 2:
                print("Early stopping"); break

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    acc, f1, mse, mae, rmse, rpacc = base.evaluate(model, test_loader)
    print(f"=== [{cfg.name}] Test === ACC={acc:.4f}, F1={f1:.4f}, "
          f"MSE={mse:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}, RP-ACC={rpacc:.4f}")

    return {
        **asdict(cfg),
        "dev_best_mae": float(best_mae),
        "test_acc": float(acc), "test_f1": float(f1),
        "test_mse": float(mse), "test_mae": float(mae),
        "test_rmse": float(rmse), "test_rp_acc": float(rpacc)
    }

# -------- grids (与你之前相同的 8 + 12 设计) --------
def stage1_grid(seed=SEED) -> List[RunCfg]:
    grids = []
    for bs in [16, 32]:
        for ep in [3, 5]:
            for opt, wd in [("adam", 0.0), ("adamw", 0.01)]:
                grids.append(RunCfg(
                    name=f"s1_bs{bs}_ep{ep}_{opt}",
                    batch_size=bs, epochs=ep,
                    optimizer=opt, weight_decay=wd,
                    scheduler="cosine", warmup_ratio=0.10,
                    w_rp=2.0, lambda_cons=0.2,
                    n_layers=2, n_task_exp=2, n_share_exp=2,
                    seed=seed
                ))
    return grids  # 8

def stage2_grid(base_cfg: RunCfg, seed=SEED) -> List[RunCfg]:
    grids = []
    for w_rp in [1.5, 2.0]:
        for lam in [0.0, 0.2]:
            for n_task in [1, 2, 3]:
                grids.append(RunCfg(
                    name=(f"s2_{base_cfg.optimizer}_bs{base_cfg.batch_size}_ep{base_cfg.epochs}"
                          f"_wrp{str(w_rp).replace('.','p')}_lam{str(lam).replace('.','p')}_nt{n_task}"),
                    batch_size=base_cfg.batch_size, epochs=base_cfg.epochs,
                    optimizer=base_cfg.optimizer, weight_decay=base_cfg.weight_decay,
                    scheduler=base_cfg.scheduler, warmup_ratio=base_cfg.warmup_ratio,
                    w_rp=w_rp, lambda_cons=lam,
                    n_layers=2, n_task_exp=n_task, n_share_exp=2,
                    seed=seed
                ))
    return grids  # 12

if __name__ == "__main__":
    os.makedirs(base.Config.SAVE_DIR, exist_ok=True)
    all_results = []

    # Stage 1: 8 runs
    s1_cfgs = stage1_grid(seed=SEED)
    s1_results = []
    for cfg in s1_cfgs:
        print("\n=== Running", cfg.name, "===")
        # 在调用前也打印一次（更醒目）
        print_cfg(cfg)
        res = train_and_eval(cfg)
        all_results.append(res); s1_results.append(res)

    s1_best = min(s1_results, key=lambda r: r["dev_best_mae"])
    print("\n[Stage-1] Best on Dev MAE:", json.dumps(s1_best, ensure_ascii=False, indent=2))

    # Stage 2: 12 runs
    base_cfg = RunCfg(
        name="base", batch_size=s1_best["batch_size"], epochs=s1_best["epochs"],
        optimizer=s1_best["optimizer"], weight_decay=s1_best["weight_decay"],
        scheduler="cosine", warmup_ratio=0.10,
        w_rp=2.0, lambda_cons=0.2,
        n_layers=2, n_task_exp=2, n_share_exp=2,
        seed=SEED
    )
    s2_cfgs = stage2_grid(base_cfg, seed=SEED)
    for cfg in s2_cfgs:
        print("\n=== Running", cfg.name, "===")
        print_cfg(cfg)
        res = train_and_eval(cfg)
        all_results.append(res)

    ts = int(time.time())
    out_path = os.path.join(base.Config.SAVE_DIR, f"grid20_summary_{ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print("\nSaved all results to:", out_path)
