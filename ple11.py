# ple_ablation_all.py
# PLE + Per-Aspect Attention + 3-class ACSA + L1(RP) + Consistency
# Ablations: no PLE / no consistency / uniform gate / no per-aspect attention /
#            sentence pooling {mean, cls} / freeze queries / RP loss = MSE
import os, time, math, random, warnings, json, copy
from math import comb
from typing import Dict, List, Tuple
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error
from transformers import AutoModel, get_cosine_schedule_with_warmup
from tqdm.auto import tqdm
import pandas as pd

# --------------------- Configuration ---------------------
class Config:
    MODEL_NAME = '/root/autodl-tmp/chinese-macbert-base'
    BASE_SEED  = 42
    N_REPEAT   = 5                      # 跑 5 个种子
    BATCH_SIZE = 16
    EPOCHS     = 5
    LR         = 5e-5
    DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DATA_DIR   = '/root/autodl-tmp/'
    SAVE_DIR   = '/root/autodl-tmp/train/'
    TRAIN_NPZ  = DATA_DIR + 'tok_train_512.npz'
    DEV_NPZ    = DATA_DIR + 'tok_dev_512.npz'
    TEST_NPZ   = DATA_DIR + 'tok_test_512.npz'
    # 损失权重
    W_RP       = 2.0                    # 评分任务损失权重（与 s1 一致）
    LAMBDA_CONS= 0.2                    # 一致性损失权重（baseline 开）

os.makedirs(Config.SAVE_DIR, exist_ok=True)
warnings.filterwarnings('ignore')

# --------------------- Data ---------------------
IGNORE_IDX = -100

def load_npz(path):
    z = np.load(path, mmap_mode='r')
    return z['ids'], z['attn'], z['lab'], z['star']

class NPZDataset(Dataset):
    def __init__(self, npz_path, mode='both'):
        ids, attn, lab, star = load_npz(npz_path)
        self.ids, self.attn = ids, attn
        self.lab, self.star = lab, star
        self.mode = mode
    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        ex = {
            'input_ids':      torch.tensor(self.ids[i], dtype=torch.long),
            'attention_mask': torch.tensor(self.attn[i], dtype=torch.long)
        }
        if self.mode in ('acsa','both'):
            raw = self.lab[i].astype(int)
            y = np.full_like(raw, IGNORE_IDX, dtype=np.int64)
            mask = (raw != -2)
            if mask.any():
                y[mask] = (raw[mask] + 1).astype(np.int64)  # -1->0, 0->1, 1->2
            ex['labels'] = torch.tensor(y, dtype=torch.long)
        if self.mode in ('rp','both'):
            ex['stars'] = torch.tensor(self.star[i]/5.0, dtype=torch.float)  # 归一化到 [0,1]
        return ex

def collate_fn(batch):
    keys = batch[0].keys()
    out = {k: torch.stack([b[k] for b in batch]) for k in keys}
    return out

train_loader = DataLoader(NPZDataset(Config.TRAIN_NPZ, 'both'),
                          batch_size=Config.BATCH_SIZE, shuffle=True,
                          num_workers=8, pin_memory=True, collate_fn=collate_fn)
dev_loader   = DataLoader(NPZDataset(Config.DEV_NPZ,   'both'),
                          batch_size=Config.BATCH_SIZE, shuffle=False,
                          num_workers=8, pin_memory=True, collate_fn=collate_fn)
test_loader  = DataLoader(NPZDataset(Config.TEST_NPZ,  'both'),
                          batch_size=Config.BATCH_SIZE, shuffle=False,
                          num_workers=8, pin_memory=True, collate_fn=collate_fn)

# 方面数量
_tmp = NPZDataset(Config.TRAIN_NPZ, 'both')
NUM_ASPECTS = _tmp.lab.shape[1]

# --------------------- Blocks ---------------------
class Expert(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ff = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.LayerNorm(dim))
    def forward(self, x): return self.ff(x)

class PLELayer(nn.Module):
    def __init__(self, dim, n_task=2, n_share=2):
        super().__init__()
        self.acsa_experts  = nn.ModuleList([Expert(dim) for _ in range(n_task)])
        self.rp_experts    = nn.ModuleList([Expert(dim) for _ in range(n_task)])
        self.share_experts = nn.ModuleList([Expert(dim) for _ in range(n_share)])
        self.gate_acsa  = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, n_task+n_share))
        self.gate_rp    = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, n_task+n_share))
        self.gate_share = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 2*n_task+n_share))
    def forward(self, ha, hr, hs):
        Ea = torch.stack([e(ha) for e in self.acsa_experts], 1)
        Er = torch.stack([e(hr) for e in self.rp_experts],   1)
        Es = torch.stack([e(hs) for e in self.share_experts],1)
        wa = torch.softmax(self.gate_acsa(ha), -1).unsqueeze(-1)
        wr = torch.softmax(self.gate_rp(hr),   -1).unsqueeze(-1)
        ws = torch.softmax(self.gate_share(hs),-1).unsqueeze(-1)
        ha_o = (wa * torch.cat([Ea, Es], 1)).sum(1)
        hr_o = (wr * torch.cat([Er, Es], 1)).sum(1)
        hs_o = (ws * torch.cat([Ea, Er, Es], 1)).sum(1)
        return ha_o, hr_o, hs_o

# --------------------- Model ---------------------
class JointPLE(nn.Module):
    def __init__(self, backbone, hidden_size,
                 n_layers=2, n_task_expert=2, n_share_expert=2,
                 use_per_aspect_attn=True,
                 use_consistency=True,
                 uniform_consistency_gate=False,
                 freeze_queries=False,
                 rp_loss_type='l1'):
        super().__init__()
        self.backbone = backbone
        self.H = hidden_size
        self.dim = hidden_size * 2
        self.use_per_aspect_attn = use_per_aspect_attn
        self.use_consistency = use_consistency
        self.uniform_consistency_gate = uniform_consistency_gate
        self.rp_loss_type = rp_loss_type

        # PLE stack (n_layers=0 即为"去掉 PLE")
        self.layers = nn.ModuleList([PLELayer(self.dim, n_task_expert, n_share_expert) for _ in range(n_layers)])

        # Per-aspect attention
        self.aspect_queries = nn.Parameter(torch.randn(NUM_ASPECTS, self.H))
        if freeze_queries:
            self.aspect_queries.requires_grad_(False)
        self.wq = nn.Linear(self.H, self.H, bias=False)
        self.aspect_proj = nn.Linear(self.H, self.dim)  # H -> 2H

        # ACSA Head
        self.acsa_head = nn.Linear(self.dim*2, 3)

        # RP Head -> normalized [0,1]
        self.rp_head = nn.Sequential(
            nn.Linear(self.dim, self.dim//2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(self.dim//2, 1)
        )

        # Consistency gate & scale
        self.gate = nn.Sequential(nn.Linear(self.dim*2, self.dim//2), nn.GELU(), nn.Linear(self.dim//2, 1))
        self.cons_scale = nn.Parameter(torch.tensor(1.0))
        self.cons_bias  = nn.Parameter(torch.tensor(0.0))

        # Losses
        self.ce = nn.CrossEntropyLoss(ignore_index=IGNORE_IDX)
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()
        self.huber = nn.SmoothL1Loss()

    def _sentence_pool(self, last_hidden_state, attention_mask):
        # mean pooling over tokens
        mask = attention_mask.unsqueeze(-1).bool()
        x = last_hidden_state
        x_masked = x.masked_fill(~mask, 0.0)
        lengths = attention_mask.sum(1).clamp(min=1).unsqueeze(-1).float()
        mean_pool = x_masked.sum(1) / lengths                      # [B, H]
        # mean+max (baseline)
        neg_inf = torch.finfo(x.dtype).min
        max_pool, _ = x.masked_fill(~mask, neg_inf).max(1)
        return torch.cat([mean_pool, max_pool], -1)                # [B, 2H]

    def _per_aspect(self, hidden, attn_mask):
        if not self.use_per_aspect_attn:
            mask = attn_mask.unsqueeze(-1).bool()
            x_masked = hidden.masked_fill(~mask, 0.0)
            lengths = attn_mask.sum(1).clamp(min=1).unsqueeze(-1).float()
            mean_pool = x_masked.sum(1) / lengths                  # [B, H]
            return mean_pool.unsqueeze(1).expand(-1, NUM_ASPECTS, -1)  # [B, A, H]
        # Per-aspect attention
        Q = self.wq(hidden)                                        # [B, L, H]
        scores = torch.einsum('blh,ah->bla', Q, self.aspect_queries) / math.sqrt(self.H)
        mask = (attn_mask == 0).unsqueeze(-1)                      # [B, L, 1]
        neg_large = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(mask, neg_large)
        attn = torch.softmax(scores, dim=1)                        # over tokens L
        return torch.einsum('bla,blh->bah', attn, hidden)          # [B, A, H]

    def forward(self, input_ids, attention_mask, labels=None, stars=None):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state
        h0 = self._sentence_pool(hidden, attention_mask)           # [B, 2H]

        # PLE
        ha = hr = hs = h0
        for layer in self.layers:                                   # n_layers=0 时跳过
            ha, hr, hs = layer(ha, hr, hs)

        # ACSA: per-aspect attention
        m = self._per_aspect(hidden, attention_mask)               # [B, A, H]
        m_proj = self.aspect_proj(m)                               # [B, A, 2H]
        ha_expand = ha.unsqueeze(1).expand(-1, NUM_ASPECTS, -1)    # [B, A, 2H]
        z = torch.cat([m_proj, ha_expand], dim=-1)                 # [B, A, 4H]
        logits = self.acsa_head(z)                                 # [B, A, 3]

        # RP normalized to [0,1]
        rp_raw = self.rp_head(hr).squeeze(-1)                      # [B]
        rp_norm = torch.sigmoid(rp_raw)                            # [B]

        # Losses
        la = lr = lc = torch.tensor(0., device=hidden.device)
        if labels is not None:
            la = self.ce(logits.reshape(-1, 3), labels.reshape(-1))
        if stars is not None:
            if self.rp_loss_type == 'mse':
                lr = self.mse(rp_norm, stars)
            else:
                lr = self.l1(rp_norm, stars)

        # Consistency
        if self.use_consistency and stars is not None:
            with torch.no_grad() if (labels is None) else torch.enable_grad():
                probs = torch.softmax(logits, dim=-1)              # [B, A, 3]
                s_i = probs[..., 2] - probs[..., 0]                # [B, A]
                if self.uniform_consistency_gate:
                    w = torch.full_like(s_i, 1.0/NUM_ASPECTS)
                else:
                    w_pre = self.gate(z).squeeze(-1)               # [B, A]
                    w = torch.softmax(w_pre, dim=-1)               # [B, A]
                agg = (w * s_i).sum(dim=-1)                        # [B]
                r_from_aspects = torch.sigmoid(self.cons_scale * agg + self.cons_bias)
            lc = nn.SmoothL1Loss()(r_from_aspects, stars)

        total = la + Config.W_RP*lr + (Config.LAMBDA_CONS*lc if self.use_consistency else 0.0)
        return logits, rp_norm, total, la, lr, lc

# --------------------- Metrics ---------------------
def rating_accuracy(y_true_1to5, y_pred_float_1to5):
    true_int = np.asarray(y_true_1to5, dtype=np.int64)
    pred_int = np.clip(np.rint(np.asarray(y_pred_float_1to5)), 1, 5).astype(np.int64)
    return float((pred_int == true_int).mean())

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    ac_t, ac_p = [], []
    rp_t, rp_p = [], []
    for b in loader:
        b = {k:v.to(Config.DEVICE) for k,v in b.items()}
        logits, rp_norm, *_ = model(b['input_ids'], b['attention_mask'], b.get('labels'), b.get('stars'))
        # ACSA
        labs  = b['labels'].cpu().numpy().flatten()
        preds = logits.argmax(-1).cpu().numpy().flatten()
        mask = labs != IGNORE_IDX
        if mask.any():
            ac_t.extend(labs[mask].tolist())
            ac_p.extend(preds[mask].tolist())
        # RP
        true_stars = (b['stars'].cpu().numpy() * 5).flatten()
        pred_stars = torch.clamp(rp_norm * 5.0, 1.0, 5.0).cpu().numpy().flatten()
        rp_t.extend(true_stars.tolist())
        rp_p.extend(pred_stars.tolist())

    acc  = accuracy_score(ac_t, ac_p) if ac_t else 0.0
    f1   = f1_score(ac_t, ac_p, average='macro') if ac_t else 0.0
    mse  = mean_squared_error(rp_t, rp_p) if rp_t else 0.0
    mae  = mean_absolute_error(rp_t, rp_p) if rp_t else 0.0
    rmse = math.sqrt(mse) if rp_t else 0.0
    rp_acc = rating_accuracy(rp_t, rp_p) if rp_t else 0.0
    return acc, f1, mse, mae, rmse, rp_acc

# --------------------- Training loop (one run) ---------------------
def train_one_run(name, model_args, seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    backbone = AutoModel.from_pretrained(Config.MODEL_NAME)
    model = JointPLE(backbone, backbone.config.hidden_size, **model_args).to(Config.DEVICE)

    opt = Adam(model.parameters(), lr=Config.LR, weight_decay=0.0)
    total_steps = len(train_loader) * Config.EPOCHS
    sch = get_cosine_schedule_with_warmup(opt, int(total_steps*0.1), total_steps, num_cycles=0.5)
    scaler = GradScaler()

    best_mae, best_state = 1e9, None
    patience, max_patience = 3, 3

    print(f"\n=== Running {name} | seed={seed} ===")

    for ep in range(1, Config.EPOCHS+1):
        model.train()
        pbar = tqdm(train_loader, desc=f"[{name}] e{ep}/{Config.EPOCHS}")
        for b in pbar:
            b = {k:v.to(Config.DEVICE) for k,v in b.items()}
            with autocast():
                _, _, total, la, lr, lc = model(b['input_ids'], b['attention_mask'], b.get('labels'), b.get('stars'))
            opt.zero_grad()
            scaler.scale(total).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sch.step()
            pbar.set_postfix({"L_ac": f"{la.item():.4f}", "L_rp": f"{lr.item():.4f}", "L_cons": f"{lc.item():.4f}"})

        acc_d, f1_d, mse_d, mae_d, rmse_d, rpacc_d = evaluate(model, dev_loader)
        print(f"[{name}][Dev] ACC={acc_d:.4f}, F1={f1_d:.4f}, MSE={mse_d:.4f}, MAE={mae_d:.4f}, RMSE={rmse_d:.4f}, RP-ACC={rpacc_d:.4f}")

        # 以 Dev MAE 选最好（与前文一致）
        if mae_d < best_mae:
            best_mae, best_state, patience = mae_d, copy.deepcopy(model.state_dict()), 0
        else:
            patience += 1
            if patience >= max_patience:
                print("Early stopping"); break

    if best_state is not None:
        model.load_state_dict(best_state)
    # Final Test
    acc_t, f1_t, mse_t, mae_t, rmse_t, rpacc_t = evaluate(model, test_loader)
    print(f"=== [{name}] Test === ACC={acc_t:.4f}, F1={f1_t:.4f}, MSE={mse_t:.4f}, MAE={mae_t:.4f}, RMSE={rmse_t:.4f}, RP-ACC={rpacc_t:.4f}")
    torch.cuda.empty_cache()
    return dict(dev_acc=acc_d, dev_f1=f1_d, dev_mae=mae_d, dev_rmse=rmse_d,
                test_acc=acc_t, test_f1=f1_t, test_mae=mae_t, test_rmse=rmse_t,
                dev_rp_acc=rpacc_d, test_rp_acc=rpacc_t)

# --------------------- Run an ablation (5 seeds) ---------------------
def run_ablation(tag, model_args):
    rows = []
    for i in range(Config.N_REPEAT):
        seed = Config.BASE_SEED + 5*i
        out = train_one_run(f"{tag}_run{i+1}", model_args, seed)
        out['seed'] = seed; out['ablation'] = tag
        rows.append(out)

    df = pd.DataFrame(rows)
    # 关键指标的均值/方差
    agg = df[['dev_acc','dev_f1','dev_mae','dev_rp_acc','test_acc','test_f1','test_mae','test_rp_acc']] \
            .agg(['mean','std']).round(4)
    print("\n---- Summary:", tag, "----")
    print(agg)

    ts = int(time.time())
    csv_path = os.path.join(Config.SAVE_DIR, f"ablation_{tag}_{ts}.csv")
    df.to_csv(csv_path, index=False)
    print("Saved seeds detail to:", csv_path)
    return df, agg

# --------------------- Define all ablations ---------------------
# baseline（与你的 s1_bs16_ep5_adam 一致）
BASE_ARGS = dict(
    n_layers=2, n_task_expert=2, n_share_expert=2,
    use_per_aspect_attn=True,
    use_consistency=True,
    uniform_consistency_gate=False,
    freeze_queries=False,
    rp_loss_type='l1',
)

ABLATIONS = [
    ('baseline',                        BASE_ARGS),
    ('no_ple',                          {**BASE_ARGS, 'n_layers': 0}),
    ('no_consistency',                  {**BASE_ARGS, 'use_consistency': False}),
    ('uniform_gate',                    {**BASE_ARGS, 'uniform_consistency_gate': True}),
    ('no_aspect_attention',             {**BASE_ARGS, 'use_per_aspect_attn': False}),
    ('freeze_queries',                  {**BASE_ARGS, 'freeze_queries': True}),
    ('rp_loss_mse',                     {**BASE_ARGS, 'rp_loss_type': 'mse'}),
]

# --------------------- Paired significance tests ---------------------
def _orientation(metric: str) -> str:
    # 'max' 表示数值越大越好；'min' 表示越小越好
    if metric in ('test_acc', 'test_f1', 'test_rp_acc'):
        return 'max'
    elif metric in ('test_mae',):
        return 'min'
    else:
        # 默认按“越大越好”
        return 'max'

def _prep_diffs(base_vals: np.ndarray, other_vals: np.ndarray, metric: str) -> np.ndarray:
    """
    以“baseline 是否更好”为方向构造配对差异：
    对越大越好指标: diff = baseline - other
    对越小越好指标: diff = other - baseline
    diff > 0 代表 baseline 更好
    """
    orient = _orientation(metric)
    if orient == 'max':
        diffs = base_vals - other_vals
    else:  # 'min'
        diffs = other_vals - base_vals
    return diffs.astype(float)

def exact_randomization_test(diffs: np.ndarray, two_sided: bool = True) -> float:
    """
    Fisher 置换思路在 n=5 时可遍历所有 2^n 个符号翻转，得到精确 p 值。
    H0: 差异的期望为 0。统计量采用平均差的绝对值。
    """
    diffs = np.asarray(diffs, dtype=float)
    n = diffs.size
    if n == 0:
        return 1.0
    obs = abs(diffs.mean())
    total = 1 << n
    count = 0
    for s in range(total):
        # 生成 {+1,-1}^n 的一个符号向量
        signs = np.array([1 if (s >> i) & 1 else -1 for i in range(n)], dtype=float)
        stat = abs((diffs * signs).mean())
        if stat + 1e-12 >= obs:
            count += 1
    p = count / float(total)
    return p if two_sided else min(1.0, p*0.5)  # 我们统计量是绝对值，已是双侧

def sign_test_exact(diffs: np.ndarray, two_sided: bool = True) -> float:
    """
    精确符号检验：丢弃 ties，计算 wins/losses 在 Bernoulli(0.5) 下的尾部概率并做双侧修正。
    """
    diffs = np.asarray(diffs, dtype=float)
    wins = int(np.sum(diffs > 0))
    losses = int(np.sum(diffs < 0))
    n_eff = wins + losses
    if n_eff == 0:
        return 1.0
    # 更小的一侧
    k = min(wins, losses)
    cdf = sum(comb(n_eff, i) for i in range(0, k+1)) / (2 ** n_eff)
    p = min(1.0, 2.0 * cdf) if two_sided else cdf
    return float(p)

def paired_bootstrap_test(diffs: np.ndarray, n_boot: int = 10000, seed: int = 1234) -> float:
    """
    配对自助法：对 n 个配对差值做有放回采样，统计均值的绝对值与观测值的对比。
    """
    rng = np.random.default_rng(seed)
    diffs = np.asarray(diffs, dtype=float)
    n = diffs.size
    if n == 0:
        return 1.0
    obs = abs(diffs.mean())
    cnt = 0
    for _ in range(n_boot):
        sample = diffs[rng.integers(0, n, size=n)]
        if abs(sample.mean()) + 1e-12 >= obs:
            cnt += 1
    return cnt / float(n_boot)

def compare_against_baseline(all_runs: Dict[str, pd.DataFrame],
                             metrics: List[str] = ['test_acc','test_f1','test_mae','test_rp_acc']
                             ) -> pd.DataFrame:
    """
    输入：每个消融 tag 的 5 种子明细 DataFrame（run_ablation 返回的 df）
    输出：与 baseline 的配对检验结果（同一 seed 对齐）
    """
    assert 'baseline' in all_runs, "baseline 结果缺失"
    base = all_runs['baseline'][['seed'] + metrics].copy()

    rows = []
    for tag, df in all_runs.items():
        if tag == 'baseline': 
            continue
        # 按 seed 内连接，保证配对
        merged = base.merge(df[['seed'] + metrics], on='seed', suffixes=('_base', '_other'))
        common_n = len(merged)
        if common_n == 0:
            continue
        for m in metrics:
            a = merged[f'{m}_base'].to_numpy()
            b = merged[f'{m}_other'].to_numpy()
            diffs = _prep_diffs(a, b, m)   # >0 表示 baseline 更好
            # 三种检验
            p_rand = exact_randomization_test(diffs)
            p_sign = sign_test_exact(diffs)
            p_boot = paired_bootstrap_test(diffs)
            # 均值方向
            orient = _orientation(m)
            mean_base = float(a.mean())
            mean_other= float(b.mean())
            better = 'baseline' if (mean_base > mean_other if orient=='max' else mean_base < mean_other) else tag
            rows.append({
                'ablation': tag,
                'metric': m,
                'n_pairs': common_n,
                'baseline_mean': round(mean_base, 6),
                'ablation_mean': round(mean_other, 6),
                'better': better,
                'p_randomization': round(p_rand, 6),
                'p_sign': round(p_sign, 6),
                'p_bootstrap': round(p_boot, 6),
            })
    out = pd.DataFrame(rows)
    ts = int(time.time())
    out_path = os.path.join(Config.SAVE_DIR, f"paired_tests_{ts}.csv")
    out.to_csv(out_path, index=False)
    print("\n===== Paired significance vs. baseline =====")
    if not out.empty:
        print(out.to_string(index=False))
    print("Saved paired significance to:", out_path)
    return out

# --------------------- Main ---------------------
if __name__ == '__main__':
    all_summaries = []
    all_runs: Dict[str, pd.DataFrame] = {}
    for tag, args in ABLATIONS:
        df, agg = run_ablation(tag, args)
        all_runs[tag] = df
        # 记录简表（均值）
        row = {'ablation': tag}
        row.update({f'{k}_mean': v for k,v in agg.loc['mean'].to_dict().items()})
        row.update({f'{k}_std': v for k,v in agg.loc['std'].to_dict().items()})
        all_summaries.append(row)

    summary_df = pd.DataFrame(all_summaries)
    sum_path = os.path.join(Config.SAVE_DIR, f"ablation_summary_{int(time.time())}.csv")
    summary_df.to_csv(sum_path, index=False)
    print("\n===== All Ablations (5 seeds each) =====")
    print(summary_df.to_string(index=False))
    print("Saved summary to:", sum_path)

    # ---- 新增：与 baseline 的配对显著性检验 ----
    _ = compare_against_baseline(all_runs,
                                 metrics=['test_acc','test_f1','test_mae','test_rp_acc'])
