# ple_ablation.py  ——  PLE + Per-Aspect Attention + 3-class ACSA + L1(RP) + Consistency
import os, time, math, random, warnings, logging
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW, Adam
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error
from transformers import AutoModel, AutoConfig, get_cosine_schedule_with_warmup
from tqdm.auto import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# --------------------- Logging ---------------------
logging.basicConfig(filename='ablation.log', level=logging.INFO,
                    format='[%(asctime)s] %(levelname)s: %(message)s')

# --------------------- Configuration ---------------------
class Config:
    MODEL_NAME = '/root/autodl-tmp/chinese-macbert-base'
    SEED       = 42
    BATCH_SIZE = 16
    EPOCHS     = 5
    LR         = 5e-5
    DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DATA_DIR   = '/root/autodl-tmp/'
    SAVE_DIR   = '/root/autodl-tmp/train/'
    TRAIN_NPZ  = DATA_DIR + 'tok_train_512.npz'
    DEV_NPZ    = DATA_DIR + 'tok_dev_512.npz'
    TEST_NPZ   = DATA_DIR + 'tok_test_512.npz'
    LAMBDA_CONS = 0.2       # 一致性损失权重，可调

warnings.filterwarnings('ignore')
random.seed(Config.SEED); np.random.seed(Config.SEED)
torch.manual_seed(Config.SEED); torch.cuda.manual_seed_all(Config.SEED)
os.makedirs(Config.SAVE_DIR, exist_ok=True); os.makedirs("runs", exist_ok=True)

# --------------------- Data ---------------------
to3 = {-1:0, 0:1, 1:2}
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
    out = {}
    for k in keys:
        out[k] = torch.stack([b[k] for b in batch])
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

num_aspects = NPZDataset(Config.TRAIN_NPZ, 'both').lab.shape[1]

# --------------------- PLE blocks ---------------------
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
        Ea = torch.stack([e(ha) for e in self.acsa_experts], 1)  # [B, Ta, D]
        Er = torch.stack([e(hr) for e in self.rp_experts],   1)  # [B, Tr, D]
        Es = torch.stack([e(hs) for e in self.share_experts],1)  # [B, S,  D]
        wa = torch.softmax(self.gate_acsa(ha), -1).unsqueeze(-1)
        wr = torch.softmax(self.gate_rp(hr),   -1).unsqueeze(-1)
        ws = torch.softmax(self.gate_share(hs),-1).unsqueeze(-1)
        ha_o = (wa * torch.cat([Ea, Es], 1)).sum(1)
        hr_o = (wr * torch.cat([Er, Es], 1)).sum(1)
        hs_o = (ws * torch.cat([Ea, Er, Es], 1)).sum(1)
        return ha_o, hr_o, hs_o

# --------------------- Joint Model ---------------------
class JointPLE(nn.Module):
    """
    - 编码器：BERT
    - 句向量：mean+max pooling -> h0 (2H)
    - PLE：产生 h_acsa(2H), h_rp(2H)
    - ACSA：每方面注意力池化，得到 m_i(H) -> 投影到 2H，与 h_acsa 拼接得到 z_i(4H)，分类为 3 类
    - RP：h_rp -> MLP -> 评分（归一化到 0~1，使用 Sigmoid）
    - Consistency：由各方面 3 类概率得 s_i=p_pos-p_neg，经门控 softmax 聚合得到 rating_from_aspects，再与真实评分做 SmoothL1
    """
    def __init__(self, backbone, hidden_size, n_layers=2, n_task_expert=2, n_share_expert=2):
        super().__init__()
        self.backbone = backbone
        self.H = hidden_size
        self.dim = hidden_size * 2        # mean+max
        # PLE stack
        self.layers = nn.ModuleList([PLELayer(self.dim, n_task_expert, n_share_expert) for _ in range(n_layers)])

        # Per-aspect attention
        self.aspect_queries = nn.Parameter(torch.randn(num_aspects, self.H))
        self.wq = nn.Linear(self.H, self.H, bias=False)
        self.aspect_proj = nn.Linear(self.H, self.dim)  # H -> 2H
        # ACSA Head (input 4H: [m_proj(2H); h_acsa(2H)])
        self.acsa_head = nn.Linear(self.dim*2, 3)

        # RP Head
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
        self.huber = nn.SmoothL1Loss()

    @staticmethod
    def mean_max_pool(last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).bool()
        x = last_hidden_state
        x_masked = x.masked_fill(~mask, 0.0)
        lengths = attention_mask.sum(1).clamp(min=1).unsqueeze(-1).float()
        mean_pool = x_masked.sum(1) / lengths
        neg_inf = torch.finfo(x.dtype).min
        max_pool, _ = x.masked_fill(~mask, neg_inf).max(1)
        return torch.cat([mean_pool, max_pool], -1)  # [B, 2H]

    def per_aspect_attention(self, hidden, attn_mask):
        """
        hidden: [B, L, H], attn_mask: [B, L]
        returns m: [B, A, H]
        """
        Q = self.wq(hidden)                      # [B, L, H]
        # scores[b, l, a] = <Q[b,l], query[a]>
        scores = torch.einsum('blh,ah->bla', Q, self.aspect_queries) / math.sqrt(self.H)
        # mask padding
        mask = (attn_mask == 0).unsqueeze(-1)    # [B, L, 1]
        neg_large = torch.finfo(scores.dtype).min  # fp16下是-65504，fp32下是-3.4e38
        scores = scores.masked_fill(mask, neg_large)
        attn = torch.softmax(scores, dim=1)      # over tokens L
        m = torch.einsum('bla,blh->bah', attn, hidden)  # [B, A, H]
        return m

    def forward(self, input_ids, attention_mask, labels=None, stars=None):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state                 # [B, L, H]
        h0 = self.mean_max_pool(hidden, attention_mask)  # [B, 2H]

        # PLE
        ha, hr, hs = h0, h0, h0
        for layer in self.layers:
            ha, hr, hs = layer(ha, hr, hs)

        # ACSA: per-aspect attention
        m = self.per_aspect_attention(hidden, attention_mask)     # [B, A, H]
        m_proj = self.aspect_proj(m)                              # [B, A, 2H]
        ha_expand = ha.unsqueeze(1).expand(-1, num_aspects, -1)   # [B, A, 2H]
        z = torch.cat([m_proj, ha_expand], dim=-1)                # [B, A, 4H]
        logits = self.acsa_head(z)                                # [B, A, 3]

        # RP: normalized to [0,1] via sigmoid
        rp_raw = self.rp_head(hr).squeeze(-1)                     # [B]
        rp_norm = torch.sigmoid(rp_raw)                           # [B] in [0,1]

        # Losses
        la = lr = lc = torch.tensor(0., device=hidden.device)
        if labels is not None:
            flat_logits = logits.reshape(-1, 3)
            flat_labels = labels.reshape(-1)
            la = self.ce(flat_logits, flat_labels)
        if stars is not None:
            lr = self.l1(rp_norm, stars)

        # Consistency: s_i = p_pos - p_neg
        with torch.no_grad() if (labels is None and stars is None) else torch.enable_grad():
            probs = torch.softmax(logits, dim=-1)                 # [B, A, 3]
            s_i = probs[..., 2] - probs[..., 0]                   # [B, A] in [-1,1]
            w_pre = self.gate(z).squeeze(-1)                      # [B, A]
            w = torch.softmax(w_pre, dim=-1)                      # [B, A]
            agg = (w * s_i).sum(dim=-1)                           # [B]
            # map to [0,1] by sigmoid(a*agg + b)
            r_from_aspects = torch.sigmoid(self.cons_scale * agg + self.cons_bias)
        if stars is not None:
            lc = self.huber(r_from_aspects, stars)
            
        total = la + 2*lr + Config.LAMBDA_CONS*lc
        return logits, rp_norm, total, la, lr, lc

def rating_accuracy(y_true_1to5, y_pred_float_1to5):
    # y_true_1to5: list/ndarray of ints in [1..5]
    # y_pred_float_1to5: list/ndarray of floats (pred stars before rounding)
    true_int = np.asarray(y_true_1to5, dtype=np.int64)
    pred_int = np.clip(np.rint(np.asarray(y_pred_float_1to5)), 1, 5).astype(np.int64)
    return float((pred_int == true_int).mean())

# --------------------- Evaluation ---------------------
def evaluate(model, loader):
    model.eval()
    ac_t, ac_p = [], []
    rp_t, rp_p = [], []
    with torch.no_grad():
        for b in loader:
            b = {k:v.to(Config.DEVICE) for k,v in b.items()}
            logits, rp_norm, _, _, _, _ = model(
                b['input_ids'], b['attention_mask'],
                b.get('labels'), b.get('stars')
            )
            # ACSA
            labs  = b['labels'].cpu().numpy().flatten()
            preds = logits.argmax(-1).cpu().numpy().flatten()
            mask = labs != IGNORE_IDX
            if mask.any():
                ac_t.extend(labs[mask].tolist())
                ac_p.extend(preds[mask].tolist())
            # RP（连续 -> 1..5）
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

# --------------------- Training (single variant) ---------------------
def run_variant(name='ple_clean', n_layers=2, n_task_exp=2, n_share_exp=2):
    backbone = AutoModel.from_pretrained(Config.MODEL_NAME)
    model = JointPLE(backbone, backbone.config.hidden_size,
                     n_layers=n_layers, n_task_expert=n_task_exp, n_share_expert=n_share_exp).to(Config.DEVICE)

    opt = Adam(model.parameters(), lr=Config.LR, weight_decay=0)
    total_steps = len(train_loader) * Config.EPOCHS
    sch = get_cosine_schedule_with_warmup(opt, int(total_steps*0.1), total_steps, num_cycles=0.5)
    scaler = GradScaler()
    writer = SummaryWriter(log_dir=f"runs/{name}")

    best_mae, patience = 1.0, 0
    save_path = os.path.join(Config.SAVE_DIR, f"best_{name}.pt")

    for ep in range(1, Config.EPOCHS+1):
        model.train()
        pbar = tqdm(train_loader, desc=f"[{name}] Epoch {ep}")
        for b in pbar:
            b = {k:v.to(Config.DEVICE) for k,v in b.items()}
            with autocast():
                _, _, total, la, lr, lc = model(b['input_ids'], b['attention_mask'],
                                                b.get('labels'), b.get('stars'))
            opt.zero_grad()
            scaler.scale(total).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sch.step()
            pbar.set_postfix({"L_ac": f"{la.item():.4f}", "L_rp": f"{lr.item():.4f}", "L_cons": f"{lc.item():.4f}"})

        acc_dev, f1_dev, mse_dev, mae_dev, rmse_dev, rpacc_dev = evaluate(model, dev_loader)
        print(f"[{name}][Dev] ACC={acc_dev:.4f}, F1={f1_dev:.4f}, "
              f"MSE={mse_dev:.4f}, MAE={mae_dev:.4f}, RMSE={rmse_dev:.4f}, RP-ACC={rpacc_dev:.4f}")
        writer.add_scalar('Dev/acc',  acc_dev,  ep)
        writer.add_scalar('Dev/f1',   f1_dev,   ep)
        writer.add_scalar('Dev/mse',  float(mse_dev),  ep)
        writer.add_scalar('Dev/mae',  float(mae_dev),  ep)
        writer.add_scalar('Dev/rmse', float(rmse_dev), ep)
        writer.add_scalar('Dev/rp_acc', float(rpacc_dev), ep)


        if mae_dev < best_mae:
            best_mae, patience = mae_dev, 0
            torch.save(model.state_dict(), save_path)
        else:
            patience += 1
            if patience >= 3:
                print("Early stopping"); break

    model.load_state_dict(torch.load(save_path, map_location=Config.DEVICE))
    acc, f1, mse, mae, rmse, rpacc = evaluate(model, test_loader)
    print(f"=== [{name}] Test Results === ACC={acc:.4f}, F1={f1:.4f}, "
          f"MSE={mse:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}, RP-ACC={rpacc:.4f}")
    writer.add_hparams({'lr': Config.LR, 'batch': Config.BATCH_SIZE},
                       {'test_f1': f1, 'test_mae': mae, 'test_rmse': rmse, 'test_rp_acc': rpacc})
    writer.close()

    return {
        'variant': name,
        'dev_acc': round(acc_dev,4),  'dev_f1': round(f1_dev,4),
        'dev_mse': round(mse_dev,4),  'dev_mae': round(mae_dev,4),
        'dev_rmse': round(rmse_dev,4),'dev_rp_acc': round(rpacc_dev,4),
        'test_acc': round(acc,4),     'test_f1': round(f1,4),
        'test_mse': round(mse,4),     'test_mae': round(mae,4),
        'test_rmse': round(rmse,4),   'test_rp_acc': round(rpacc,4)
    }

# --------------------- Main ---------------------
if __name__ == '__main__':
    results = []
    N_REPEAT = 5
    for i in range(N_REPEAT):
        seed = Config.SEED + 5*i
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        out = run_variant(name=f'ple_clean_run{i+1}')
        out['seed'] = seed
        results.append(out)

    df = pd.DataFrame(results)
    csv_path = os.path.join(Config.SAVE_DIR, f'ple_clean_{int(time.time())}.csv')
    df.to_csv(csv_path, index=False)
    print("\nSaved results to:", csv_path)

    # 简单可视化
    for m in ['test_f1','test_mae','test_rmse']:
        plt.figure(figsize=(6,4))
        plt.plot(range(1, N_REPEAT+1), df[m].values, marker='o')
        plt.title(m); plt.xlabel('run'); plt.grid(True); plt.tight_layout()
        plt.savefig(os.path.join(Config.SAVE_DIR, f'ple_clean_{m}.png'), dpi=160)
        plt.close()
