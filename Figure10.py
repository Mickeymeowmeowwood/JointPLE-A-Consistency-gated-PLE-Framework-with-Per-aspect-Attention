# FigureS3.py
'''
python Figure10.py \
  --ckpt /root/autodl-tmp/0813/best_ple_clean_run1.pt \
  --test_npz /root/autodl-tmp/tok_test_512.npz \
  --batch_sizes 8 16 32 48 64 \
  --steps 80 --warmup 20 \
  --out_prefix S3_efficiency
  '''
import argparse, time, math, numpy as np, torch, matplotlib.pyplot as plt
from matplotlib import font_manager as fm
plt.rcParams['axes.unicode_minus'] = False

def pick_cjk_font():
    cand = ["Noto Sans CJK SC","Noto Sans CJK JP","Noto Sans CJK TC","Microsoft YaHei","SimHei","PingFang SC","WenQuanYi Zen Hei","Arial Unicode MS"]
    for n in cand:
        try:
            p = fm.findfont(n)
            if p and p.lower().endswith(('.ttf','.otf','.ttc')):
                return fm.FontProperties(fname=p).get_name()
        except Exception: pass
    return None

def load_npz(p):
    z = np.load(p, mmap_mode='r'); return z['ids'], z['attn']

def collate(batch):
    ids = torch.tensor(np.stack([b[0] for b in batch]), dtype=torch.long)
    att = torch.tensor(np.stack([b[1] for b in batch]), dtype=torch.long)
    return {'input_ids': ids, 'attention_mask': att}

def make_loader(ids, att, bs):
    N = ids.shape[0]
    def it():
        for s in range(0, N, bs):
            e = min(N, s+bs)
            yield collate([(ids[i], att[i]) for i in range(s,e)])
    return it

def trim_to_len(batch, L):
    if L is None: return batch
    batch = {k:v[:, :L].contiguous() for k,v in batch.items()}
    return batch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--test_npz', required=True)
    ap.add_argument('--batch_sizes', nargs='+', type=int, default=[8,16,32,48,64])
    ap.add_argument('--steps', type=int, default=80)
    ap.add_argument('--warmup', type=int, default=20)
    ap.add_argument('--seq_lens', nargs='+', type=int, default=[64,128,256,512])
    ap.add_argument('--out_prefix', default='S3_efficiency')
    args = ap.parse_args()

    try:
        import ple11 as base
    except Exception:
        import ple_ablation_all as base

    name = pick_cjk_font()
    if name: plt.rcParams['font.sans-serif'] = [name]

    # model
    from transformers import AutoModel
    bb = AutoModel.from_pretrained(base.Config.MODEL_NAME)
    model = base.JointPLE(bb, bb.config.hidden_size,
                          n_layers=2, n_task_expert=2, n_share_expert=2).to(base.Config.DEVICE)
    state = torch.load(args.ckpt, map_location=base.Config.DEVICE)
    model.load_state_dict(state, strict=True); model.eval()

    ids, att = load_npz(args.test_npz)
    N = ids.shape[0]
    device = base.Config.DEVICE

    def bench(bs, L=None):
        loader = make_loader(ids, att, bs)
        t_list, n_tokens, bytes_mem = [], 0, 0
        seen = 0
        with torch.no_grad():
            for i,b in enumerate(loader()):
                if i >= args.steps: break
                b = trim_to_len(b, L)
                for k in b: b[k] = b[k].to(device, non_blocking=True)
                torch.cuda.empty_cache() if device.type=='cuda' else None
                if device.type=='cuda': torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = model(b['input_ids'], b['attention_mask'], None, None)
                if device.type=='cuda': torch.cuda.synchronize()
                t1 = time.perf_counter()
                if i >= args.warmup: t_list.append(t1 - t0)
                seen += b['input_ids'].size(0)
                n_tokens += int(b['attention_mask'].sum().item())
                if device.type=='cuda':
                    bytes_mem = max(bytes_mem, torch.cuda.max_memory_allocated(device))
                    torch.cuda.reset_peak_memory_stats(device)
        dt = np.mean(t_list) if t_list else float('nan')
        thr = (seen - 0) / (len(t_list)*dt) if t_list and dt>0 else float('nan')   # samples/s
        lat = 1000.0 * (1.0 / thr) if thr>0 else float('nan')                      # ms/sample
        mem_mb = bytes_mem/1024/1024 if bytes_mem>0 else float('nan')
        return thr, lat, mem_mb

    bss = args.batch_sizes
    thr_list, lat_list, mem_list = [], [], []
    for bs in bss:
        thr, lat, mem = bench(bs, L=None)
        thr_list.append(thr); lat_list.append(lat); mem_list.append(mem)

    # 长度敏感性（固定 batch size=中位）
    mid_bs = bss[len(bss)//2]
    lens = args.seq_lens
    thr_len = []
    for L in lens:
        t,_l,_m = bench(mid_bs, L=L)
        thr_len.append(t)

    # --- plot ---
    fig = plt.figure(figsize=(11.5, 7.5))
    gs = fig.add_gridspec(2, 2, wspace=0.25, hspace=0.3)

    ax1 = fig.add_subplot(gs[0,0]); ax1.plot(bss, thr_list, marker='o'); ax1.set_xlabel('batch size'); ax1.set_ylabel('throughput (samples/s)'); ax1.set_title('Throughput vs. batch size')
    ax2 = fig.add_subplot(gs[0,1]); ax2.plot(bss, lat_list, marker='o'); ax2.set_xlabel('batch size'); ax2.set_ylabel('latency (ms/sample)');   ax2.set_title('Latency vs. batch size')
    ax3 = fig.add_subplot(gs[1,0]); ax3.plot(bss, mem_list, marker='o'); ax3.set_xlabel('batch size'); ax3.set_ylabel('GPU memory (MB)');      ax3.set_title('Memory vs. batch size')
    ax4 = fig.add_subplot(gs[1,1]); ax4.plot(lens, thr_len, marker='o'); ax4.set_xlabel('sequence length (tokens)'); ax4.set_ylabel('throughput (samples/s)'); ax4.set_title(f'Throughput vs. length (bs={mid_bs})')

    for ax in [ax1,ax2,ax3,ax4]:
        for sp in ax.spines.values(): sp.set_visible(True)

    fig.suptitle('Efficiency & resource curves (S3)', y=0.98, fontsize=16)
    fig.tight_layout()
    fig.savefig(f'{args.out_prefix}.png', dpi=600)
    fig.savefig(f'{args.out_prefix}.pdf')
    print('[OK] Saved:', f'{args.out_prefix}.(png/pdf)')

if __name__ == '__main__':
    main()
