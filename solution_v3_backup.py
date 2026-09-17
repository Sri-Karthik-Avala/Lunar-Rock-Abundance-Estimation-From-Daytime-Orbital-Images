import os, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F

SEED = 1234
np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cudnn.benchmark = True


def find_data():
    for d in ['dataset/public', './dataset/public', '.', 'dataset', './data', 'input']:
        if os.path.exists(os.path.join(d, 'patches.npy')) and os.path.exists(os.path.join(d, 'train.csv')):
            return d
    raise FileNotFoundError('data not found')


def rankdata(a):
    a = np.asarray(a, dtype=np.float64)
    order = a.argsort(kind='mergesort')
    ranks = np.empty(len(a), dtype=np.float64)
    ranks[order] = np.arange(1, len(a) + 1)
    _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt)
    start = csum - cnt
    avg = (start + csum - 1) / 2.0 + 1.0
    return avg[inv]


def cubic_partial(rank_vals, base):
    rb = rankdata(base); rb = rb / rb.max()
    X = np.vstack([np.ones_like(rb), rb, rb ** 2, rb ** 3]).T
    coef, _, _, _ = np.linalg.lstsq(X, rank_vals, rcond=None)
    return rank_vals - X @ coef


def resid_corr(pred, truth, base):
    rp = cubic_partial(rankdata(pred), base)
    rt = cubic_partial(rankdata(truth), base)
    rp = rankdata(rp); rt = rankdata(rt)
    rp = rp - rp.mean(); rt = rt - rt.mean()
    d = np.sqrt((rp ** 2).sum() * (rt ** 2).sum())
    return float((rp * rt).sum() / d) if d > 0 else 0.0


def gn(c):
    return nn.GroupNorm(min(8, c), c)


class Blk(nn.Module):
    def __init__(self, ci, co, st=1):
        super().__init__()
        self.c1 = nn.Conv2d(ci, co, 3, st, 1, bias=False); self.n1 = gn(co)
        self.c2 = nn.Conv2d(co, co, 3, 1, 1, bias=False); self.n2 = gn(co)
        self.sc = nn.Sequential() if (st == 1 and ci == co) else nn.Sequential(nn.Conv2d(ci, co, 1, st, bias=False), gn(co))
        self.a = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.a(self.n2(self.c2(self.a(self.n1(self.c1(x))))) + self.sc(x))


class Net(nn.Module):
    def __init__(self, cin=2):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(cin, 32, 3, 1, 1, bias=False), gn(32), nn.ReLU(True))
        self.l1 = nn.Sequential(Blk(32, 32), Blk(32, 64, 2))
        self.l2 = nn.Sequential(Blk(64, 64), Blk(64, 128, 2))
        self.l3 = nn.Sequential(Blk(128, 128), Blk(128, 256, 2))
        self.l4 = nn.Sequential(Blk(256, 256, 2))
        self.drop = nn.Dropout(0.3)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.l4(self.l3(self.l2(self.l1(self.stem(x)))))
        a = F.adaptive_avg_pool2d(x, 1).flatten(1)
        m = F.adaptive_max_pool2d(x, 1).flatten(1)
        x = torch.cat([a, m], 1)
        x = self.drop(F.relu(self.fc1(self.drop(x))))
        return self.fc2(x).squeeze(1)


def d4(t, k):
    if k & 1:
        t = torch.flip(t, [3])
    return torch.rot90(t, k >> 1, [2, 3])


def augment(xb):
    n = xb.shape[0]
    if np.random.rand() < 0.5:
        xb = torch.flip(xb, [3])
    theta = torch.rand(n, device=xb.device) * (2 * np.pi)
    cs = torch.cos(theta); sn = torch.sin(theta)
    mat = torch.zeros(n, 2, 3, device=xb.device)
    mat[:, 0, 0] = cs; mat[:, 0, 1] = -sn
    mat[:, 1, 0] = sn; mat[:, 1, 1] = cs
    grid = F.affine_grid(mat, xb.shape, align_corners=False)
    xb = F.grid_sample(xb, grid, padding_mode='reflection', align_corners=False)
    c = 0.85 + 0.30 * torch.rand(n, 1, 1, 1, device=xb.device)
    xb = xb * c + 0.03 * torch.randn_like(xb)
    return xb


def corr_loss(pred, target):
    p = pred - pred.mean(); t = target - target.mean()
    d = torch.sqrt((p * p).sum() * (t * t).sum()) + 1e-8
    return 1.0 - (p * t).sum() / d


def train_fold(Xtr, ytr, Xva_list, epochs=22, bs=128):
    cin = Xtr.shape[1]
    net = Net(cin).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), 1.2e-3, weight_decay=1e-4)
    steps = (len(Xtr) + bs - 1) // bs
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 1.2e-3, epochs=epochs, steps_per_epoch=steps, pct_start=0.25)
    hub = nn.SmoothL1Loss()
    for e in range(epochs):
        net.train(); pp = torch.randperm(len(Xtr), device=Xtr.device)
        tot = 0.0
        for i in range(0, len(Xtr), bs):
            b = pp[i:i + bs]
            xb = augment(Xtr[b])
            yb = ytr[b]
            opt.zero_grad()
            out = net(xb)
            loss = hub(out, yb) + 0.5 * corr_loss(out, yb)
            loss.backward(); opt.step(); sch.step()
            tot += float(loss.detach())
        print(f'    epoch {e + 1}/{epochs} loss {tot / steps:.4f}', flush=True)
    net.eval()
    preds = []
    with torch.no_grad():
        for xv in Xva_list:
            out = np.zeros(len(xv), dtype=np.float64)
            for i in range(0, len(xv), 512):
                xb = xv[i:i + 512]
                acc = torch.zeros(len(xb), device=DEV)
                for k in range(8):
                    acc += net(d4(xb, k))
                out[i:i + 512] = (acc / 8).cpu().numpy()
            preds.append(out)
    if DEV == 'cuda':
        torch.cuda.empty_cache()
    return preds


def compute_channels(arr, stats):
    out = torch.empty((len(arr), 2, 100, 100), dtype=torch.float32, device=DEV)
    gm, gs, hps, fm, fs = stats
    for i in range(0, len(arr), 1024):
        t = torch.tensor(arr[i:i + 1024], device=DEV).unsqueeze(1)
        hp = (t - gm) / gs
        hp = hp - F.avg_pool2d(hp, 9, 1, 4)
        hp = hp / hps
        a = t - t.mean(dim=(2, 3), keepdim=True)
        sp = torch.fft.fftshift(torch.fft.fft2(a), dim=(2, 3))
        mag = torch.log1p(sp.abs() ** 2)
        mag = (mag - fm) / fs
        out[i:i + 1024, 0:1] = mag
        out[i:i + 1024, 1:2] = hp
    return out


def fit_stats(arr):
    gm = float(arr.mean()); gs = float(arr.std()) + 1e-6
    hl, ml = [], []
    for i in range(0, len(arr), 1024):
        t = torch.tensor(arr[i:i + 1024], device=DEV).unsqueeze(1)
        hp = (t - gm) / gs
        hp = hp - F.avg_pool2d(hp, 9, 1, 4)
        hl.append(hp.flatten())
        a = t - t.mean(dim=(2, 3), keepdim=True)
        mag = torch.log1p(torch.fft.fftshift(torch.fft.fft2(a), dim=(2, 3)).abs() ** 2)
        ml.append(mag.flatten())
    hp_all = torch.cat(hl); mag_all = torch.cat(ml)
    hps = float(hp_all.std()) + 1e-6
    fm = float(mag_all.mean()); fs = float(mag_all.std()) + 1e-6
    del hp_all, mag_all, hl, ml
    if DEV == 'cuda':
        torch.cuda.empty_cache()
    return (gm, gs, hps, fm, fs)


def main():
    D = find_data()
    patches = np.load(os.path.join(D, 'patches.npy')).astype(np.float32)
    tr = pd.read_csv(os.path.join(D, 'train.csv'))
    te = pd.read_csv(os.path.join(D, 'test.csv'))

    tr_idx = tr.tile_uid.values; te_idx = te.tile_uid.values
    Xtr_all = patches[tr_idx]; Xte = patches[te_idx]
    y = tr.rock_abundance.values.astype(np.float32)

    stats = fit_stats(Xtr_all)

    yl = np.log(np.clip(y, 1e-6, None))
    ym = yl.mean(); ys = yl.std() + 1e-6
    yt = ((yl - ym) / ys).astype(np.float32)

    bright = Xtr_all.mean(axis=(1, 2))

    Xg = compute_channels(Xtr_all, stats)
    Xteg = compute_channels(Xte, stats)
    yg = torch.tensor(yt, device=DEV)

    nf = 5
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(tr_idx))
    folds = np.array_split(perm, nf)

    oof = np.zeros(len(tr_idx)); test_pred = np.zeros(len(te_idx))
    for f in range(nf):
        va = folds[f]; trn = np.concatenate([folds[j] for j in range(nf) if j != f])
        ti = torch.tensor(trn, device=DEV); vi = torch.tensor(va, device=DEV)
        print(f'fold {f} start (train {len(trn)} / val {len(va)})', flush=True)
        out_va, out_te = train_fold(Xg[ti], yg[ti], [Xg[vi], Xteg], epochs=22)
        oof[va] = out_va
        test_pred += out_te / nf
        print(f'fold {f} resid-corr {resid_corr(oof[va], y[va], bright[va]):.4f}', flush=True)

    print(f'OOF pooled resid-corr {resid_corr(oof, y, bright):.4f}', flush=True)

    ysort = np.sort(y)
    tr_ranks = rankdata(test_pred) - 1
    pct = tr_ranks / (len(test_pred) - 1)
    pos = pct * (len(ysort) - 1)
    lo = np.floor(pos).astype(int); hi = np.ceil(pos).astype(int); fr = pos - lo
    mapped = ysort[lo] * (1 - fr) + ysort[hi] * fr
    mapped = np.clip(mapped, 0.0, 1.0)

    os.makedirs('working', exist_ok=True)
    sub = pd.DataFrame({'tile_uid': te_idx, 'rock_abundance': mapped})
    sub.to_csv('working/submission.csv', index=False)
    print('wrote working/submission.csv', sub.shape, flush=True)


if __name__ == '__main__':
    main()
