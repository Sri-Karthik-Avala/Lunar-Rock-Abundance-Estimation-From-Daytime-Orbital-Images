import os, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F

SEED = 1234
np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

BACKBONES = [
    {'name': 'convnext_tiny.fb_in22k_ft_in1k', 'tv': 'convnext_tiny', 'fdim': 768, 'res': 128, 'lr': 3e-4, 'ep': 12, 'folds': 5},
    {'name': 'convnext_small.fb_in22k_ft_in1k', 'tv': 'convnext_small', 'fdim': 768, 'res': 128, 'lr': 2.5e-4, 'ep': 11, 'folds': 4},
]


def find_data():
    for d in ['dataset/public', './dataset/public', '.', 'dataset', './data', 'input']:
        if os.path.exists(os.path.join(d, 'patches.npy')) and os.path.exists(os.path.join(d, 'train.csv')):
            return d
    raise FileNotFoundError('data not found')


def rankdata(a):
    a = np.asarray(a, dtype=np.float64)
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


class ScratchBackbone(nn.Module):
    def __init__(self, cin=3):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(cin, 32, 3, 1, 1, bias=False), gn(32), nn.ReLU(True))
        self.l1 = nn.Sequential(Blk(32, 32), Blk(32, 64, 2))
        self.l2 = nn.Sequential(Blk(64, 64), Blk(64, 128, 2))
        self.l3 = nn.Sequential(Blk(128, 128), Blk(128, 256, 2))
        self.l4 = nn.Sequential(Blk(256, 256, 2))

    def forward(self, x):
        return self.l4(self.l3(self.l2(self.l1(self.stem(x)))))


class HeadNet(nn.Module):
    def __init__(self, backbone, fdim, mode, res):
        super().__init__()
        self.backbone = backbone
        self.mode = mode
        self.res = res
        self.drop = nn.Dropout(0.3)
        self.fc1 = nn.Linear(fdim * 2, 256)
        self.fc2 = nn.Linear(256, 1)

    def feat(self, x):
        if self.mode == 'timm':
            return self.backbone.forward_features(x)
        if self.mode == 'tv':
            return self.backbone.features(x)
        return self.backbone(x)

    def forward(self, x):
        x = F.interpolate(x, size=self.res, mode='bilinear', align_corners=False)
        f = self.feat(x)
        a = F.adaptive_avg_pool2d(f, 1).flatten(1)
        m = F.adaptive_max_pool2d(f, 1).flatten(1)
        x = torch.cat([a, m], 1)
        x = self.drop(F.relu(self.fc1(self.drop(x))))
        return self.fc2(x).squeeze(1)


def build_model(spec):
    try:
        import timm
        bb = timm.create_model(spec['name'], pretrained=True, num_classes=0, global_pool='')
        return HeadNet(bb, spec['fdim'], 'timm', spec['res']), True
    except Exception as e:
        print(f"  timm {spec['name']} unavailable ({str(e)[:50]})", flush=True)
    try:
        import torchvision as tv
        bb = getattr(tv.models, spec['tv'])(weights='DEFAULT')
        return HeadNet(bb, spec['fdim'], 'tv', spec['res']), True
    except Exception as e:
        print(f"  torchvision {spec['tv']} unavailable ({str(e)[:50]}) -> from-scratch", flush=True)
        return HeadNet(ScratchBackbone(3), 256, 'seq', spec['res']), False


def d4(t, k):
    if k & 1:
        t = torch.flip(t, [3])
    return torch.rot90(t, k >> 1, [2, 3])


def augment(xb):
    n = xb.shape[0]
    c = 0.75 + 0.50 * torch.rand(n, 1, 1, 1, device=xb.device)
    xb = xb * c
    off = torch.zeros_like(xb)
    off[:, 0:1] = 0.30 * torch.randn(n, 1, 1, 1, device=xb.device)
    xb = xb + off + 0.02 * torch.randn_like(xb)
    return xb


def corr_loss(pred, target):
    p = pred - pred.mean(); t = target - target.mean()
    d = torch.sqrt((p * p).sum() * (t * t).sum()) + 1e-8
    return 1.0 - (p * t).sum() / d


def train_fold(spec, Xtr, ytr, Xva_list, bs=64):
    model, is_pre = build_model(spec)
    model = model.to(DEV)
    lr = spec['lr'] if is_pre else 1.3e-3
    epochs = spec['ep']
    opt = torch.optim.AdamW(model.parameters(), lr, weight_decay=1e-4)
    steps = (len(Xtr) + bs - 1) // bs
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, epochs=epochs, steps_per_epoch=steps, pct_start=0.25)
    hub = nn.SmoothL1Loss()
    for e in range(epochs):
        model.train(); pp = torch.randperm(len(Xtr), device=Xtr.device)
        for i in range(0, len(Xtr), bs):
            b = pp[i:i + bs]
            xb = augment(d4(Xtr[b], np.random.randint(8)))
            yb = ytr[b]
            opt.zero_grad()
            out = model(xb)
            loss = hub(out, yb) + 0.6 * corr_loss(out, yb)
            loss.backward(); opt.step(); sch.step()
    model.eval()
    preds = []
    with torch.no_grad():
        for xv in Xva_list:
            out = np.zeros(len(xv), dtype=np.float64)
            for i in range(0, len(xv), 256):
                xb = xv[i:i + 256]
                acc = torch.zeros(len(xb), device=DEV)
                for k in range(8):
                    acc += model(d4(xb, k))
                out[i:i + 256] = (acc / 8).cpu().numpy()
            preds.append(out)
    del model
    if DEV == 'cuda':
        torch.cuda.empty_cache()
    return preds


def fit_stats(arr):
    gm = float(arr.mean()); gs = float(arr.std()) + 1e-6
    h9, h3 = [], []
    for i in range(0, len(arr), 1024):
        t = torch.tensor((arr[i:i + 1024] - gm) / gs, device=DEV).unsqueeze(1)
        h9.append((t - F.avg_pool2d(t, 9, 1, 4)).flatten())
        h3.append((t - F.avg_pool2d(t, 3, 1, 1)).flatten())
    s9 = float(torch.cat(h9).std()) + 1e-6
    s3 = float(torch.cat(h3).std()) + 1e-6
    del h9, h3
    if DEV == 'cuda':
        torch.cuda.empty_cache()
    return (gm, gs, s9, s3)


def compute_channels(arr, stats):
    gm, gs, s9, s3 = stats
    out = torch.empty((len(arr), 3, 100, 100), dtype=torch.float32, device=DEV)
    for i in range(0, len(arr), 1024):
        t = torch.tensor((arr[i:i + 1024] - gm) / gs, device=DEV).unsqueeze(1)
        out[i:i + 1024, 0:1] = t
        out[i:i + 1024, 1:2] = (t - F.avg_pool2d(t, 9, 1, 4)) / s9
        out[i:i + 1024, 2:3] = (t - F.avg_pool2d(t, 3, 1, 1)) / s3
    return out


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

    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(tr_idx))
    n_tr = len(tr_idx); n_te = len(te_idx)

    oof_ranks = []; test_ranks = []
    for spec in BACKBONES:
        nf = spec['folds']
        folds = np.array_split(perm, nf)
        oof = np.zeros(n_tr); test_pred = np.zeros(n_te)
        for f in range(nf):
            va = folds[f]; trn = np.concatenate([folds[j] for j in range(nf) if j != f])
            ti = torch.tensor(trn, device=DEV); vi = torch.tensor(va, device=DEV)
            print(f"[{spec['tv']}] fold {f}/{nf} start", flush=True)
            out_va, out_te = train_fold(spec, Xg[ti], yg[ti], [Xg[vi], Xteg])
            oof[va] = out_va
            test_pred += out_te / nf
        sc = resid_corr(oof, y, bright)
        print(f"[{spec['tv']}] OOF resid-corr {sc:.4f}", flush=True)
        oof_ranks.append(rankdata(oof)); test_ranks.append(rankdata(test_pred))

    oof_ranks = np.array(oof_ranks); test_ranks = np.array(test_ranks)
    best_w = np.ones(len(BACKBONES)) / len(BACKBONES); best_sc = -1
    for w1 in np.linspace(0, 1, 41):
        w = np.array([w1, 1 - w1]) if len(BACKBONES) == 2 else best_w
        blend = (w[:, None] * oof_ranks).sum(0)
        s = resid_corr(blend, y, bright)
        if s > best_sc:
            best_sc = s; best_w = w
    print(f'AUTO-WEIGHT {np.round(best_w,3)} -> ensemble OOF resid-corr {best_sc:.4f}', flush=True)

    ens_test = (best_w[:, None] * test_ranks).sum(0)

    ysort = np.sort(y)
    tr_ranks = rankdata(ens_test) - 1
    pct = tr_ranks / (n_te - 1)
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
