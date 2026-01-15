# %% [markdown]
# # Trading Score MLP (v5.10) — TB dir/mag/signed + Meta-Label Gate (OOF / walk-forward)
#
# Goal:
# - Top-50 stable (mean high) and tail controlled (Worst5 / Min not exploding)
# - Top-10 and Top-50 should not diverge too much
#
# Key change vs v5.9:
# - Gate head is trained on a meta label: "Is this bar worth trading for THIS primary model?"
#   meta = 1{ (pred_trade_dir * y_target - META_EXTRA_COST) > 0 }
#   computed via out-of-fold / walk-forward predictions on TRAIN (no leakage).
#
# Keep:
# - TB continuous cost-aware label y_target for dir/mag/signed
# - rank loss, non-event suppression, wrong-direction emphasis, EMA
#
# Notes:
# - OOF meta labeling can be expensive. Start with META_FOLDS=4 and META_EPOCHS=60 to sanity check.
#
# %%
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from labels.label_generator import generate_tb_labels
from data.loader import load_kline_data
from modules.split_config import SplitConfig

# %% [markdown]
# ## Config
# %%
ASSET = "BTCUSDT"
MODE = "research"

BATCH_SIZE = 1024
EPOCHS = 450
LR = 1e-3
WEIGHT_DECAY = 1e-4
DEVICE = "cuda"

HIDDEN_DIMS = [128, 128, 64]
DROPOUT = 0.1
CLIP_VALUE = 5.0
SEED = 50

# TB target shaping
ENTRY_COST = 0.05
LAMBDA_DECAY = 1.0

# Gate: meta labeling
USE_META_GATE = True
META_EXTRA_COST = 0.03
META_FOLDS = 6
META_EPOCHS = 120
META_LR = 1e-3
META_BATCH_SIZE = 2048
META_MIN_TRAIN_BARS = 50_000
META_USE_EMA = True
META_EMA_DECAY = 0.999

# Loss weights (main)
LAMBDA_GATE = 0.50
GATE_WARMUP_EPOCHS = 20

LAMBDA_DIR = 0.15
LAMBDA_MAG = 0.30
LAMBDA_SIGNED = 1.00
ALPHA_YW = 4.0

# rank loss (main)
USE_RANK_LOSS = True
LAMBDA_RANK = 0.12
RANK_Q_CAND = 0.10
RANK_MIN_CAND = 64
RANK_PAIRS_PER_BATCH = 1024
RANK_MARGIN = 0.0
TEMP_DIR = 1.0

# non-event suppression
USE_NONEVENT_SUPPRESS = True
LAMBDA_NE_SIGNED = 0.10
LAMBDA_NE_MAG = 0.10
LAMBDA_NE_DIR = 0.00

# wrong-direction emphasis
USE_WRONG_EMPHASIS = True
WRONG_WARMUP_EPOCHS = 60
ALPHA_WRONG = 2.5
WRONG_W_CLIP = 4.0
APPLY_WRONG_TO_DIR = True
ALPHA_WRONG_DIR = 0.5

# stability
GRAD_CLIP_NORM = 3.0
USE_EMA_FOR_VAL = True
EMA_DECAY = 0.999

# checkpoint selection after warmups
CKPT_START_EPOCH = max(GATE_WARMUP_EPOCHS, WRONG_WARMUP_EPOCHS) + 10
print("CKPT_START_EPOCH:", CKPT_START_EPOCH)

# deploy selection
Q_GATE_DEPLOY = 0.005
Q_GATE_ALT = 0.010
DEPLOY_BETA = 1.0
DEPLOY_GAMMA = 0.5          # soft use of gate_prob in key within filtered set
EVAL_EXTRA_COST = META_EXTRA_COST

# -------------------------
# fast guard: avoid long runs dying due to missing config symbols
# -------------------------
def _sanity_check_globals():
    required = [
        'Q_GATE_DEPLOY','Q_GATE_ALT','DEPLOY_BETA','DEPLOY_GAMMA','EVAL_EXTRA_COST',
        'GATE_WARMUP_EPOCHS','WRONG_WARMUP_EPOCHS','CKPT_START_EPOCH',
        'META_EXTRA_COST','USE_META_GATE','USE_WRONG_EMPHASIS','USE_RANK_LOSS',
    ]
    missing = [k for k in required if k not in globals()]
    if missing:
        raise RuntimeError(f'Missing required globals: {missing}')

_sanity_check_globals()


# scale
PRED_SCALE_Q = 0.995

# %% [markdown]
# ## Seed
# %%
def set_seed(seed=1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)
device = DEVICE if (DEVICE == "cuda" and torch.cuda.is_available()) else "cpu"
print("Device:", device)

# %% [markdown]
# ## Splits
# %%
split_cfg = SplitConfig.load("configs/splits.json")
DATA_SPLIT = split_cfg.get(ASSET, mode=MODE)
print(DATA_SPLIT)

# %% [markdown]
# ## Load data + priors
# %%
data_file = Path(
    r"C:\Users\user\Desktop\binance-public-data-master\python\data\spot\monthly\klines"
    r"\BTCUSDT\5m\BTCUSDT-5m-2020_to_2025_10.csv"
)
df = load_kline_data(data_file).sort_index()

df = df.join(
    [
        pd.read_parquet("prior_features/hmm_features.parquet").sort_index(),
        pd.read_parquet("prior_features/markov_features.parquet").sort_index(),
        pd.read_parquet("prior_features/ngram_features.parquet").sort_index(),
        pd.read_parquet("prior_features/garch_features.parquet").sort_index(),
        pd.read_parquet("prior_features/pe_panel_features.parquet").sort_index(),
    ],
    how="inner",
)

assert df.index.is_monotonic_increasing
assert not df.index.has_duplicates
print("df rows:", len(df), "range:", df.index.min(), "->", df.index.max())

# %% [markdown]
# ## Baseline feature engineering (train-only z + clip)
# %%
from features.numba_feats.price_features import _compute_pct_return, _compute_volatility
from features.numba_feats.momentum_features import _compute_rsi

close = df["Close"].to_numpy(np.float32)

df["ret_1"] = _compute_pct_return(close, 1)
df["vol_20"] = _compute_volatility(close, 20)
df["rsi_14"] = _compute_rsi(close, 14)

feature_cols = ["ret_1", "vol_20", "rsi_14"]

train_mask_full = (df.index >= DATA_SPLIT["train"][0]) & (df.index <= DATA_SPLIT["train"][1])
mean = df.loc[train_mask_full, feature_cols].mean()
std = df.loc[train_mask_full, feature_cols].std().replace(0, np.nan)
df[feature_cols] = ((df[feature_cols] - mean) / std).clip(-CLIP_VALUE, CLIP_VALUE)

# %% [markdown]
# ## Generate cost-aware TB target y
# %%
tb = generate_tb_labels(df)

hit_up = tb.hit_up_labels.fillna(0).astype(np.int8)
hit_down = tb.hit_down_labels.fillna(0).astype(np.int8)

sign_evt = np.zeros(len(df), np.float32)
sign_evt[hit_up == 1] = 1.0
sign_evt[hit_down == 1] = -1.0

tau = np.full(len(df), np.nan, np.float32)
tau[hit_up == 1] = tb.hit_up_tau[hit_up == 1]
tau[hit_down == 1] = tb.hit_down_tau[hit_down == 1]

raw_strength = np.exp(-LAMBDA_DECAY * tau)
adj_strength = np.maximum(raw_strength - ENTRY_COST, 0.0)

y_target = sign_evt * adj_strength
y_target[np.isnan(y_target)] = 0.0

print("tb_event_rate (y!=0):", float((y_target != 0).mean()))

mag_target = np.abs(y_target).astype(np.float32)
dir_target = (y_target > 0).astype(np.float32)

# %% [markdown]
# ## Split helper
# %%
def split(df_, y, key):
    m = (df_.index >= DATA_SPLIT[key][0]) & (df_.index <= DATA_SPLIT[key][1])
    return df_.loc[m], y[m], m

df_train, y_train, m_train = split(df, y_target, "train")
df_val, y_val, m_val = split(df, y_target, "val")
df_test, y_test, m_test = split(df, y_target, "test")

mag_train, mag_val, mag_test = (mag_target[m_train], mag_target[m_val], mag_target[m_test])
dir_train, dir_val, dir_test = (dir_target[m_train], dir_target[m_val], dir_target[m_test])

print("Train:", df_train.index.min(), "->", df_train.index.max(), len(df_train))
print("Val  :", df_val.index.min(), "->", df_val.index.max(), len(df_val))
print("Test :", df_test.index.min(), "->", df_test.index.max(), len(df_test))

# %% [markdown]
# ## Build feature matrix
# %%
PRIOR_FEATURES = [
    c for c in df.columns
    if c.startswith(("ng", "mk", "hmm", "garch", "ewma", "vol", "pe", "m3", "m4"))
    and c not in feature_cols
]
X_cols = list(dict.fromkeys(feature_cols + PRIOR_FEATURES))

X_train = df_train[X_cols].values.astype(np.float32)
X_val = df_val[X_cols].values.astype(np.float32)
X_test = df_test[X_cols].values.astype(np.float32)

print("num features:", len(X_cols))

# %% [markdown]
# ## Scale + PRED_SCALE
# %%
evt_mask_train = y_train != 0
PRED_SCALE = float(np.quantile(np.abs(y_train[evt_mask_train]), PRED_SCALE_Q)) if evt_mask_train.any() else 1.0
print("PRED_SCALE:", PRED_SCALE)

# %% [markdown]
# ## Model
# %%
class TradingMLP3Head(nn.Module):
    def __init__(self, d_in: int, pred_scale: float):
        super().__init__()
        self.pred_scale = float(pred_scale)
        layers = []
        d = d_in
        for h in HIDDEN_DIMS:
            layers += [nn.Linear(d, h), nn.GELU(), nn.Dropout(DROPOUT)]
            d = h
        self.trunk = nn.Sequential(*layers)
        self.head_gate = nn.Linear(d, 1)
        self.head_dir = nn.Linear(d, 1)
        self.head_mag = nn.Linear(d, 1)

    def forward(self, x):
        h = self.trunk(x)
        gate_logits = self.head_gate(h).squeeze(-1)
        dir_logits = self.head_dir(h).squeeze(-1)
        mag_logit = self.head_mag(h).squeeze(-1)
        mag = self.pred_scale * torch.sigmoid(mag_logit)
        return gate_logits, dir_logits, mag

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        sd = model.state_dict()
        for k, v in sd.items():
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=(1.0 - self.decay))

    def state_dict(self):
        return {k: v.clone() for k, v in self.shadow.items()}

# %% [markdown]
# ## Utilities
# %%
def _sigmoid_np(x: np.ndarray):
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x))

def compute_scores_numpy(dir_logits: np.ndarray, mag: np.ndarray):
    p_up = _sigmoid_np(dir_logits)
    edge = 2.0 * p_up - 1.0
    signed_score = edge * mag
    score_abs = np.abs(signed_score)
    trade_dir = np.sign(signed_score)
    trade_dir[trade_dir == 0] = 1.0
    return signed_score, score_abs, trade_dir

def compute_scores_torch(dir_logits: torch.Tensor, mag: torch.Tensor):
    p_up = torch.sigmoid(dir_logits)
    edge = 2.0 * p_up - 1.0
    signed_score = edge * mag
    score_abs = torch.abs(signed_score)
    trade_dir = torch.sign(signed_score)
    trade_dir = torch.where(trade_dir == 0, torch.ones_like(trade_dir), trade_dir)
    return signed_score, score_abs, trade_dir

def candidate_mask_from_gate(gate_prob: torch.Tensor, *, q: float, min_cand: int) -> torch.Tensor:
    n = gate_prob.numel()
    k = int(max(min_cand, np.ceil(q * n)))
    k = int(min(k, n))
    if k <= 0:
        return torch.zeros_like(gate_prob, dtype=torch.bool)
    _, idx = torch.topk(gate_prob, k=k, largest=True, sorted=False)
    mask = torch.zeros_like(gate_prob, dtype=torch.bool)
    mask[idx] = True
    return mask

def pairwise_rank_loss(rank_score, reward, cand_mask, *, n_pairs: int, margin: float):
    if cand_mask.sum() < 2 or n_pairs <= 0:
        return 0.0 * rank_score.mean()
    idx_c = torch.nonzero(cand_mask, as_tuple=False).squeeze(1)
    m = idx_c.numel()
    ii = torch.randint(0, m, (n_pairs,), device=rank_score.device)
    jj = torch.randint(0, m, (n_pairs,), device=rank_score.device)
    same = ii == jj
    if same.any():
        jj[same] = (jj[same] + 1) % m
    i = idx_c[ii]
    j = idx_c[jj]
    ri, rj = reward[i], reward[j]
    si, sj = rank_score[i], rank_score[j]
    s = ri - rj
    sign = torch.sign(s)
    mask = sign != 0
    if mask.sum() == 0:
        return 0.0 * rank_score.mean()
    diff = (si - sj - float(margin)) * sign
    w = torch.clamp(torch.abs(s) / (PRED_SCALE + 1e-12), 0.0, 5.0).detach()
    return (torch.nn.functional.softplus(-diff[mask]) * (1.0 + w[mask])).mean()

def conf_from_dir_logit_np(dir_logit: np.ndarray):
    return _sigmoid_np(np.abs(dir_logit))  # (0.5,1)

def pnl_proxy_np(y: np.ndarray, trade_dir: np.ndarray, *, extra_cost: float):
    y = np.asarray(y, np.float64)
    td = np.asarray(trade_dir, np.float64)
    pnl = td * y
    pnl = pnl - float(extra_cost)
    pnl[np.abs(pnl) < 1e-15] = 0.0
    return pnl

def gate_profit_report(gate_prob: np.ndarray, trade_dir: np.ndarray, y: np.ndarray, *, q_list=(0.01,0.03), extra_cost: float = 0.03):
    gp = np.asarray(gate_prob, np.float64)
    td = np.asarray(trade_dir, np.float64)
    y = np.asarray(y, np.float64)
    base_pnl = pnl_proxy_np(y, td, extra_cost=extra_cost)
    y_pos = (base_pnl > 0).astype(np.int8)
    base_rate = float(y_pos.mean())
    rows = []
    for q in q_list:
        th = float(np.quantile(gp, 1.0 - float(q)))
        pred = (gp >= th).astype(np.int8)
        tp = int(((pred == 1) & (y_pos == 1)).sum())
        fp = int(((pred == 1) & (y_pos == 0)).sum())
        fn = int(((pred == 0) & (y_pos == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        rec = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        lift = (prec / base_rate) if (base_rate > 0 and np.isfinite(prec)) else np.nan
        rows.append({"q": float(q), "thr": th, "prec": float(prec), "rec": float(rec), "lift": float(lift)})
    return {"base_rate": base_rate, "rows": rows}

# %% [markdown]
# ## Walk-forward folds on TRAIN
# %%
def build_walkforward_folds(n: int, n_folds: int, min_train: int) -> Sequence[Tuple[np.ndarray, np.ndarray]]:
    n = int(n)
    n_folds = int(max(2, n_folds))
    edges = np.linspace(0, n, n_folds + 1, dtype=np.int64)
    folds = []
    for k in range(1, n_folds):
        o0, o1 = int(edges[k]), int(edges[k + 1])
        t1 = o0
        if t1 < int(min_train) or (o1 - o0) <= 0:
            continue
        tr = np.arange(0, t1, dtype=np.int64)
        oof = np.arange(o0, o1, dtype=np.int64)
        folds.append((tr, oof))
    return folds

# %% [markdown]
# ## Primary model training for meta labels (per fold)
# %%
def train_primary_for_meta(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    mag_tr: np.ndarray,
    dir_tr: np.ndarray,
    *,
    pred_scale: float,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    use_ema: bool,
    ema_decay: float,
    seed: int,
):
    set_seed(seed)
    model = TradingMLP3Head(X_tr.shape[1], pred_scale=pred_scale).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    ema = EMA(model, decay=ema_decay) if use_ema else None

    N = len(X_tr)
    for _ in range(int(epochs)):
        model.train()
        perm = np.random.permutation(N)
        for i0 in range(0, N, int(batch_size)):
            idx = perm[i0 : i0 + int(batch_size)]
            xb = torch.tensor(X_tr[idx], device=device)
            yb = torch.tensor(y_tr[idx], device=device)
            mb = torch.tensor(mag_tr[idx], device=device)
            db = torch.tensor(dir_tr[idx], device=device)

            opt.zero_grad(set_to_none=True)
            gate_logits, dir_logits, mag = model(xb)

            evt_mask = (yb != 0)
            if evt_mask.any():
                loss_dir = torch.nn.functional.binary_cross_entropy_with_logits(dir_logits[evt_mask], db[evt_mask])
                loss_mag = torch.nn.functional.smooth_l1_loss(mag[evt_mask], mb[evt_mask], reduction="mean")
                signed_all, _, _ = compute_scores_torch(dir_logits, mag)
                per = torch.nn.functional.smooth_l1_loss(signed_all[evt_mask], yb[evt_mask], reduction="none")
                w = 1.0 + float(ALPHA_YW) * (torch.abs(yb[evt_mask]) / (pred_scale + 1e-12))
                loss_signed = (w * per).mean()
            else:
                loss_dir = 0.0 * gate_logits.mean()
                loss_mag = 0.0 * gate_logits.mean()
                loss_signed = 0.0 * gate_logits.mean()

            non_evt = (yb == 0)
            if USE_NONEVENT_SUPPRESS and non_evt.any():
                signed_all, _, _ = compute_scores_torch(dir_logits, mag)
                loss_ne_signed = torch.nn.functional.smooth_l1_loss(
                    signed_all[non_evt], torch.zeros_like(signed_all[non_evt]), reduction="mean"
                )
                loss_ne_mag = torch.nn.functional.smooth_l1_loss(
                    mag[non_evt], torch.zeros_like(mag[non_evt]), reduction="mean"
                )
            else:
                loss_ne_signed = 0.0 * gate_logits.mean()
                loss_ne_mag = 0.0 * gate_logits.mean()

            loss = (LAMBDA_DIR * loss_dir) + (LAMBDA_MAG * loss_mag) + (LAMBDA_SIGNED * loss_signed) \
                   + (LAMBDA_NE_SIGNED * loss_ne_signed) + (LAMBDA_NE_MAG * loss_ne_mag)

            loss.backward()
            if GRAD_CLIP_NORM and GRAD_CLIP_NORM > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            opt.step()
            if ema is not None:
                ema.update(model)

    if ema is not None:
        model.load_state_dict(ema.state_dict(), strict=True)
    model.eval()
    return model

# %% [markdown]
# ## Build OOF meta gate labels on TRAIN
# %%
def build_meta_gate_oof(
    X_train: np.ndarray,
    y_train: np.ndarray,
    mag_train: np.ndarray,
    dir_train: np.ndarray,
    *,
    pred_scale: float,
    device: str,
    folds: int,
    min_train_bars: int,
    extra_cost: float,
):
    n = len(y_train)
    meta = np.zeros(n, np.float32)
    meta_valid = np.zeros(n, np.int8)

    wf = build_walkforward_folds(n, folds, min_train=min_train_bars)
    print(f"[META] folds requested={folds} -> usable={len(wf)} | min_train_bars={min_train_bars}")

    for k, (tr_idx, oof_idx) in enumerate(wf, 1):
        print(f"[META] fold {k}/{len(wf)} | train={len(tr_idx)} | oof={len(oof_idx)} | o0={oof_idx[0]} o1={oof_idx[-1]}")
        model = train_primary_for_meta(
            X_train[tr_idx], y_train[tr_idx], mag_train[tr_idx], dir_train[tr_idx],
            pred_scale=pred_scale, device=device,
            epochs=META_EPOCHS, batch_size=META_BATCH_SIZE,
            lr=META_LR, weight_decay=WEIGHT_DECAY,
            use_ema=META_USE_EMA, ema_decay=META_EMA_DECAY,
            seed=SEED + 1000 + k,
        )

        with torch.no_grad():
            xb = torch.tensor(X_train[oof_idx], device=device)
            _, d, m = model(xb)
            dir_logit = d.detach().cpu().numpy()
            mag = m.detach().cpu().numpy()

        _, _, td = compute_scores_numpy(dir_logit, mag)
        pnl = pnl_proxy_np(y_train[oof_idx], td, extra_cost=extra_cost)
        meta[oof_idx] = (pnl > 0).astype(np.float32)
        meta_valid[oof_idx] = 1

    meta_rate = float(meta[meta_valid == 1].mean()) if meta_valid.sum() > 0 else 0.0
    print(f"[META] meta_valid_rate={float(meta_valid.mean()):.4f} | meta_pos_rate(valid)={meta_rate:.4f}")
    return meta, meta_valid

# %% [markdown]
# ## Build gate targets (meta)
# %%
if USE_META_GATE:
    gate_train, gate_valid = build_meta_gate_oof(
        X_train, y_train, mag_train, dir_train,
        pred_scale=PRED_SCALE, device=device,
        folds=META_FOLDS, min_train_bars=META_MIN_TRAIN_BARS,
        extra_cost=META_EXTRA_COST,
    )
else:
    gate_train = (y_train != 0).astype(np.float32)
    gate_valid = np.ones(len(y_train), np.int8)

print("gate_valid_rate:", float(gate_valid.mean()))
print("gate_pos_rate(valid):", float(gate_train[gate_valid == 1].mean()) if gate_valid.sum() else 0.0)

# ---- Known-bad negatives that are ALWAYS unprofitable under pnl_proxy ----
# y==0 => pnl = -extra_cost < 0, so meta=0 is known; mark as valid negatives to teach gate to suppress zeros.
# |y| <= META_EXTRA_COST => even if direction correct, pnl - extra_cost <= 0 (too small edge); also valid negative.
known_bad = (y_train == 0) | (np.abs(y_train) <= (META_EXTRA_COST + 1e-12))
gate_train = gate_train.copy()
gate_valid = gate_valid.copy()
gate_train[known_bad] = 0.0
gate_valid[known_bad] = 1
print("[META] known_bad marked valid:", int(known_bad.sum()), "rate=%.4f" % float(known_bad.mean()))
print("gate_valid_rate(after known_bad):", float(gate_valid.mean()))
print("gate_pos_rate(valid, after known_bad):", float(gate_train[gate_valid == 1].mean()) if gate_valid.sum() else 0.0)

# pos_weight for gate BCE (only on valid rows)
gate_rate_valid = float(gate_train[gate_valid == 1].mean()) if gate_valid.sum() else 1e-6
pos_w = (1.0 - gate_rate_valid) / max(gate_rate_valid, 1e-6)
POS_WEIGHT = torch.tensor([pos_w], device=device, dtype=torch.float32)
print("pos_weight (gate):", float(POS_WEIGHT.item()))

# %% [markdown]
# ## Training (main) — gate=meta, dir/mag/signed=TB
# %%
model = TradingMLP3Head(X_train.shape[1], pred_scale=PRED_SCALE).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
ema = EMA(model, decay=EMA_DECAY)

def deploy_rank_key(score_abs: np.ndarray, dir_logit: np.ndarray, gate_prob: np.ndarray, *, beta: float, gamma: float):
    conf = conf_from_dir_logit_np(dir_logit)
    gp = np.clip(np.asarray(gate_prob, np.float64), 1e-12, 1.0)
    return np.asarray(score_abs, np.float64) * (conf ** float(beta)) * (gp ** float(gamma))

def pick_filter_key(score_abs, gate_prob, dir_logit, base_idx, *, q_gate: float, k: int, beta: float, gamma: float):
    base_idx = np.asarray(base_idx, dtype=np.int64)
    if len(base_idx) == 0:
        return base_idx
    g = np.asarray(gate_prob, np.float64)[base_idx]
    th = np.quantile(g, 1.0 - float(q_gate))
    cand = base_idx[g >= th]
    if len(cand) == 0:
        return cand
    kk = min(int(k), len(cand))
    key = deploy_rank_key(score_abs[cand], dir_logit[cand], gate_prob[cand], beta=beta, gamma=gamma)
    order = np.argsort(-key)
    return cand[order[:kk]]

CKPT_RATIOS = (0.002, 0.005, 0.010)
MIN_K = 10

def val_stress_metric(val_gate_prob, val_score_abs, val_dir_logit, val_trade_dir, y):
    N = len(y)
    all_idx = np.arange(N, dtype=np.int64)
    cut = N // 2
    segs = [all_idx, np.arange(0, cut, dtype=np.int64), np.arange(cut, N, dtype=np.int64)]
    scores = []
    for base_idx in segs:
        for r in CKPT_RATIOS:
            k = int(np.floor(r * len(base_idx)))
            k = max(MIN_K, k)
            k = min(k, len(base_idx))
            pick = pick_filter_key(val_score_abs, val_gate_prob, val_dir_logit, base_idx, q_gate=Q_GATE_DEPLOY, k=k, beta=DEPLOY_BETA, gamma=DEPLOY_GAMMA)
            if len(pick) == 0:
                scores.append(np.nan)
            else:
                pnl = pnl_proxy_np(y[pick], val_trade_dir[pick], extra_cost=EVAL_EXTRA_COST)
                scores.append(float(np.mean(pnl)))
    scores = np.array(scores, dtype=np.float64)
    worst = float(np.nanmin(scores))
    med = float(np.nanmedian(scores))
    ratio = float(worst / med) if (np.isfinite(med) and med != 0) else np.nan
    return worst, med, ratio

best = -1e9
best_ema_state = None
N = len(X_train)

gate_train_t = gate_train.astype(np.float32)
gate_valid_t = gate_valid.astype(np.int8)

for ep in range(1, EPOCHS + 1):
    model.train()
    losses = []
    perm = np.random.permutation(N)
    lam_gate_eff = 0.0 if ep <= GATE_WARMUP_EPOCHS else LAMBDA_GATE
    wrong_on = USE_WRONG_EMPHASIS and (ep > WRONG_WARMUP_EPOCHS)

    for i0 in range(0, N, BATCH_SIZE):
        idx = perm[i0 : i0 + BATCH_SIZE]
        xb = torch.tensor(X_train[idx], device=device)
        yb = torch.tensor(y_train[idx], device=device)
        mb = torch.tensor(mag_train[idx], device=device)
        db = torch.tensor(dir_train[idx], device=device)

        gb = torch.tensor(gate_train_t[idx], device=device)
        gv = torch.tensor(gate_valid_t[idx], device=device)  # 0/1

        opt.zero_grad(set_to_none=True)
        gate_logits, dir_logits, mag = model(xb)
        gate_prob = torch.sigmoid(gate_logits)

        tb_evt_mask = (yb != 0)
        non_evt = (yb == 0)

        # gate loss only on valid meta rows
        if lam_gate_eff > 0 and (gv.sum() > 0):
            m_valid = gv == 1
            loss_gate = torch.nn.functional.binary_cross_entropy_with_logits(
                gate_logits[m_valid], gb[m_valid], pos_weight=POS_WEIGHT
            )
        else:
            loss_gate = 0.0 * gate_logits.mean()

        signed_all, score_abs_all, _ = compute_scores_torch(dir_logits, mag)

        if tb_evt_mask.any():
            if APPLY_WRONG_TO_DIR and wrong_on:
                per_dir = torch.nn.functional.binary_cross_entropy_with_logits(
                    dir_logits[tb_evt_mask], db[tb_evt_mask], reduction="none"
                )
                y_sign = torch.sign(yb[tb_evt_mask])
                conf_wrong = torch.sigmoid(-y_sign * dir_logits[tb_evt_mask])
                w_wrong_dir = 1.0 + float(ALPHA_WRONG_DIR) * conf_wrong
                loss_dir = (w_wrong_dir * per_dir).mean()
            else:
                loss_dir = torch.nn.functional.binary_cross_entropy_with_logits(
                    dir_logits[tb_evt_mask], db[tb_evt_mask]
                )

            loss_mag = torch.nn.functional.smooth_l1_loss(mag[tb_evt_mask], mb[tb_evt_mask], reduction="mean")

            y_evt = yb[tb_evt_mask]
            y_abs = torch.abs(y_evt)
            w_base = 1.0 + float(ALPHA_YW) * (y_abs / (PRED_SCALE + 1e-12))

            if wrong_on:
                y_sign = torch.sign(y_evt)
                conf_wrong = torch.sigmoid(-y_sign * dir_logits[tb_evt_mask])
                w_wrong = 1.0 + float(ALPHA_WRONG) * conf_wrong
                w_wrong = torch.clamp(w_wrong, 1.0, float(WRONG_W_CLIP))
            else:
                w_wrong = 1.0

            w = w_base * w_wrong
            per = torch.nn.functional.smooth_l1_loss(signed_all[tb_evt_mask], y_evt, reduction="none")
            loss_signed = (w * per).mean()
        else:
            loss_dir = 0.0 * gate_logits.mean()
            loss_mag = 0.0 * gate_logits.mean()
            loss_signed = 0.0 * gate_logits.mean()

        if USE_NONEVENT_SUPPRESS and non_evt.any():
            loss_ne_signed = torch.nn.functional.smooth_l1_loss(
                signed_all[non_evt], torch.zeros_like(signed_all[non_evt]), reduction="mean"
            )
            loss_ne_mag = torch.nn.functional.smooth_l1_loss(
                mag[non_evt], torch.zeros_like(mag[non_evt]), reduction="mean"
            )
            loss_ne_dir = 0.0 * gate_logits.mean()
        else:
            loss_ne_signed = 0.0 * gate_logits.mean()
            loss_ne_mag = 0.0 * gate_logits.mean()
            loss_ne_dir = 0.0 * gate_logits.mean()

        if USE_RANK_LOSS:
            cand_mask = candidate_mask_from_gate(gate_prob, q=RANK_Q_CAND, min_cand=RANK_MIN_CAND)
            y_sign_all = torch.sign(yb)
            correct = torch.sigmoid((y_sign_all * dir_logits) / float(TEMP_DIR))
            g_eff = gb * gv.float()
            reward = g_eff * torch.abs(yb) * correct
            loss_rank = pairwise_rank_loss(score_abs_all, reward, cand_mask, n_pairs=RANK_PAIRS_PER_BATCH, margin=RANK_MARGIN)
        else:
            loss_rank = 0.0 * gate_logits.mean()

        loss = (
            (lam_gate_eff * loss_gate)
            + (LAMBDA_DIR * loss_dir)
            + (LAMBDA_MAG * loss_mag)
            + (LAMBDA_SIGNED * loss_signed)
            + (LAMBDA_RANK * loss_rank)
            + (LAMBDA_NE_SIGNED * loss_ne_signed)
            + (LAMBDA_NE_MAG * loss_ne_mag)
            + (LAMBDA_NE_DIR * loss_ne_dir)
        )

        loss.backward()
        if GRAD_CLIP_NORM and GRAD_CLIP_NORM > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        opt.step()
        ema.update(model)
        losses.append(float(loss.item()))

    # VAL
    model.eval()
    with torch.no_grad():
        cur_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if USE_EMA_FOR_VAL:
            model.load_state_dict(ema.state_dict(), strict=True)

        vg, vd, vm = model(torch.tensor(X_val, device=device))
        val_gate_prob = torch.sigmoid(vg).cpu().numpy()
        val_dir_logits = vd.cpu().numpy()
        val_mag = vm.cpu().numpy()

        if USE_EMA_FOR_VAL:
            model.load_state_dict(cur_state, strict=True)

    _, val_score_abs, val_trade_dir = compute_scores_numpy(val_dir_logits, val_mag)
    val_worst, val_med, val_ratio = val_stress_metric(val_gate_prob, val_score_abs, val_dir_logits, val_trade_dir, y_val)

    if ep >= CKPT_START_EPOCH and val_worst > best:
        best = val_worst
        best_ema_state = ema.state_dict()

    if ep % 10 == 0 or ep == 1:
        rep = gate_profit_report(val_gate_prob, val_trade_dir, y_val, q_list=(Q_GATE_DEPLOY, Q_GATE_ALT), extra_cost=EVAL_EXTRA_COST)
        print(
            f"Epoch {ep:03d} | TrainLoss={float(np.mean(losses)):.5f} | "
            f"ValWorst={val_worst:+.5f} | BestWorst(after={CKPT_START_EPOCH})={best:+.5f} | "
            f"lam_gate={lam_gate_eff:.2f} | wrong_on={int(wrong_on)}"
        )
        print("  [VAL gate@q profit] base_rate=%.4f rows=%s" % (rep["base_rate"], rep["rows"]))

if USE_EMA_FOR_VAL and best_ema_state is not None:
    model.load_state_dict(best_ema_state, strict=True)
    print("✔ Loaded best EMA model (after CKPT_START_EPOCH)")
else:
    print("⚠ No EMA best found after CKPT_START_EPOCH.")

# %% [markdown]
# ## Final evaluation (TEST)
# %%
model.eval()
with torch.no_grad():
    tg, td, tm = model(torch.tensor(X_test, device=device))
    test_gate_prob = torch.sigmoid(tg).cpu().numpy()
    test_dir_logits = td.cpu().numpy()
    test_mag = tm.cpu().numpy()

_, test_score_abs, test_trade_dir = compute_scores_numpy(test_dir_logits, test_mag)

def eval_filter_key(gate_prob, score_abs, dir_logit, trade_dir, y, *, q_gate: float, K: int, beta: float, gamma: float, extra_cost: float):
    base_idx = np.arange(len(y), dtype=np.int64)
    pick = pick_filter_key(score_abs, gate_prob, dir_logit, base_idx, q_gate=q_gate, k=min(K, len(y)), beta=beta, gamma=gamma)
    yy = y[pick] if len(pick) else np.array([], dtype=np.float64)
    pnl = pnl_proxy_np(yy, trade_dir[pick], extra_cost=extra_cost) if len(pick) else np.array([np.nan])

    event_rate = float((yy != 0).mean()) if len(yy) else np.nan
    zero_rate = float((yy == 0).mean()) if len(yy) else np.nan
    pos_rate = float((yy > 0).mean()) if len(yy) else np.nan
    neg_rate = float((yy < 0).mean()) if len(yy) else np.nan

    min_pnl = float(np.nanmin(pnl))
    worst5 = float(np.nanmean(np.sort(pnl)[: min(5, len(pnl))]))
    worst3 = float(np.nanmean(np.sort(pnl)[: min(3, len(pnl))]))
    return float(np.nanmean(pnl)), float(np.nanmean(pnl > 0)), min_pnl, worst3, worst5, event_rate, zero_rate, pos_rate, neg_rate

def print_eval(label, q_gate):
    print(f"=== FINAL TOP-K ({label}, q={q_gate:.3f}) ===")
    for K in (10, 20, 50, 100, 200, 500, 1000):
        mp, wr, mn, w3, w5, er, zr, pr, nr = eval_filter_key(
            test_gate_prob, test_score_abs, test_dir_logits, test_trade_dir, y_test,
            q_gate=q_gate, K=K, beta=DEPLOY_BETA, gamma=DEPLOY_GAMMA, extra_cost=EVAL_EXTRA_COST
        )
        if K == 50:
            print(
                f"Top-{K:4d} | MeanPnL={mp:+.4f} | WinRate={wr:.3f} | "
                f"MinPnL={mn:+.4f} | Worst3Mean={w3:+.4f} | Worst5Mean={w5:+.4f} | "
                f"event={er:.2f} zero={zr:.2f} pos={pr:.2f} neg={nr:.2f}"
            )
        else:
            print(
                f"Top-{K:4d} | MeanPnL={mp:+.4f} | WinRate={wr:.3f} | "
                f"MinPnL={mn:+.4f} | Worst3Mean={w3:+.4f} | event={er:.2f} zero={zr:.2f}"
            )

print_eval(f"DEPLOY=FILTER(q_gate)+KEY(|signed|*conf^{DEPLOY_BETA}) (pnl - extra_cost={EVAL_EXTRA_COST})", Q_GATE_DEPLOY)
print()
print_eval(f"DEPLOY=FILTER(q_gate)+KEY(|signed|*conf^{DEPLOY_BETA}) (pnl - extra_cost={EVAL_EXTRA_COST})", Q_GATE_ALT)

# %% [markdown]
# ## Optional: export signal tape (TEST)
# %%
def export_signal_tape_minimal(
    *,
    out_dir="signals",
    run_tag=None,
    index: pd.Index,
    gate_prob=None,
    dir_logit=None,
    mag=None,
    trade_dir=None,
    meta: dict = None,
    compression="snappy",
    keep_float32=True,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if run_tag is None:
        run_tag = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = out_dir / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    gate_prob = np.asarray(gate_prob)
    dir_logit = np.asarray(dir_logit)
    mag = np.asarray(mag)

    if trade_dir is None:
        _, _, trade_dir = compute_scores_numpy(dir_logit, mag)

    conf = conf_from_dir_logit_np(dir_logit)
    key_raw = np.abs((2*_sigmoid_np(dir_logit)-1.0) * mag) * conf

    df_out = pd.DataFrame(
        {
            "score_gate_prob": gate_prob,
            "score_dir_logit": dir_logit,
            "score_mag": mag,
            "score_trade_dir": trade_dir,
            "score_conf": conf,
            "score_key_raw": key_raw,
        },
        index=index,
    )
    if keep_float32:
        for c in df_out.columns:
            df_out[c] = df_out[c].astype(np.float32)

    tape_path = run_dir / "signal_tape.parquet"
    df_out.to_parquet(tape_path, index=True, compression=compression)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_tag": run_tag,
        "rows": int(len(df_out)),
        "columns": list(df_out.columns),
        "files": {"signal_tape": "signal_tape.parquet"},
        "meta": meta or {},
    }
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("✅ saved:", run_dir)
    return run_dir, df_out

# Example:
# meta = {"asset": ASSET, "v": "v5.10_metaGate_OOF", "meta_extra_cost": float(META_EXTRA_COST), "deploy_beta": float(DEPLOY_BETA)}
# run_tag = f"{ASSET}_5m_{DATA_SPLIT['test'][0]}_{DATA_SPLIT['test'][1]}_{meta['v']}"
# export_signal_tape_minimal(out_dir="signals", run_tag=run_tag, index=df_test.index,
#                            gate_prob=test_gate_prob, dir_logit=test_dir_logits, mag=test_mag, trade_dir=test_trade_dir, meta=meta)
