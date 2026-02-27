# Architecture v1 — BTC 5m Contract Trading Model

> **One-liner:** 5m K-bar → technical indicators + priors (HMM / Markov / N-gram / GARCH / PermEnt / ACF / tail-stats, strict as-of) → sequence window N=256 → TCN/DRN backbone → TB 3-head pretrain (direction / pace / vol, uncertainty weighting) → action-bank oracle ranking main training (cost-aware, coarse→fine) → Phase A/B fine-tune (warmup + L2-SP) → rolling / worst-case acceptance → cadence / drift-triggered retraining → deployment.

---

## 0 · Overview & Goal

| Item | Spec |
|---|---|
| Asset | BTC-USDT perpetual (Binance or equivalent) |
| Bar | 5-minute OHLCV |
| Objective | Output a discrete **contract action** (direction / leverage / TP / SL / time-stop) each bar; evaluate on **cost-inclusive** net PnL |
| Non-goal (v1) | Portfolio allocation, multi-asset, order-book features, funding-rate P&L |
| Two-stage approach | **Pretrain** (learn market representation via Triple-Barrier multi-task) → **Main Training** (learn action selection via oracle ranking) |

---

## 1 · Data & Leakage Contract

### 1.1 Raw Data

| Field | Detail |
|---|---|
| Source | Exchange REST / WebSocket; fallback: aggregator with ≥ 2 y history |
| Fields per bar | `open, high, low, close, volume, close_time` |
| Minimum history | 2 years (≈ 210 k bars) for train+val+test |
| Missing bar policy | Forward-fill OHLC from previous close; volume = 0; flag `is_filled=1`; if gap > 3 consecutive bars → exclude segment |

### 1.2 Cost Model (v1 hard-coded)

| Param | Value | Note |
|---|---|---|
| Taker fee | 0.045 % | Entry & exit both taker (worst-case) |
| Maker fee | 0.02 % | v2: model may specify limit orders |
| Slippage (one-way) | 0.02 % | Conservative for BTC |
| Round-trip cost (taker) | **(0.045 + 0.02) × 2 = 0.13 %** | Applied in oracle & backtest |
| Funding rate | **Ignored in v1** | v2: incorporate 8-h funding as cost |
| Min notional | 5 USDT | Actions below min notional → filtered |
| Liquidation | Isolated margin; liq price = entry × (1 ∓ 1/leverage × (1 − maint_margin_rate)) | maint_margin_rate = 0.4 % for BTC ≤ 50× |

### 1.3 Time-Split & Leakage Rules

```
|<--- Train (70 %) --->|<-- Val (15 %) -->|<-- Test (15 %) -->|
                       ↑ purge+embargo     ↑ purge+embargo
```

| Rule | Spec |
|---|---|
| Split | **Chronological only**; no shuffle, no K-fold |
| Purge | Between each split boundary, discard `H_max` bars from the **end** of the earlier segment. H_max = max(TB horizon, max time-stop) = **24 bars** |
| Embargo | Additional `E = 16` bars after purge gap; these bars exist in data but are **never** used as label anchors |
| Total gap | purge (24) + embargo (16) = **40 bars ≈ 3.33 h** between each split |
| Prior models | **P2 per-split fit**: fit/calibrate **only on train split**; val/test use frozen parameters in causal (as-of) inference only |
| Technical indicators | Must be **causal / rolling** (use data ≤ t only); no centred filters |
| Oracle labels | Use future path `[t+1 … t+max_time_stop]`; purge/embargo already accounts for this |

> **P0 Rule — No Exceptions:** Any feature, label, or normalisation that touches data from a future split **invalidates the entire experiment**. If in doubt, over-purge.

### 1.4 Normalisation / Scaling

| What | Method | Boundary rule |
|---|---|---|
| Price-derived features (ret, log-ret) | Already stationary; no scaling | — |
| Volume | Rolling z-score, lookback = 256 bars | Causal (window ending at t) |
| Indicator outputs | Per-feature robust-scale (median / IQR) fitted on **train split only** → frozen for val/test | Refit when priors refit |
| Prior model outputs | Same as above | Same as above |

---

## 2 · Feature Layer

### 2.1 Technical Indicators

All indicators are **causal** (use data ≤ t only).

| # | Indicator | Params (v1) | Output dim |
|---|---|---|---|
| 1 | Log return | `log(close_t / close_{t-1})` | 1 |
| 2 | Rolling volatility | `std(log_ret, window=w)` for w ∈ {16, 64, 256} | 3 |
| 3 | RSI | period = 14 | 1 |
| 4 | MACD | fast=12, slow=26, signal=9 → (macd, signal, hist) | 3 |
| 5 | Bollinger %B | window=20, std=2 | 1 |
| 6 | ATR | period=14 | 1 |
| 7 | OBV z-score | rolling z, window=64 | 1 |
| 8 | VWAP deviation | `(close − vwap) / atr` | 1 |
| 9 | Volume z-score | rolling z, window=256 | 1 |
| 10 | Bar range | `(high − low) / close` | 1 |
| 11 | `is_filled` flag | 0/1 from missing-bar handling | 1 |
| **Total tech dim** | | | **16** |

### 2.2 Prior Models

Purpose: compress market state into regime / probability / risk / structure / predictability summaries for the backbone.

| # | Model | One-liner | Output dim |
|---|---|---|---|
| 1 | **HMM** (Gaussian, 3 states) | Market as hidden-regime switching; output = state posterior probs + most-likely state | 4 |
| 2 | **Markov transition** (return-bucket, 5 bins) | Discretise returns into buckets, learn transition matrix; output = row of transition probs from current bucket | 5 |
| 3 | **N-gram** (n=3, 8 symbols) | Discretise short patterns to symbol sequences; output = current-pattern probability + surprisal | 2 |
| 4 | **GARCH(1,1)** | Conditional volatility estimator; output = σ\_t, σ\_t / rolling\_vol ratio | 2 |
| 5 | **EGARCH(1,1)** | Asymmetric vol (leverage effect); output = σ\_t, leverage-term value | 2 |
| 6 | **Permutation entropy** (m=5, τ=1, window=64) | Sequence orderliness → predictability proxy; output = PE value | 1 |
| 7 | **ACF lag-1,2,5 + rolling slope** (window=32) | Trend continuation vs mean-reversion signal | 4 |
| 8 | **Tail-event stats** | Rolling tail-event rate (\|ret\| > 3σ, window=256), current drawdown %, bars-since-last-shock | 3 |
| 9 | **(v2 only) Change-point score** | CUSUM or Bayesian online CPD; output = change-point score | (1) |
| **Total prior dim (v1)** | | | **23** |

**Total feature dim d = 16 (tech) + 23 (prior) = 39.**

### 2.3 Prior Fit / Inference Boundary Rules

| Stage | What is allowed |
|---|---|
| **Train** | Fit (estimate parameters): HMM EM, Markov transition counts, N-gram counts, GARCH MLE, PE (no fitting needed), ACF/slope (no fitting needed), tail stats rolling |
| **Val / Test** | **Frozen parameters** from train; causal forward inference only (e.g., HMM forward algorithm with fixed A/B/π; GARCH filter with fixed ω/α/β) |
| **Deployment** | Use last-refit parameters; causal inference bar-by-bar |
| **Refit trigger** | Scheduled every **8 weeks** OR when prior NLL drift (see §9) exceeds threshold |

### 2.4 Prior Numerical Safety

All prior model outputs are **clipped to [−5, +5]** after z-scoring. If a prior computation returns NaN or Inf:
- Replace with **0.0**
- Set companion flag `prior_{name}_valid = 0` (appended to feature vector as extra dim)
- This adds up to 5 extra binary dims (HMM, Markov, N-gram, GARCH, EGARCH) → **total d can be up to 44 in worst case**; v1 baseline assumes healthy priors → d=39.

---

## 3 · Time Windowing

| Param | v1 Default | Ablation |
|---|---|---|
| Window length N | **256** (≈ 21.3 h) | 128, 512 |
| Stride | 1 bar (every bar is a sample) | — |
| Padding | **None** — discard first N − 1 bars of each contiguous segment | — |
| Input shape | `X_t ∈ ℝ^{256 × 39}` | — |

---

## 4 · Backbone — TCN / DRN

### 4.1 Architecture

| Param | v1 Default |
|---|---|
| Type | Temporal Convolutional Network (causal dilated conv + residual) |
| Kernel size | 3 |
| Dilation schedule | [1, 2, 4, 8, 16, 32, 64] → **7 layers** |
| Receptive field | (kernel − 1) × Σ(dilations) + 1 = 2 × 127 + 1 = **255** ≈ N |
| Channel schedule | [64, 64, 128, 128, 256, 256, 256] |
| Activation | GELU |
| Normalisation | LayerNorm per residual block (after conv, before activation) |
| Dropout | 0.1 (spatial dropout on channel dim) |
| Residual | 1×1 conv when channel changes; identity otherwise |
| Pooling | **Last-hidden** (z\_t = output[:, −1, :]) → `z_t ∈ ℝ^{256}` |
| Ablation pooling | Mean-pool; Attention-pool (single-head, learnable query) |

### 4.2 Input Projection

- Linear `d=39 → 64` before first TCN block (with GELU + LayerNorm).
- No positional encoding in v1 (TCN has implicit positional structure via causal convolution).

---

## 5 · Pretrain — Triple-Barrier Multi-Task

### 5.0 Purpose

Train the backbone to produce a market-state representation **before** teaching it to select actions. This reduces noise in main training and improves generalisation.

### 5.1 Triple-Barrier Setup

| Param | v1 Default | Ablation |
|---|---|---|
| Horizon H | **16 bars** (80 min) | 12 |
| Barrier type | **k × vol** | — |
| Vol definition | `σ_t = std(log_ret, window=64)` — same rolling vol used in tech features | — |
| k (upper = lower) | **1.5** | 1.0, 2.0 |
| Upper barrier | `close_t × (1 + k × σ_t)` | — |
| Lower barrier | `close_t × (1 − k × σ_t)` | — |
| Time barrier | bar `t + H` | — |

### 5.2 Pretrain Heads

All heads share the same backbone; each is a 2-layer MLP head on top of `z_t ∈ ℝ^{256}`.

#### Head 1 — TB Outcome (direction)

| Item | Spec |
|---|---|
| Target | Which barrier is hit first: `UP` (upper), `DOWN` (lower), `TIME` (neither within H bars) |
| Classes | 3: {UP=0, DOWN=1, TIME=2} |
| Head arch | Linear(256→128) → GELU → Dropout(0.1) → Linear(128→3) |
| Loss | Cross-entropy with **inverse-frequency class weights** (capped at 3×) |

#### Head 2 — Time-to-Hit (pace)

| Item | Spec |
|---|---|
| Target | If UP or DOWN hit, how many bars τ to hit? Bin: `FAST` (τ ≤ 4), `MED` (5 ≤ τ ≤ 10), `SLOW` (11 ≤ τ ≤ 16), `NONE` (TIME outcome) |
| Classes | 4: {FAST=0, MED=1, SLOW=2, NONE=3} |
| Head arch | Linear(256→128) → GELU → Dropout(0.1) → Linear(128→4) |
| Loss | Cross-entropy with inverse-frequency class weights (capped at 3×) |

#### Head 3 — Future Realised Volatility Bin

| Item | Spec |
|---|---|
| Target | `rv_t = std(log_ret[t+1 … t+H])`; bin into **quintiles** computed on **train split only** |
| Classes | 5: {Q1=0 … Q5=4} |
| Quintile boundaries | 20th/40th/60th/80th percentiles of `rv` on **train split**; freeze for val/test |
| Head arch | Linear(256→128) → GELU → Dropout(0.1) → Linear(128→5) |
| Loss | Cross-entropy (add inverse-freq weights if skew > 2:1) |

### 5.3 Multi-Task Loss

**Uncertainty weighting** (Kendall et al. 2018):

```
L_total = Σ_i [ (1 / (2 × exp(log_σ²_i))) × L_i  +  0.5 × log_σ²_i ]
```

| Param | Spec |
|---|---|
| Learnable params | `log_σ²_i` for i ∈ {outcome, pace, vol}, initialised to **0.0** (⇒ σ² = 1) |
| Gradient | σ\_i learned jointly; no manual weight tuning needed |
| Class imbalance | Each CE head uses **inverse-frequency class weights**, max cap = **3.0** |

### 5.4 Pretrain Training Config

| Param | Value |
|---|---|
| Optimiser | AdamW (β1=0.9, β2=0.999, weight\_decay=1e-4) |
| LR | 1e-3, cosine decay to 1e-5 |
| Batch size | 512 |
| Max epochs | 100 (early stop patience = 10 on val total loss) |
| Purge | Last H=16 bars of train removed; first E=16 bars of val removed |

### 5.5 Pretrain Deliverable

- Save: `backbone_pretrain.pt`, `pretrain_meta.json` (val losses per head, class distributions, vol quintile boundaries).
- The 3 pretrain heads are **discarded** after pretrain; only backbone weights carry forward.

---

## 6 · Main Training — Action Bank + Oracle Ranking

### 6.1 Action Bank (v1)

| Factor | Levels | Values |
|---|---|---|
| Direction | 3 | LONG, SHORT, FLAT |
| Leverage | 3 | 3×, 5×, 10× |
| Take-Profit | 3 | 0.5 %, 1.0 %, 2.0 % |
| Stop-Loss | 3 | 0.3 %, 0.6 %, 1.2 % |
| Time-Stop | 3 | 6 bars (30 min), 12 bars (1 h), 24 bars (2 h) |

**FLAT** collapses all other factors → **1 unique FLAT action**.

**Total actions = (2 directions × 3 × 3 × 3 × 3) + 1 FLAT = 162 + 1 = 163.**

### 6.2 Oracle Label Generation (Offline)

For every bar `t` in {train, val, test} and every non-FLAT action `a`:

1. **Entry** at `close_t` (next-bar-open approximation).
2. **Simulate forward** bar-by-bar from `t+1`:
   - Check liquidation: price crosses liq price.
   - Check SL hit: `low ≤ entry × (1 − SL)` for LONG; `high ≥ entry × (1 + SL)` for SHORT.
   - Check TP hit: `high ≥ entry × (1 + TP)` for LONG; `low ≤ entry × (1 − TP)` for SHORT.
   - Check time-stop: elapsed bars ≥ time\_stop.
3. **Same-bar priority (worst-case):** Liquidation > SL > TP > Time-stop.
4. **Cost deduction:** `net_roe = gross_roe − round_trip_cost` (0.13 %).
5. **Store per (t, a):** `net_roe`, `exit_bar`, `exit_code ∈ {TP, SL, LIQ, TIME}`, `gross_roe`, `holding_bars`.

For FLAT: `net_roe = 0`, `exit_bar = 0`, `exit_code = FLAT`.

**Oracle table shape:** `(T, 163)` per metric column.

### 6.3 Coarse → Fine

#### 6.3.1 Coarse Bucket Definition

Group by **(direction, leverage)**:

| Bucket ID | Direction | Leverage | # Fine actions inside |
|---|---|---|---|
| 0 | FLAT | — | 1 |
| 1 | LONG | 3× | 27 |
| 2 | LONG | 5× | 27 |
| 3 | LONG | 10× | 27 |
| 4 | SHORT | 3× | 27 |
| 5 | SHORT | 5× | 27 |
| 6 | SHORT | 10× | 27 |
| **Total** | | | **163 actions in 7 buckets** |

#### 6.3.2 Coarse Oracle Aggregation

For each (t, bucket\_b):

```
coarse_score(t, b) = CVaR_q( { net_roe(t, a) : a ∈ bucket_b } )
```

| Param | Value | Rationale |
|---|---|---|
| q (CVaR quantile) | **0.25** | Average of worst 25 % of fine actions in bucket — conservative |
| FLAT bucket | coarse\_score = 0 always | Benchmark: only pick a directional bucket if its CVaR > 0 |

#### 6.3.3 Two-Stage Inference

| Stage | Input | Output |
|---|---|---|
| **Stage 1 (Coarse)** | Score all 7 buckets → rank | Top-B = **3** bucket IDs |
| **Stage 2 (Fine)** | Score all fine actions in top-B buckets (≤ 82 actions) → rank | Top-1 action |

### 6.4 Main Training Objective

#### 6.4.1 Coarse Head

| Item | Spec |
|---|---|
| Head arch | Linear(256→128) → GELU → Dropout(0.1) → Linear(128→7) |
| Loss | **ListMLE** (listwise ranking loss) |
| Target | Sort 7 buckets by `coarse_score(t, b)` descending → ground-truth permutation |

#### 6.4.2 Fine Head

| Item | Spec |
|---|---|
| Head arch | Linear(256 + 8 bucket\_embed → 128) → GELU → Dropout(0.1) → Linear(128→1) (scalar score per action) |
| Bucket embedding | Learned embedding dim = **8** per bucket; concatenated to z\_t |
| Loss | **Pairwise margin ranking loss**: for each (t), sample pairs (a⁺, a⁻) where `net_roe(a⁺) > net_roe(a⁻)` |
| Pair sampling | Per sample t: **16 pairs**, stratified — 8 cross-sign (positive vs negative roe) + 8 random |
| Margin | `margin = 0.0` (require correct ordering only) |
| Auxiliary loss | **ROE regression** (MSE on `net_roe`, weight = **0.1**) to anchor score scale |
| Total fine loss | `L_fine = L_pairwise + 0.1 × L_roe_mse` |

#### 6.4.3 Combined Main Loss

```
L_main = L_coarse_listmle + L_fine_pairwise + 0.1 × L_fine_roe_mse
```

(Coarse and fine heads are trained jointly; gradients from both flow into backbone during Phase B.)

### 6.5 Oracle Purge

Max future lookahead = `max_time_stop = 24 bars`.

| Boundary | Purge from earlier split | Embargo on later split |
|---|---|---|
| Train → Val | Last **24** bars of train | First **16** bars of val |
| Val → Test | Last **24** bars of val | First **16** bars of test |

---

## 7 · Pretrain → Main Handoff

| Step | Detail |
|---|---|
| 1 | Load `backbone_pretrain.pt` (TCN weights + input projection) |
| 2 | Discard pretrain heads |
| 3 | Attach coarse head (7 outputs) + fine head (per-action scorer) — **random init (Kaiming uniform)** |
| 4 | Freeze backbone → enter Phase A |

---

## 8 · Fine-Tuning Schedule

### Phase A — Head Alignment

| Param | Value |
|---|---|
| Trainable | Coarse head + Fine head **only** (backbone frozen) |
| Optimiser | AdamW (β1=0.9, β2=0.999, wd=1e-4) |
| LR (head) | **3e-4** |
| LR schedule | Linear warmup 5 % of Phase A steps → constant |
| Duration | **15 % of total training steps** OR val loss plateau for 5 epochs (whichever first) |
| Purpose | Align randomly-initialised heads to frozen pretrained backbone features |

### Phase B — Full Fine-Tune

| Param | Value |
|---|---|
| Trainable | All parameters (backbone + heads) |
| LR (head) | **3e-4** (continues from Phase A schedule) |
| LR (backbone) | **3e-5** (= head\_lr × 0.1) |
| LR schedule | Cosine decay to 1e-6 for both groups |
| Backbone warmup | First **5 %** of Phase B steps: backbone LR ramps 0 → 3e-5 |
| L2-SP regularisation | `λ_sp = 0.01`; penalty = `λ_sp × ‖θ_backbone − θ_pretrain‖²` |
| Duration | **85 % of total steps** |
| Early stopping | Patience **15 epochs** on val metric: `mean(rolling_7d_net_sharpe)` on val oracle simulation |
| Total epochs (A+B) | Max **80**; typical convergence 40–60 |
| Batch size | **256** |

---

## 9 · Retraining Cadence

| Component | Schedule | Early refit trigger |
|---|---|---|
| **Pretrain** (backbone) | Every **12 weeks** | Val pretrain total loss degrades > 10 % |
| **Main training** | Every **4 weeks** (warm-start from last checkpoint) | Rolling\_7d Sharpe < 0.0 for 3 consecutive days |
| **Priors (P2 refit)** | Every **8 weeks** | Prior NLL drift exceeds threshold |

### Prior NLL Drift Monitor

For each prior model with a likelihood (HMM, GARCH, EGARCH, N-gram):

1. At train time, record `NLL_train_mean` and `NLL_train_std` (on train split).
2. In production, compute rolling 1-week average NLL on live data.
3. **Trigger:** `NLL_rolling − NLL_train_mean > 2 × NLL_train_std` → flag drift → refit.
4. For non-likelihood priors (PE, ACF, tail stats): monitor feature distribution via **KS-test**, threshold p < 0.01.

---

## 10 · Evaluation & Acceptance

### 10.1 Metrics Definitions

All metrics on **selected action's net\_roe** per bar (aggregated to daily where noted).

| Metric | Definition |
|---|---|
| `daily_pnl` | Sum of net\_roe of all actions triggered in that calendar day |
| `rolling_7d_sharpe` | `mean(daily_pnl, 7d) / std(daily_pnl, 7d) × √365` |
| `rolling_30d_sharpe` | Same over 30-day window |
| `rolling_30d_return` | Cumulative net\_roe over 30 d |
| `worst_1d` | Within each 30-d window, the single worst daily\_pnl |
| `p05_daily` | 5th percentile of daily\_pnl across all test days |
| `max_drawdown` | Max peak-to-trough of cumulative net\_roe |
| `win_rate` | Fraction of non-FLAT trades with net\_roe > 0 |
| `avg_roe_per_trade` | Mean net\_roe of non-FLAT trades |
| `flat_rate` | Fraction of bars where model selects FLAT |
| `trades_per_day` | Average non-FLAT trades per day |

### 10.2 Selection Rule

| Param | v1 Default |
|---|---|
| Method | **Top-1** from Stage-2 fine ranking |
| Threshold gate | Execute only if `fine_score(top-1) > θ`; else FLAT. **θ = 0.0** (tune on val) |
| Max concurrent positions | **1** (new action only after previous exits) |

### 10.3 Acceptance Criteria (Pass / Fail on Test Split)

| Criterion | Pass threshold | Rationale |
|---|---|---|
| `rolling_30d_sharpe` (median) | ≥ **1.0** | Minimum risk-adjusted return |
| `p05_daily` | ≥ **−0.5 %** | Worst-case daily loss bounded |
| `worst_1d` (median across windows) | ≥ **−1.5 %** | Single-day catastrophe cap |
| `max_drawdown` | ≤ **5 %** | Drawdown tolerance |
| `avg_roe_per_trade` | > **0.02 %** | Must exceed fees |
| `flat_rate` | ≤ **80 %** | Model must trade enough to evaluate |
| `trades_per_day` | ≥ **3** | Same rationale |

> **Fail on any one criterion → model does not deploy.** Investigate, retrain, re-evaluate.

### 10.4 Ablation Plan (v1 Minimum Experiments)

| # | Experiment | Variable | Expected outcome |
|---|---|---|---|
| A1 | Pretrained vs from-scratch | Backbone init | Pretrained ≥ +0.3 Sharpe |
| A2 | Window N | 128 / **256** / 512 | 256 sweet spot |
| A3 | TB horizon H | **16** / 12 | 16 better for 80-min horizon |
| A4 | Coarse-only vs Coarse→Fine | Skip Stage-2 | Coarse→Fine improves avg\_roe |
| A5 | Pooling | Last-hidden / mean / attention | Pick best on val |
| A6 | With priors vs without priors | Feature set (d=39 vs d=16) | Priors ≥ +0.1 Sharpe |

---

## 11 · Deployment (Inference Pipeline)

Every 5 minutes when a new bar closes:

```
 1. Ingest new bar → append to rolling buffer (≥ 512 bars)
 2. Compute technical indicators (causal, on buffer)
 3. Compute prior features (as-of, frozen params, forward-filter)
 4. Assemble x_t ∈ ℝ^{39}
 5. Build X_t = [x_{t-255} ... x_t] ∈ ℝ^{256×39}
 6. Forward pass:
    a. z_t = backbone(X_t)
    b. coarse_scores = coarse_head(z_t) → top-B=3 buckets
    c. fine_scores = fine_head(z_t, bucket_embed) for actions in top-B
    d. best_action = argmax(fine_scores)
 7. Threshold gate: if fine_score(best_action) < θ → FLAT
 8. Position check: if already in position → skip (wait for exit)
 9. Execute: submit order with action params
10. Risk filter: reject if leverage × notional > max_risk_budget
```

### Latency Budget

| Step | Target |
|---|---|
| Data ingest + features | < 1 s |
| Prior inference | < 0.5 s |
| Model forward pass | < 0.2 s |
| Order submission | < 0.5 s |
| **Total** | **< 3 s** |

---

## 12 · Risks & Mitigations

### P0 — Project-Killing / Leakage

| ID | Risk | Impact | Mitigation |
|---|---|---|---|
| P0-1 | **Prior models fit on full dataset** | All features contain future info → backtest invalid | Enforce P2 per-split fit; automated assertion: `prior.last_fit_ts <= train_end` |
| P0-2 | **Scaler fit on val/test data** | Feature normalisation leaks future distribution | Scaler fitted on train only; freeze; automated assertion in pipeline |
| P0-3 | **Purge/embargo not applied** | Train/val share oracle label futures | Reusable `purge_embargo()` function with unit tests on known boundaries |
| P0-4 | **Lookahead in indicators** | Centred MAs or future-peeking pandas ops | Code review checklist: every indicator uses `data[:t+1]` only; fuzz test live vs batch |
| P0-5 | **Oracle same-bar priority wrong** | TP checked before SL → oracle is optimistic | Hard-coded priority: LIQ > SL > TP > TIME; unit test with crafted bar data |
| P0-6 | **Vol quintile boundaries from val/test** | Pretrain Head 3 labels leak | Boundaries computed on train only; frozen; tested |

### P1 — High Risk

| ID | Risk | Impact | Mitigation |
|---|---|---|---|
| P1-1 | **Regime shift post-deployment** | Model trained on one regime, deployed in another | Drift monitors §9; retrain cadence; worst-case acceptance criteria |
| P1-2 | **Cost model too optimistic** | Real slippage > 0.02 % in volatile markets | Log real fills vs model assumption; recalibrate; v1 uses worst-case taker |
| P1-3 | **Prior numerical instability** | NaN/Inf during extreme moves | Clip to [−5,+5]; fallback=0 with validity flag (§2.4) |
| P1-4 | **Overfitting on oracle** | Model memorises (t→best\_action) | Ranking loss > classification; L2-SP; early stop; monitor train-val gap |
| P1-5 | **Position overlap** | New signal while position open | v1: max 1 position; queue signals |
| P1-6 | **TB class imbalance** | TIME dominates if barriers too wide | Monitor distribution; narrow k or use focal loss if TIME > 60 % |
| P1-7 | **Model always picks FLAT** | Safe but useless | flat\_rate ≤ 80 % acceptance criterion; consider FLAT penalty |
| P1-8 | **Entry price assumption** | `close_t` ≠ actual fill at `t+1 open` | v1 absorbs in slippage; v2: use `open_{t+1}` if available |

### P2 — Deferred to v2

| ID | Risk |
|---|---|
| P2-1 | No funding rate modelling |
| P2-2 | Single asset only (BTC) |
| P2-3 | No order-book / microstructure features |
| P2-4 | No Kelly / portfolio sizing |
| P2-5 | Change-point model not in v1 |
| P2-6 | No limit-order strategy |

---

## 13 · v1 Hyperparameter Summary

| Category | Param | Value |
|---|---|---|
| **Data** | Bar | 5 min |
| | Min history | 2 years |
| | Train / Val / Test | 70 / 15 / 15 % (chrono) |
| | Purge | 24 bars |
| | Embargo | 16 bars |
| | Total gap per boundary | 40 bars (3.33 h) |
| | Round-trip cost | 0.13 % |
| **Features** | Tech dim | 16 |
| | Prior dim | 23 |
| | Total d | 39 |
| **Window** | N | 256 |
| **Backbone** | Type | TCN causal dilated conv + residual |
| | Layers | 7 |
| | Channels | [64, 64, 128, 128, 256, 256, 256] |
| | Kernel | 3 |
| | Dilations | [1, 2, 4, 8, 16, 32, 64] |
| | Receptive field | 255 |
| | Dropout | 0.1 spatial |
| | Pooling | Last-hidden |
| | z\_t dim | 256 |
| **Pretrain** | H | 16 bars |
| | k (barrier) | 1.5 × σ\_t (σ = rolling-64 vol) |
| | Head 1 | 3 classes (UP/DOWN/TIME) |
| | Head 2 | 4 classes (FAST/MED/SLOW/NONE) |
| | Head 3 | 5 classes (vol quintiles, train boundaries) |
| | Loss | Uncertainty weighting (log σ² init=0) |
| | Class weights | Inverse-freq, cap 3× |
| | LR | 1e-3 cosine → 1e-5 |
| | Batch | 512 |
| | Epochs | max 100, patience 10 |
| **Action Bank** | Directions | LONG / SHORT / FLAT |
| | Leverage | 3× / 5× / 10× |
| | TP | 0.5 / 1.0 / 2.0 % |
| | SL | 0.3 / 0.6 / 1.2 % |
| | Time-stop | 6 / 12 / 24 bars |
| | Total actions | 163 |
| **Coarse→Fine** | Buckets | 7 (direction × leverage + FLAT) |
| | CVaR q | 0.25 |
| | Top-B | 3 |
| **Main Loss** | Coarse | ListMLE |
| | Fine | Pairwise ranking + 0.1 × ROE MSE |
| | Pairs / sample | 16 (8 cross-sign + 8 random) |
| **Fine-tune** | Phase A (head only) | 15 % steps, lr=3e-4 |
| | Phase B (all) | 85 % steps, lr\_head=3e-4, lr\_bb=3e-5 |
| | BB warmup | 5 % of Phase B |
| | L2-SP λ | 0.01 |
| | Early stop | patience 15, val rolling\_7d Sharpe |
| | Epochs | max 80 |
| | Batch | 256 |
| **Retraining** | Pretrain | 12 weeks |
| | Main | 4 weeks |
| | Priors | 8 weeks |
| | Prior drift | NLL > 2σ |
| **Eval** | Selection | Top-1, threshold θ=0.0 |
| | Max positions | 1 |
| | 30d Sharpe (median) | ≥ 1.0 |
| | p05\_daily | ≥ −0.5 % |
| | worst\_1d (median) | ≥ −1.5 % |
| | max\_drawdown | ≤ 5 % |
| | avg\_roe\_per\_trade | > 0.02 % |
| | flat\_rate | ≤ 80 % |
| | trades\_per\_day | ≥ 3 |

---

## 14 · v1 TODO Checklist

Development modules ordered by dependency.

### Phase 0 — Infrastructure & Data

- [ ] `config/v1_config.yaml` — All hyperparameters from §13; every script reads from here
- [ ] `data/fetch_bars.py` — Download/stream 5m OHLCV; store as Parquet (partitioned by date)
- [ ] `data/clean_bars.py` — Missing-bar handling (forward-fill, `is_filled`), dedup, timestamp validation
- [ ] `data/split.py` — Chronological train/val/test split; output boundary timestamps
- [ ] `data/purge_embargo.py` — Given boundaries + H + E → valid index masks. **Unit tests required.**

### Phase 1 — Features

- [ ] `features/technical.py` — 16 technical indicators; causal assertion per function
- [ ] `features/priors/hmm.py` — HMM 3-state Gaussian: fit on train, filter on val/test, save/load params
- [ ] `features/priors/markov_transition.py` — 5-bin return-bucket transition matrix: fit on train, apply
- [ ] `features/priors/ngram.py` — N-gram (n=3, 8 symbols): count on train, surprisal on val/test
- [ ] `features/priors/garch.py` — GARCH(1,1) + EGARCH(1,1): MLE on train, filter; NaN/clip guard
- [ ] `features/priors/permutation_entropy.py` — PE (m=5, τ=1, w=64): pure rolling
- [ ] `features/priors/acf_slope.py` — ACF lag-1/2/5 + slope (w=32): pure rolling
- [ ] `features/priors/tail_stats.py` — Tail rate / drawdown / bars-since-shock: pure rolling
- [ ] `features/prior_runner.py` — Orchestrate priors; enforce fit boundary; output prior matrix
- [ ] `features/scaling.py` — Robust-scaler: fit on train, freeze, apply; save/load
- [ ] `features/build_features.py` — End-to-end pipeline: bars → tech + priors → scaled → save

### Phase 2 — Labels

- [ ] `labels/triple_barrier.py` — TB labels (outcome, time-to-hit, rv quintile bins) per bar
- [ ] `labels/action_bank.py` — Define 163 actions, bucket mapping, metadata
- [ ] `labels/oracle_runner.py` — For each (t, action): simulate exit → net\_roe, exit\_bar, exit\_code. **Unit tests with hand-crafted paths.**
- [ ] `labels/coarse_oracle.py` — Fine oracle → coarse CVaR-0.25 scores per bucket

### Phase 3 — Dataset & DataLoader

- [ ] `dataset/sequence_dataset.py` — `SequenceDataset(features, labels, N=256)`: returns `(X_t, tb_labels, oracle_labels)`; discards first N−1
- [ ] `dataset/pair_sampler.py` — Generate 16 (a⁺, a⁻) pairs per sample per §6.4.2
- [ ] `dataset/dataloader.py` — Split-aware DataLoader; train: shuffle at sample level; val/test: sequential

### Phase 4 — Model

- [ ] `model/tcn.py` — TCN backbone: input projection → 7 dilated causal conv blocks → last-hidden pooling. Config-driven.
- [ ] `model/pretrain_heads.py` — 3 MLP heads + uncertainty weighting module
- [ ] `model/action_heads.py` — Coarse head (7-way) + Fine head (per-action scorer, bucket embed)
- [ ] `model/loss.py` — ListMLE, pairwise margin ranking, ROE MSE, uncertainty weighting composite

### Phase 5 — Training

- [ ] `train/pretrain.py` — Pretrain loop: backbone + 3 heads, AdamW, cosine LR, early stop, checkpoint
- [ ] `train/main_train.py` — Phase A (freeze backbone) → Phase B (unfreeze, differential LR, L2-SP), early stop
- [ ] `train/utils.py` — Checkpoint IO, metric logging (TB/W&B), LR schedule helpers

### Phase 6 — Evaluation

- [ ] `eval/backtest.py` — Simulate trades on test (1-position, threshold gate) → daily\_pnl, rolling metrics
- [ ] `eval/metrics.py` — All metric functions from §10.1
- [ ] `eval/acceptance.py` — Automated pass/fail against §10.3; output report JSON + human-readable
- [ ] `eval/ablation_runner.py` — Run experiments A1–A6; output comparison table

### Phase 7 — Deployment

- [ ] `deploy/inference.py` — Real-time: bar → features → priors → window → model → action → threshold
- [ ] `deploy/risk_filter.py` — Max-position, notional cap, leverage cap, post-liquidation cooldown
- [ ] `deploy/drift_monitor.py` — Prior NLL, feature KS-test, rolling Sharpe; alerting
- [ ] `deploy/order_manager.py` — Exchange order submission; position tracking; exit handling

### Tests (cross-cutting)

- [ ] `tests/test_no_leakage.py` — (1) prior fit ts ≤ train\_end, (2) scaler train-only, (3) indicator causality fuzz, (4) purge/embargo correctness
- [ ] `tests/test_oracle.py` — Oracle with crafted paths: TP hit, SL hit, LIQ, TIME, same-bar conflict
- [ ] `tests/test_model.py` — Shape tests: (B,256,39)→z(B,256); head output dims; gradient flow
- [ ] `tests/test_leakage_integration.py` — Full pipeline on synthetic data; verify no future info

---

*Document version: v1.0 — ready for implementation.*
