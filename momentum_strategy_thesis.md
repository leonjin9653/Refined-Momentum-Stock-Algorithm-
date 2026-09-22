# Regime-Conditioned Momentum Strategy: Thesis & Design Specification

## 1. Thesis Statement

Momentum — the tendency of an asset that has been rising (or falling) to continue doing so over intermediate horizons — is one of the most persistent and well-documented anomalies in financial markets. However, naive momentum strategies suffer from two well-known failure modes: **whipsaw** (false signals during choppy, non-trending markets) and **momentum crashes** (sharp, violent reversals after extended low-volatility uptrends).

This strategy's core thesis is that momentum's edge can be meaningfully improved by:

1. **Only trading momentum when the market is structurally in a trending regime** (filtering out the conditions where momentum is known to fail),
2. **Sizing risk by volatility rather than by fixed capital**, so that no single trade — regardless of the asset's current turbulence — can disproportionately impact the portfolio, and
3. **Confirming signals across multiple independent lenses** (price structure, trend acceleration, magnitude, and directional strength) rather than relying on any single indicator, which reduces false positives.

The strategy is explicitly *not* optimized to maximize backtested return. It is optimized to maximize **risk-adjusted, out-of-sample robustness** — the property that actually matters when a strategy meets real, unseen data.

---

## 2. Market Regime Framework

Before any momentum logic runs, the market is classified along two independent axes:

| Axis | States | Source |
|---|---|---|
| Trend | `trending` / `ranging` | ADX (Average Directional Index), threshold at 25 |
| Volatility | `high_vol` / `low_vol` | 20-day realized volatility, percentile-ranked over a 100-day lookback, split at the 70th percentile |

These combine into four regimes: `trending_low_vol`, `trending_high_vol`, `ranging_low_vol`, `ranging_high_vol`.

**Trend is the strategy switch** — momentum only trades in the two `trending_*` regimes. **Volatility is a risk-tuning knob** — it doesn't change *whether* the strategy trades, only *how large and how tightly-stopped* each position is.

---

## 3. Signal Ensemble (Entry Logic)

Rather than a single indicator, entries require agreement across four independent signals, each casting a vote of `+1` (bullish), `0` (neutral), or `-1` (bearish). A long entry requires a minimum combined score (e.g. at least 3 of 4 signals agreeing bullish).

### 3.1 Donchian Breakout
**What it measures:** whether price is making a genuine structural breakout.
**Definition:** price closes above the highest close of the prior *N* days (commonly N=20).
**Why it's in the ensemble:** it's a pure price-action signal — it doesn't care about the *shape* of the move, only that a new extreme has been set. This anchors the ensemble in price structure rather than derived indicators.

### 3.2 MACD Histogram Slope
**What it measures:** whether trend momentum is *accelerating*, not just present.
**Definition:** MACD histogram (12-day EMA − 26-day EMA, minus its own 9-day EMA signal line) is positive and increasing over the last few bars.
**Why it's in the ensemble:** a breakout with decelerating momentum behind it is a weaker signal than one with accelerating momentum. This adds a "second derivative" check that price level alone can't give.

### 3.3 Rate of Change (ROC)
**What it measures:** the magnitude of the recent move.
**Definition:** `ROC = (price_today − price_n_days_ago) / price_n_days_ago × 100`.
**Why it's in the ensemble:** filters out breakouts that are technically valid but trivially small — a new 20-day high by 0.1% is a very different animal from one by 8%. A minimum ROC threshold screens out marginal breakouts.

### 3.4 ADX Slope
**What it measures:** whether directional strength is *building*, using the ADX already computed by `RegimeDetector`.
**Definition:** ADX is not just above the trending threshold (25), but rising over the last several bars.
**Why it's in the ensemble:** the regime filter already requires ADX > 25 to trade at all; this signal adds a finer-grained confirmation that trend strength is still developing rather than plateauing or rolling over.

**Ensemble decision rule:** sum the four votes. Enter long only if score ≥ 3 (i.e. at least 3 of 4 agree) *and* the regime is `trending_*`. This dual-gate (regime + ensemble) is deliberately conservative — it will miss some moves, but the goal is signal quality, not signal quantity.

---

## 4. Position Sizing — Volatility Targeting

Rather than a fixed share count or fixed dollar amount per trade, position size is scaled inversely to the asset's current volatility, so that every trade carries approximately the same amount of *risk*, regardless of how turbulent the underlying currently is.

**Definition:**
```
position_size = (target_risk_pct × portfolio_capital) / (ATR_n × price)
```
where `ATR_n` is the n-day Average True Range (a volatility measure in price units) and `target_risk_pct` is a fixed fraction of capital the strategy is willing to risk per trade (e.g. 1%).

**Why this matters:** this is the actual technique used by real time-series momentum strategies (this framework closely mirrors the methodology in Moskowitz, Ooi & Pedersen's *"Time Series Momentum"*, 2012) — sizing by volatility rather than notional keeps a calm, steadily-trending asset and a wildly volatile one contributing comparable risk to the portfolio, instead of the volatile one dominating losses.

---

## 5. Exit Logic — ATR Trailing Stop (Chandelier Stop)

**Definition:**
```
stop_level = highest_close_since_entry − k × ATR_n
```
The stop only ever moves up (for a long position), never down, and the position exits when price closes below the current stop level.

**Why this over a fixed-target exit:** the strategy's earlier prototype (RSI/Bollinger-band based) exits as soon as price nears the upper band — which caps upside artificially and defeats the purpose of momentum ("cut losses short, let winners run"). A trailing stop lets a strong trend keep contributing profit for as long as it persists, while still providing a hard, mechanical loss limit.

---

## 6. Regime-Conditioned Parameters

The same signal and exit logic runs in both trending regimes, but with different risk tuning:

| Parameter | `trending_low_vol` | `trending_high_vol` |
|---|---|---|
| ATR stop multiplier (k) | Tighter (e.g. 2×) | Wider (e.g. 3.5×) |
| Position size | Larger (steadier trend, more confidence) | Smaller (turbulent conditions, wider stops eat more risk budget per unit size) |

This isn't two strategies — it's one strategy with regime-aware dials, which is the practical distinction to be clear about in interviews: the *logic* doesn't change, only the *risk parameters*.

---

## 7. Multi-Timeframe Confirmation

**Definition:** resample price data to weekly bars, run the same trend classification (ADX-based) on the weekly series, and require the weekly trend to also read `trending` before acting on a daily-level entry signal.

**Why this matters:** a daily-level breakout inside a weekly ranging market is far more likely to be noise than a genuine trend. This is a standard technique for reducing whipsaw — trading in the direction of the higher timeframe while timing entries on the lower one.

---

## 8. Momentum Crash Filter

**The problem this solves:** momentum strategies have a well-documented failure mode — a long, low-volatility uptrend can reverse violently and rapidly (the 2009 momentum crash following the 2008 financial crisis is the canonical example; this dynamic is studied directly in Daniel & Moskowitz's *"Momentum Crashes"*). These crashes tend to follow *specifically* the conditions this strategy trades most confidently in: extended `trending_low_vol` periods.

**Definition:** monitor for a sharp spike in the volatility percentile (e.g. a jump of more than X percentile points within a short window, such as 5 days) occurring immediately after a sustained `trending_low_vol` regime. When triggered: flatten all open positions and block new entries for a cooldown period (e.g. 10 trading days).

**Why this is the most "quant" component:** it directly encodes an awareness of momentum's specific, empirically-documented tail risk, rather than relying on the trailing stop alone to catch it — trailing stops react to price *after* the move happens; this filter tries to recognize the *precondition* for the move.

---

## 9. Validation Methodology — Walk-Forward Analysis

**The problem this solves:** optimizing all parameters (Donchian lookback, ROC threshold, ATR multiplier, vote threshold, etc.) once against the full historical dataset and reporting that backtest's return is close to worthless — it is very likely overfit to that specific history.

**Definition:** split the full price history into sequential windows. For each step:
1. **Train window:** grid-search parameters to maximize a risk-adjusted metric (Sharpe ratio) on this slice only.
2. **Test window:** apply those fixed parameters, unseen, to the *next* slice.
3. Roll both windows forward and repeat.
4. Stitch together only the test-window results into one continuous equity curve.

**Why this is the headline result:** the stitched, out-of-sample curve is the honest estimate of how the strategy would have actually performed if run live, re-fitting periodically — not the inflated number from a single in-sample optimization.

---

## 10. Performance Evaluation

Final results should be reported using, at minimum:

- **Sharpe Ratio** — return per unit of total volatility
- **Sortino Ratio** — return per unit of *downside* volatility (more relevant for a strategy explicitly trying to manage crash risk)
- **Maximum Drawdown** — the worst peak-to-trough decline, a key risk-tolerance metric
- **Calmar Ratio** — annualized return divided by max drawdown
- **Transaction cost drag** — modeled explicitly (a few basis points per round-trip trade), since ignoring costs is one of the most common — and most easily criticized — mistakes in student backtests
- **Benchmark comparison** — versus simple buy-and-hold on the same asset/period, so the strategy's value-add (or lack thereof) is explicit rather than implied

All of the above should be reported from the **walk-forward out-of-sample results**, not the in-sample optimization.

---

## 11. Summary — The Story This Tells

Signal generation (ensemble voting) → regime filtering (trend as switch, volatility as risk dial) → risk-scaled sizing (volatility targeting) → adaptive exits (ATR trailing stop) → structural confirmation (multi-timeframe) → tail-risk awareness (crash filter) → honest validation (walk-forward) → risk-adjusted reporting.

That chain — signal quality, regime awareness, risk-scaled execution, and rigorous out-of-sample validation — is the actual substance of a systematic quant strategy, and is what this project should communicate.

---

### References (for further reading / citation)
- Moskowitz, T., Ooi, Y.H., & Pedersen, L.H. (2012). *Time Series Momentum.* Journal of Financial Economics.
- Daniel, K., & Moskowitz, T. (2016). *Momentum Crashes.* Journal of Financial Economics.
