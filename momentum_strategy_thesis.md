# Regime-Conditioned Momentum Strategy: Thesis & Design Specification

## 1. Thesis Statement

Momentum — the tendency of an asset that has been rising (or falling) to continue doing so over intermediate horizons — is one of the most persistent and well-documented anomalies in financial markets. However, naive momentum strategies suffer from two well-known failure modes: **whipsaw** (false signals during choppy, non-trending markets) and **momentum crashes** (sharp, violent reversals after extended low-volatility uptrends).

This strategy's core thesis is that momentum's edge can be meaningfully improved by:

1. **Only trading momentum when the market is structurally in a trending regime** (filtering out the conditions where momentum is known to fail),
2. **Sizing risk by volatility rather than by fixed capital**, so that no single trade — regardless of the asset's current turbulence — can disproportionately impact the portfolio, and
3. **Confirming signals across multiple independent lenses** (price structure, trend acceleration, magnitude, and directional strength) rather than relying on any single indicator, which reduces false positives.

The strategy is explicitly *not* optimized to maximize backtested return. It is optimized to maximize **risk-adjusted, out-of-sample robustness** — the property that actually matters when a strategy meets real, unseen data.

**What out-of-sample testing found.** Each idea was tested on 18 tickers (index ETFs, mega-caps, and 2002-era large caps), with walk-forward parameter selection and 2006–2025 out of sample:

1. **Trend-regime gating did not hold up.** Gating entries on an ADX trending regime lowered Sharpe on 13 of 18 tickers and was removed (§2). The regime now only sets stop width, which tested as a tie against a single fixed stop width (§6).
2. **Volatility-scaled sizing is kept**, since it fixes the risk taken per trade by construction, but it was not tested against an alternative sizing rule. At 1% risk per trade on a single stock, the strategy only ever has part of its capital invested.
3. **The four-signal ensemble did not beat a plain Donchian breakout** at default parameters (median −0.05 Sharpe; better on 7 of 18 tickers).

Overall, the strategy's median out-of-sample Sharpe is 0.37 against 0.60 for buy-and-hold, and it beat buy-and-hold on 1 of 18 tickers. Its drawdowns are far smaller (median −10.4% vs −58.8%), but mostly because it is in the market about half the time with partial positions, not because of better timing.

---

## 2. Market Regime Framework

Before any momentum logic runs, the market is classified along two independent axes:

| Axis | States | Source |
|---|---|---|
| Trend | `trending` / `ranging` | ADX (Average Directional Index), threshold at 25 |
| Volatility | `high_vol` / `low_vol` | 20-day realized volatility, percentile-ranked over a 100-day lookback, split at the 70th percentile |

These combine into four regimes: `trending_low_vol`, `trending_high_vol`, `ranging_low_vol`, `ranging_high_vol`.

**Trend is the strategy switch** — momentum only trades in the two `trending_*` regimes. **Volatility is a risk-tuning knob** — it doesn't change *whether* the strategy trades, only *how large and how tightly-stopped* each position is.

**Outcome: the trend switch was tested and rejected; the regime is now a risk setting only.** Three ways of using the regime were put through the same walk-forward on 18 tickers (2006–2025 out of sample):

| Variant | Median OOS Sharpe | Better than the trend gate on |
|---|---|---|
| Trend gate at ADX 25 (as specified above) | 0.33 | — |
| No gate: enter in any regime, regime sets stop width only | 0.37 | 13 of 18 (median +0.11) |
| Trend gate at ADX 20 | 0.33 | 9 of 18 (median −0.01) |

The gate mostly reduced time in the market (36% vs 54%) rather than picking better moments to be in it: removing it helped most on strong risers (AAPL, AMZN, GOOGL, GE) and hurt on weaker stocks (INTC, XOM, JNJ, PFE), where being out more often was the advantage. Removing it also deepened the median max drawdown (−7.2% → −10.4%), so Calmar did not improve. A likely cause is that ADX lags: by the time it crosses 25, much of the move has happened.

The volatility axis is kept, but only weakly supported: setting the stop width by volatility regime versus one fixed width (the midpoint) was a tie — the fixed width was better on 11 of 18 tickers by a median of +0.01 Sharpe, with median Sharpe 0.31 vs 0.37 and mean 0.40 vs 0.37 (see §6).

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

*Update:* the regime half of this dual gate was dropped after testing (see §2), so entries now depend on the ensemble score alone. The ADX slope vote (§3.4) is unaffected; it still checks that ADX is rising, it just no longer sits on top of an ADX > 25 gate.

---

## 4. Position Sizing — Volatility Targeting

Rather than a fixed share count or fixed dollar amount per trade, position size is scaled inversely to the asset's current volatility, so that every trade carries approximately the same amount of *risk*, regardless of how turbulent the underlying currently is.

**Definition:**
```
shares = (target_risk_pct × portfolio_capital) / (k × ATR_n)
```
where `ATR_n` is the n-day Average True Range (a volatility measure in price units), `k × ATR_n` is the initial stop distance from §5, and `target_risk_pct` is a fixed fraction of capital the strategy is willing to risk per trade (e.g. 1%). Because the denominator is the dollar loss per share if the stop is hit, a stop-out loses roughly `target_risk_pct` of capital (more if price gaps through the stop). Position value is capped at a fixed fraction of capital (100% by default, i.e. no leverage) so that a quiet stock with a tiny ATR can't produce an oversized position.

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
| Resulting position size | Larger | Smaller |

Position size is not a separate dial: because §4 sizes by `k × ATR_n`, a tighter `k` in `trending_low_vol` already produces a larger position and a wider `k` in `trending_high_vol` a smaller one, at the same risk per trade. Adding an extra size multiplier on top would double-count the regime. `k` is fixed at entry from the regime at that time and does not change if the regime flips mid-trade.

This isn't two strategies — it's one strategy with regime-aware dials, which is the practical distinction to be clear about in interviews: the *logic* doesn't change, only the *risk parameters*.

**Outcome:** with the trend gate removed (§2), this table now applies in every regime: `k` follows the volatility half of the regime only (`*_low_vol` → tighter, `*_high_vol` → wider). It was tested against a single fixed `k` set to the midpoint of each pair (2, 2.75, 4 instead of 1.5/2.5, 2/3.5, 3/5), so both have the same average stop width:

| Variant | Median OOS Sharpe | Mean OOS Sharpe | Median max drawdown |
|---|---|---|---|
| Regime-based `k` | 0.37 | 0.37 | −10.4% |
| Fixed `k` | 0.31 | 0.40 | −8.7% |

The fixed `k` was better on 11 of 18 tickers but by a median of only +0.01 Sharpe; it did better on the index ETFs and growth names (SPY +0.23, AAPL +0.24) and worse on the defensive stocks (WMT, JPM, IBM, JNJ). This is a tie: regime-based stop width neither clearly helps nor clearly hurts. It is kept as the default because it is the thesis as specified, but a fixed `k` is the simpler choice with no measurable loss.

---

## 7. Multi-Timeframe Confirmation

**Definition:** resample price data to weekly bars, run the same trend classification (ADX-based) on the weekly series, and require the weekly trend to also read `trending` before acting on a daily-level entry signal.

**Why this matters:** a daily-level breakout inside a weekly ranging market is far more likely to be noise than a genuine trend. This is a standard technique for reducing whipsaw — trading in the direction of the higher timeframe while timing entries on the lower one.

**Outcome: tested and rejected.** Implemented as specified (weekly bars built only from completed weeks, weekly ADX ≥ 25 with +DI > −DI). In the ablation study across 18 tickers (index ETFs, mega-caps and 2002-era large caps, 2006–2025 out of sample), adding the filter lowered Sharpe on 15 of 18 tickers (median −0.25): it vetoed too many entries without improving the ones it kept enough to compensate. It is disabled by default and removed from the walk-forward grid; the code remains so the result can be reproduced.

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

**What held up after testing.** The validation half of the chain (next-open fills, transaction costs, walk-forward selection, buy-and-hold benchmark, ablation across 18 tickers) is what the project demonstrates most strongly — it is what caught the components that did not work. Of the strategy components, the trend gate (§2) and the weekly filter (§7) were rejected. The ensemble, regime-based stop width, crash filter and trailing stop each measured close to zero contribution (median between −0.06 and 0.00 Sharpe across the 18 tickers at default parameters, each helping on 6 to 9 of them). Median training-window Sharpe of 0.84 fell to 0.33 out of sample, which is the overfitting gap the walk-forward exists to expose.

---

### References (for further reading / citation)
- Moskowitz, T., Ooi, Y.H., & Pedersen, L.H. (2012). *Time Series Momentum.* Journal of Financial Economics.
- Daniel, K., & Moskowitz, T. (2016). *Momentum Crashes.* Journal of Financial Economics.
