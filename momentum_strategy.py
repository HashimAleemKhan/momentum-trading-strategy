"""
Cross-Sectional Momentum Strategy
==================================
A cross-sectional momentum strategy across 50+ US equities:

  - Ranks stocks each month by trailing 12-1 month returns (skip most recent
    month to avoid short-term reversal contamination)
  - Goes long the top quintile / short the bottom quintile, equal-weighted
  - Analyses signal decay (Spearman information coefficient by forward horizon)
  - Analyses turnover and the transaction-cost drag on returns
  - Breaks performance down by bull vs. bear market regime

Usage
-----
    python momentum_strategy.py

By default the script pulls real daily prices via yfinance. If that fails
(no internet, ticker issue, etc.) it automatically falls back to simulated
price data so the script still runs end-to-end.

Requirements
------------
    pip install numpy pandas matplotlib yfinance scipy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
USE_REAL_DATA = True          # tries yfinance first, falls back to simulation
START_DATE = "2015-01-01"
END_DATE = "2025-01-01"
LOOKBACK_MONTHS = 12          # momentum formation window
SKIP_MONTHS = 1               # skip most recent month (12-1 momentum)
HOLDING_MONTHS = 1            # rebalance frequency
QUANTILE = 0.2                # top/bottom 20% -> long/short
TRANSACTION_COST_BPS = 10     # one-way cost per unit turnover

TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AVGO", "ORCL", "CRM", "AMD", "INTC",
    "JPM", "BAC", "GS", "MS", "WFC", "BLK", "C", "AXP", "USB", "PNC",
    "JNJ", "UNH", "LLY", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY", "AMGN",
    "AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "TGT", "LOW", "COST", "WMT",
    "CAT", "BA", "HON", "UPS", "GE", "MMM", "LMT", "RTX", "DE", "EMR",
    "XOM", "CVX", "COP", "EOG", "SLB",
]


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------
def load_prices():
    """Return a DataFrame of monthly adjusted close prices (rows=dates, cols=tickers)."""
    if USE_REAL_DATA:
        try:
            import yfinance as yf
            print(f"Downloading data for {len(TICKERS)} tickers from Yahoo Finance...")
            raw = yf.download(
                TICKERS, start=START_DATE, end=END_DATE,
                auto_adjust=True, progress=False,
            )["Close"]
            raw = raw.dropna(axis=1, how="all")
            monthly = raw.resample("ME").last()
            monthly = monthly.dropna(axis=1, thresh=int(len(monthly) * 0.8))
            monthly = monthly.ffill().dropna(axis=0, how="any")
            if monthly.shape[1] < 10:
                raise ValueError("Too few tickers survived cleaning; falling back to simulation.")
            print(f"Loaded {monthly.shape[1]} tickers, {monthly.shape[0]} monthly observations.")
            return monthly
        except Exception as e:
            print(f"[warning] Real data fetch failed ({e}). Falling back to simulated data.")

    return simulate_prices()


def simulate_prices():
    """Simulate monthly prices with regime-dependent drift so bull/bear analysis is meaningful."""
    print("Simulating monthly price data...")
    rng = np.random.default_rng(42)
    dates = pd.date_range(START_DATE, END_DATE, freq="ME")
    n_months, n_assets = len(dates), len(TICKERS)

    # Piece together alternating bull/bear regimes over the sample
    regime_len = 24  # months per regime block
    monthly_drift = np.zeros(n_months)
    bull = True
    i = 0
    while i < n_months:
        end = min(i + regime_len, n_months)
        monthly_drift[i:end] = 0.012 if bull else -0.01
        bull = not bull
        i = end

    # Cross-sectional dispersion in skill/beta so momentum has something to rank on
    asset_alpha = rng.normal(0, 0.006, n_assets)
    asset_vol = rng.uniform(0.04, 0.09, n_assets)

    shocks = rng.normal(0, 1, size=(n_months, n_assets))
    returns = (monthly_drift[:, None] + asset_alpha[None, :]) + shocks * asset_vol[None, :]

    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    return pd.DataFrame(prices, index=dates, columns=TICKERS)


# ---------------------------------------------------------------------------
# 2. Signal construction
# ---------------------------------------------------------------------------
def compute_momentum_signal(prices, lookback=LOOKBACK_MONTHS, skip=SKIP_MONTHS):
    """Trailing (lookback)-(skip) month return, e.g. 12-1 month momentum."""
    return prices.shift(skip) / prices.shift(lookback) - 1


# ---------------------------------------------------------------------------
# 3. Portfolio construction
# ---------------------------------------------------------------------------
def build_portfolio_weights(signal, quantile=QUANTILE):
    """Equal-weighted long top quantile, short bottom quantile, dollar-neutral each month."""
    weights = pd.DataFrame(0.0, index=signal.index, columns=signal.columns)

    for date, row in signal.iterrows():
        valid = row.dropna()
        if len(valid) < 10:
            continue
        n_leg = max(1, int(len(valid) * quantile))
        ranked = valid.sort_values(ascending=False)
        longs = ranked.index[:n_leg]
        shorts = ranked.index[-n_leg:]
        weights.loc[date, longs] = 1.0 / n_leg
        weights.loc[date, shorts] = -1.0 / n_leg

    return weights


def compute_turnover(weights):
    """Monthly turnover = sum of absolute weight changes / 2."""
    delta = weights.diff().abs().sum(axis=1)
    return (delta / 2).fillna(0)


def backtest(prices, weights, cost_bps=TRANSACTION_COST_BPS):
    """Apply weights (formed at t, based on info up to t) to forward returns at t+1."""
    fwd_returns = prices.pct_change().shift(-1)
    gross = (weights * fwd_returns).sum(axis=1)

    turnover = compute_turnover(weights)
    cost = turnover * (cost_bps / 10000)
    net = gross - cost.reindex(gross.index).fillna(0)

    return gross.dropna(), net.dropna(), turnover


# ---------------------------------------------------------------------------
# 4. Signal decay: information coefficient by forward horizon
# ---------------------------------------------------------------------------
def signal_decay(prices, signal, horizons=(1, 3, 6, 9, 12)):
    """Spearman rank correlation between the signal at t and forward returns
    over each horizon, averaged across all months."""
    results = {}
    for h in horizons:
        fwd_return_h = prices.shift(-h) / prices.shift(-0) - 1  # from t to t+h
        ics = []
        common_dates = signal.index.intersection(fwd_return_h.index)
        for date in common_dates:
            s = signal.loc[date].dropna()
            r = fwd_return_h.loc[date].dropna()
            common = s.index.intersection(r.index)
            if len(common) < 10:
                continue
            ic, _ = spearmanr(s[common], r[common])
            if not np.isnan(ic):
                ics.append(ic)
        results[h] = np.mean(ics) if ics else np.nan
    return results


# ---------------------------------------------------------------------------
# 5. Regime analysis (bull vs bear, defined off a broad market proxy)
# ---------------------------------------------------------------------------
def regime_analysis(strategy_returns, prices):
    market_proxy = prices.mean(axis=1).pct_change()
    market_proxy = market_proxy.reindex(strategy_returns.index).dropna()
    common = strategy_returns.index.intersection(market_proxy.index)

    trailing_mkt = market_proxy.loc[common].rolling(6, min_periods=3).mean()
    bull_mask = trailing_mkt > 0
    bear_mask = trailing_mkt <= 0

    results = {}
    for label, mask in [("Bull regime", bull_mask), ("Bear regime", bear_mask)]:
        r = strategy_returns.loc[common][mask.reindex(common).fillna(False)]
        if len(r) == 0:
            continue
        ann_return = (1 + r).prod() ** (12 / len(r)) - 1
        ann_vol = r.std() * np.sqrt(12)
        sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan
        results[label] = {
            "Months": len(r),
            "Ann. Return": f"{ann_return:.2%}",
            "Ann. Vol": f"{ann_vol:.2%}",
            "Sharpe": f"{sharpe:.2f}",
        }
    return results


# ---------------------------------------------------------------------------
# 6. Reporting
# ---------------------------------------------------------------------------
def performance_summary(returns, label):
    ann_return = (1 + returns).prod() ** (12 / len(returns)) - 1
    ann_vol = returns.std() * np.sqrt(12)
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan
    cum = (1 + returns).cumprod()
    max_dd = (cum / cum.cummax() - 1).min()
    print(f"\n  {label}")
    print(f"  {'-' * 40}")
    print(f"    Ann. Return   : {ann_return:.2%}")
    print(f"    Ann. Vol      : {ann_vol:.2%}")
    print(f"    Sharpe Ratio  : {sharpe:.2f}")
    print(f"    Max Drawdown  : {max_dd:.2%}")
    return {"ann_return": ann_return, "ann_vol": ann_vol, "sharpe": sharpe, "max_dd": max_dd}


def plot_results(gross, net, decay_results, path="momentum_results.png"):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Cumulative performance
    (1 + gross).cumprod().plot(ax=axes[0, 0], label="Gross")
    (1 + net).cumprod().plot(ax=axes[0, 0], label="Net of costs")
    axes[0, 0].set_title("Cumulative Return")
    axes[0, 0].legend()
    axes[0, 0].set_ylabel("Growth of $1")

    # Drawdown
    cum = (1 + net).cumprod()
    dd = cum / cum.cummax() - 1
    dd.plot(ax=axes[0, 1], color="firebrick")
    axes[0, 1].set_title("Net Strategy Drawdown")
    axes[0, 1].set_ylabel("Drawdown")

    # Signal decay
    horizons = list(decay_results.keys())
    ics = [decay_results[h] for h in horizons]
    axes[1, 0].bar([str(h) for h in horizons], ics, color="steelblue")
    axes[1, 0].axhline(0, color="black", linewidth=0.8)
    axes[1, 0].set_title("Signal Decay: IC by Forward Horizon (months)")
    axes[1, 0].set_ylabel("Spearman IC")

    # Rolling 12m Sharpe
    roll_sharpe = (net.rolling(12).mean() / net.rolling(12).std()) * np.sqrt(12)
    roll_sharpe.plot(ax=axes[1, 1], color="darkgreen")
    axes[1, 1].axhline(0, color="black", linewidth=0.8)
    axes[1, 1].set_title("Rolling 12-Month Sharpe Ratio (net)")

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    print(f"\nSaved chart to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    prices = load_prices()
    signal = compute_momentum_signal(prices)
    weights = build_portfolio_weights(signal)
    gross, net, turnover = backtest(prices, weights)

    print("\n" + "=" * 55)
    print("  PERFORMANCE SUMMARY")
    print("=" * 55)
    performance_summary(gross, "Gross of transaction costs")
    performance_summary(net, "Net of transaction costs")

    avg_turnover = turnover.mean()
    print(f"\n    Avg monthly turnover : {avg_turnover:.1%}")
    print(f"    Est. annual tx cost  : {avg_turnover * 12 * (TRANSACTION_COST_BPS / 10000):.2%} "
          f"(@ {TRANSACTION_COST_BPS} bps one-way)")

    print("\n" + "=" * 55)
    print("  REGIME ANALYSIS")
    print("=" * 55)
    for regime, stats in regime_analysis(net, prices).items():
        print(f"\n  {regime}")
        print(f"  {'-' * 40}")
        for k, v in stats.items():
            print(f"    {k:<14}: {v}")

    print("\n" + "=" * 55)
    print("  SIGNAL DECAY (Spearman IC by Forward Horizon)")
    print("=" * 55)
    decay_results = signal_decay(prices, signal)
    for h, ic in decay_results.items():
        bar = "#" * int(abs(ic) * 100) if not np.isnan(ic) else ""
        print(f"    {h:>2}m horizon : IC = {ic:.4f}  {bar}")

    print("\nGenerating plots...")
    plot_results(gross, net, decay_results)
    print("\nDone.")


if __name__ == "__main__":
    main()
