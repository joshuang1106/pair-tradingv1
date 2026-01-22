# =========================
# Pairs MR backtest with entry/exit dots & per-trade PnL
# =========================
# Requirements: pip install yfinance pandas numpy statsmodels matplotlib

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from statsmodels.api import OLS, add_constant
from statsmodels.tsa.stattools import coint

# ---------- User settings ----------
TICKERS    = ["^N225", "^GSPC"]     # <-- pick two tickers
START_DATE = "2020-01-01"
END_DATE   = "2025-10-15"                # None = up to today

ENTRY_Z = 2                      # entry threshold
EXIT_Z  = 0.5                      # exit threshold (do not set to 0; this improves win rate)
Z_WIN   = 60                       # z-score window
TX_COST_BPS = 1.0                  # per-leg bps (0.01%) applied on turnover

PLOT = True

# ---------- Utilities ----------
def fetch_prices(tickers, start, end=None):
    px = yf.download(tickers, start=start, end=end, auto_adjust=False, progress=False)["Adj Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame()
    return px.ffill().dropna()

def engle_granger_beta(p1, p2, use_logs=True):
    s1, s2 = p1.align(p2, join="inner")
    y = np.log(s1) if use_logs else s1
    x = np.log(s2) if use_logs else s2
    X = add_constant(x.values)
    beta = OLS(y.values, X).fit().params[1]
    stat, pval, _ = coint(y, x)
    return beta, pval

def make_spread(p1, p2, beta, use_logs=True):
    s1, s2 = p1.align(p2, join="inner")
    return (np.log(s1) - beta*np.log(s2)) if use_logs else (s1 - beta*s2)

def zscore(s, window=60):
    m = s.rolling(window).mean()
    sd = s.rolling(window).std()
    return (s - m) / sd

def generate_mr_signals(z, entry=2.0, exit=0.5):
    """
    +1 = long spread (long s1, short beta*s2)
    -1 = short spread
     0 = flat
    """
    sig = pd.Series(0.0, index=z.index)
    sig[z >  entry] = -1.0
    sig[z < -entry] = +1.0
    sig[np.abs(z) < exit] = 0.0
    # carry until exit
    sig = sig.replace(to_replace=0.0, method="ffill").fillna(0.0)
    return sig

# ---------- Core backtest with trade extraction ----------
def backtest_pair(prices, s1, s2, beta, z, mr_signal, tx_cost_bps=1.0):
    """
    Returns:
      bt_df:  DataFrame with ret_raw, ret_net, equity, pos
      trades_df: each trade's EntryDate, ExitDate, Side, PnL
    """
    # Align and returns
    p1 = prices[s1].reindex(z.index).ffill()
    p2 = prices[s2].reindex(z.index).ffill()
    r1 = p1.pct_change().fillna(0.0)
    r2 = p2.pct_change().fillna(0.0)

    # Positions (apply signal next day)
    pos = mr_signal.shift(1).fillna(0.0)
    pos1, pos2 = pos, -beta * pos

    # Scale-neutral raw returns
    gross = (np.abs(pos1) + np.abs(pos2)).replace(0.0, np.nan)
    ret_raw = (pos1 * r1 + pos2 * r2) / gross
    ret_raw = ret_raw.fillna(0.0)

    # Costs on turnover, normalized by gross so units match returns
    chg1 = np.abs(pos1 - pos1.shift(1)).fillna(np.abs(pos1))
    chg2 = np.abs(pos2 - pos2.shift(1)).fillna(np.abs(pos2))
    turnover = ((chg1 + chg2) / gross).fillna(0.0)
    tx_cost = (tx_cost_bps / 1e4) * turnover

    ret_net = ret_raw - tx_cost
    equity = (1 + ret_net).cumprod()

    bt_df = pd.DataFrame({"ret_raw": ret_raw, "ret_net": ret_net, "equity": equity, "pos": pos}, index=z.index)

    # ----- Extract trades (0->nonzero = entry; nonzero->0 = exit). PnL uses ret_net between entry..exit-1 -----
    trades = []
    in_trade = False
    entry_idx = None
    entry_side = None

    idx = bt_df.index

    for i in range(1, len(bt_df)):
        prev_pos = bt_df["pos"].iloc[i-1]
        curr_pos = bt_df["pos"].iloc[i]

        # open
        if not in_trade and prev_pos == 0 and curr_pos != 0:
            in_trade = True
            entry_idx = idx[i]
            entry_side = "Long" if curr_pos > 0 else "Short"

        # close (count returns up to previous bar i-1)
        elif in_trade and prev_pos != 0 and curr_pos == 0:
            exit_idx = idx[i]
            # returns from entry_idx .. idx[i-1]
            pnl = (1 + bt_df.loc[entry_idx:idx[i-1], "ret_net"]).prod() - 1.0
            trades.append({"EntryDate": entry_idx, "ExitDate": exit_idx, "Side": entry_side, "PnL": float(pnl)})
            in_trade = False
            entry_idx = None
            entry_side = None

    trades_df = pd.DataFrame(trades)
    return bt_df, trades_df

# ---------- Plotting ----------
def plot_pair_diagnostics(s1, s2, prices, spread, z, bt_df, trades_df, entry_z, exit_z):
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

    # 1) Combined normalized prices
    norm1 = prices[s1] / prices[s1].iloc[0]
    norm2 = prices[s2] / prices[s2].iloc[0]
    axs[0].plot(norm1.index, norm1, label=s1)
    axs[0].plot(norm2.index, norm2, label=s2)
    axs[0].set_title(f"Normalized Prices: {s1} & {s2}")
    axs[0].legend()

    # 2) Spread with entry/exit markers
    axs[1].plot(spread.index, spread, color="tab:blue", label="Spread")
    axs[1].axhline(spread.mean(), color="gray", linestyle="--", lw=1, label="Mean")
    # Markers
    if not trades_df.empty:
        # Long entries: green ^ ; Short entries: red v ; Exits: black x
        for _, tr in trades_df.iterrows():
            e, x, side = tr["EntryDate"], tr["ExitDate"], tr["Side"]
            if e in spread.index:
                if side == "Long":
                    axs[1].scatter(e, spread.loc[e], marker="^", s=90, color="green", edgecolors="black", linewidths=0.5, label="Entry (Long)")
                else:
                    axs[1].scatter(e, spread.loc[e], marker="v", s=90, color="red", edgecolors="black", linewidths=0.5, label="Entry (Short)")
            if x in spread.index:
                axs[1].scatter(x, spread.loc[x], marker="x", s=90, color="black", linewidths=2, label="Exit")
    # de-duplicate legend
    handles, labels = axs[1].get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    axs[1].legend(uniq.values(), uniq.keys())
    axs[1].set_title("Spread with Entry/Exit Markers")

    # 3) Z-score with thresholds and position overlay
    axs[2].plot(z.index, z, color="black", label="Z-score")
    axs[2].axhline(entry_z,  color="red", linestyle="--", lw=1, label=f"+Entry ({entry_z})")
    axs[2].axhline(-entry_z, color="red", linestyle="--", lw=1, label=f"-Entry (-{entry_z})")
    axs[2].axhline(exit_z,   color="gray", linestyle="--", lw=1, label=f"+Exit ({exit_z})")
    axs[2].axhline(-exit_z,  color="gray", linestyle="--", lw=1, label=f"-Exit (-{exit_z})")
    axs[2].set_title("Z-score & Thresholds")
    axs[2].legend(loc="upper right", ncol=2)

    # 4) Equity curve
    axs[3].plot(bt_df.index, bt_df["equity"], color="tab:purple")
    axs[3].set_title("Equity Curve (after costs)")
    axs[3].set_ylabel("Growth (×)")

    plt.tight_layout()
    plt.show()

    # Print per-trade PnL table and summary
    if not trades_df.empty:
        print("\nPer-trade PnL (after costs):")
        display_cols = ["EntryDate", "ExitDate", "Side", "PnL"]
        print(trades_df[display_cols].to_string(index=False))
        wins = (trades_df["PnL"] > 0).sum()
        losses = (trades_df["PnL"] <= 0).sum()
        print(f"\nTrades: {len(trades_df)} | Wins: {wins} | Losses: {losses} | Win rate: {wins/len(trades_df):.1%}")
        print(f"Total PnL: {trades_df['PnL'].sum():.2%} | Avg trade: {trades_df['PnL'].mean():.2%}")

# ---------- Driver ----------
def run_one_pair(tickers=TICKERS, start=START_DATE, end=END_DATE):
    assert len(tickers) == 2, "Please provide exactly two tickers."
    s1, s2 = tickers
    prices = fetch_prices(tickers, start, end)

    beta, pval = engle_granger_beta(prices[s1], prices[s2])
    print(f"Hedge beta (OLS on logs): {beta:.4f} | Engle–Granger p-value: {pval:.4f}")

    spread = make_spread(prices[s1], prices[s2], beta).dropna()
    z = zscore(spread, window=Z_WIN).dropna()
    mr_sig = generate_mr_signals(z, entry=ENTRY_Z, exit=EXIT_Z)

    bt_df, trades_df = backtest_pair(prices[[s1, s2]], s1, s2, beta, z, mr_sig, tx_cost_bps=TX_COST_BPS)

    if PLOT:
        plot_pair_diagnostics(s1, s2, prices, spread, z, bt_df, trades_df, ENTRY_Z, EXIT_Z)

    return bt_df, trades_df

# Run
if __name__ == "__main__":
    bt_df, trades_df = run_one_pair()

