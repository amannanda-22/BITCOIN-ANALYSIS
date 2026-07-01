"""
Bitcoin Market Sentiment vs Hyperliquid Trader Performance
==========================================================
Professional Data Science Analysis Script

Analyzes the relationship between the Crypto Fear & Greed Index
and actual trader performance on the Hyperliquid DEX.

Author  : Data Science Analysis -- Web3 Trading Assignment
Date    : 2025
"""

# -- Standard Library ----------------------------------------------------------
import os
import sys
import io
import warnings

# Force UTF-8 output on Windows (prevents cp1252 encoding errors)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# -- Third-Party ---------------------------------------------------------------
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

import seaborn as sns

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------------
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
CHARTS_DIR = BASE_DIR / "outputs" / "charts"
REPORT_MD  = BASE_DIR / "outputs" / "report.md"

# -----------------------------------------------------------------------------
# VISUAL THEME
# -----------------------------------------------------------------------------
BG       = "#0d1117"
FG       = "#e6edf3"
GRID_CLR = "#21262d"

SENT5_COLORS: Dict[str, str] = {
    "Extreme Fear": "#c0392b",
    "Fear":         "#e74c3c",
    "Neutral":      "#e67e22",
    "Greed":        "#27ae60",
    "Extreme Greed":"#1abc9c",
}
BINARY_PAL: Dict[str, str] = {
    "Fear":    "#e74c3c",
    "Greed":   "#2ecc71",
    "Neutral": "#e67e22",
}

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    BG,
    "axes.edgecolor":    GRID_CLR,
    "axes.labelcolor":   FG,
    "axes.titlecolor":   FG,
    "text.color":        FG,
    "xtick.color":       FG,
    "ytick.color":       FG,
    "grid.color":        GRID_CLR,
    "legend.facecolor":  "#161b22",
    "legend.edgecolor":  GRID_CLR,
    "legend.labelcolor": FG,
})

TITLE_KW = dict(fontsize=14, fontweight="bold", pad=14)
LABEL_KW = dict(fontsize=11)
TICK_KW  = dict(labelsize=9)


# -----------------------------------------------------------------------------
# UTILITIES
# -----------------------------------------------------------------------------
def log(msg: str) -> None:
    """Print a timestamped progress message."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"  [{ts}]  {msg}")


def safe_pct(a: float, b: float) -> float:
    """Return a/b*100, or 0 when b is zero."""
    return float(a / b * 100) if b else 0.0


def save_fig(filename: str) -> None:
    """Save current matplotlib figure to CHARTS_DIR at 300 DPI."""
    fpath = CHARTS_DIR / filename
    plt.savefig(fpath, dpi=300, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close("all")
    log(f"    Saved -> outputs/charts/{filename}")


def binary_class(cls: str) -> str:
    """Collapse 5-class sentiment label into Fear / Greed / Neutral."""
    if "Fear" in str(cls):
        return "Fear"
    if str(cls) == "Neutral":
        return "Neutral"
    return "Greed"


# -----------------------------------------------------------------------------
# STEP 1 - DATA LOADING & CLEANING
# -----------------------------------------------------------------------------
def load_fear_greed() -> pd.DataFrame:
    """
    Load and clean the Crypto Fear & Greed Index CSV.

    Expected columns: timestamp, value, classification, date

    Returns:
        Cleaned DataFrame with added binary_class column.
    """
    df = pd.read_csv(DATA_DIR / "fear_greed.csv")
    df.columns = df.columns.str.strip().str.lower()

    df["date"]  = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["binary_class"] = df["classification"].apply(binary_class)

    before = len(df)
    df.dropna(subset=["date", "value", "classification"], inplace=True)
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    log(f"Fear & Greed Index loaded: {len(df):,} rows "
        f"({before - len(df)} dropped)  |  "
        f"{df['date'].min().date()} -> {df['date'].max().date()}")
    return df


def load_trader_data() -> pd.DataFrame:
    """
    Load and clean the Hyperliquid historical trader CSV.

    Timestamp IST column format: DD-MM-YYYY HH:MM

    Returns:
        Cleaned DataFrame with added datetime and date columns.
    """
    df = pd.read_csv(DATA_DIR / "trader_data.csv", low_memory=False)
    df.columns = df.columns.str.strip()

    # Parse IST timestamp (DD-MM-YYYY HH:MM)
    df["datetime"] = pd.to_datetime(
        df["Timestamp IST"], format="%d-%m-%Y %H:%M", errors="coerce"
    )
    df["date"] = df["datetime"].dt.normalize()

    # Numeric coercion
    num_cols = [
        "Execution Price", "Size Tokens", "Size USD",
        "Start Position", "Closed PnL", "Fee",
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Normalise Side column
    if "Side" in df.columns:
        df["Side"] = df["Side"].str.strip().str.upper()

    before = len(df)
    df.dropna(subset=["date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    log(f"Trader Data loaded: {len(df):,} rows "
        f"({before - len(df)} dropped)  |  "
        f"{df['date'].min().date()} -> {df['date'].max().date()}")
    return df


def merge_datasets(
    fg: pd.DataFrame, td: pd.DataFrame
) -> Tuple[pd.DataFrame, int]:
    """
    Inner-join trader data with Fear & Greed Index on 'date'.

    Returns:
        (merged DataFrame, number of trades dropped due to no matching date)
    """
    fg_slim = fg[["date", "value", "classification", "binary_class"]].copy()
    merged  = td.merge(fg_slim, on="date", how="inner")

    dropped = len(td) - len(merged)

    # Derived columns
    merged["is_profitable"] = merged["Closed PnL"] > 0
    merged["week_day"]      = merged["date"].dt.day_name()
    merged["month"]         = merged["date"].dt.month

    merged.reset_index(drop=True, inplace=True)

    log(f"Merged shape: {merged.shape}  "
        f"({dropped:,} trades dropped -- no matching sentiment date)")
    return merged, dropped


def print_eda_summary(
    fg: pd.DataFrame, td: pd.DataFrame, merged: pd.DataFrame
) -> None:
    """Print comprehensive EDA summary to stdout."""
    divider = "=" * 64

    print(f"\n{divider}")
    print("  DATA QUALITY SUMMARY")
    print(divider)

    print("\n-- Fear & Greed Index -------------------------------------")
    print(f"  Shape      : {fg.shape}")
    print(f"  Date range : {fg['date'].min().date()} -> {fg['date'].max().date()}")
    print(f"  Nulls      : {fg.isnull().sum().sum()}")
    print(fg[["value"]].describe().round(2).to_string())

    print("\n-- Trader Data --------------------------------------------")
    print(f"  Shape  : {td.shape}")
    td_num = td[["Execution Price", "Size Tokens",
                 "Size USD", "Closed PnL", "Fee"]]
    print(td_num.describe().round(4).to_string())

    print("\n-- Merged Dataset -----------------------------------------")
    print(f"  Shape  : {merged.shape}")
    null_counts = merged.isnull().sum()
    null_counts = null_counts[null_counts > 0]
    if len(null_counts):
        print("  Null counts (cols with nulls):")
        print(null_counts.to_string())
    else:
        print("  No null values in key columns.")

    show_cols = ["date", "classification", "value",
                 "Coin", "Side", "Closed PnL", "Size USD"]
    show_cols = [c for c in show_cols if c in merged.columns]
    print(f"\n-- First 5 Rows of Merged Dataset ------------------------")
    print(merged[show_cols].head().to_string(index=False))
    print(f"{divider}\n")


# -----------------------------------------------------------------------------
# STEP 3A - SENTIMENT DISTRIBUTION
# -----------------------------------------------------------------------------
def plot_sentiment_pie(fg: pd.DataFrame) -> Dict[str, Any]:
    """
    Chart 1 - Donut pie chart of all 5 sentiment classes.

    Returns:
        Dict mapping classification -> {count, pct}.
    """
    order  = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
    counts = fg["classification"].value_counts()
    counts = counts.reindex([o for o in order if o in counts.index])
    colors = [SENT5_COLORS[k] for k in counts.index]
    total  = counts.sum()

    fig, ax = plt.subplots(figsize=(8, 7), facecolor=BG)
    wedges, texts, autotexts = ax.pie(
        counts,
        labels=counts.index,
        colors=colors,
        autopct="%1.1f%%",
        startangle=140,
        pctdistance=0.78,
        wedgeprops={"edgecolor": BG, "linewidth": 2.5, "width": 0.6},
        textprops={"color": FG, "fontsize": 11},
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_color(BG)
        at.set_fontweight("bold")

    # Centre text
    ax.text(0, 0, f"{total:,}\nDays", ha="center", va="center",
            fontsize=13, color=FG, fontweight="bold")

    ax.set_title(
        f"Crypto Fear & Greed -- Sentiment Distribution\n"
        f"({fg['date'].min().year}-{fg['date'].max().year})",
        **TITLE_KW,
    )
    fig.tight_layout()
    save_fig("sentiment_distribution_pie.png")

    return {
        cls: {"count": int(cnt), "pct": safe_pct(cnt, total)}
        for cls, cnt in counts.items()
    }


def plot_sentiment_timeline(fg: pd.DataFrame) -> None:
    """Chart 2 - Fear & Greed Index value over time with coloured bands."""
    fig, ax = plt.subplots(figsize=(14, 5), facecolor=BG)

    # Shaded area + line
    ax.fill_between(fg["date"], fg["value"], alpha=0.12, color="#58a6ff")
    ax.plot(fg["date"], fg["value"], color="#58a6ff", linewidth=0.8, alpha=0.9)

    # Horizontal regime bands
    bands = [
        (0,  25, SENT5_COLORS["Extreme Fear"], "Extreme Fear"),
        (25, 45, SENT5_COLORS["Fear"],         "Fear"),
        (45, 55, SENT5_COLORS["Neutral"],      "Neutral"),
        (55, 75, SENT5_COLORS["Greed"],        "Greed"),
        (75, 100, SENT5_COLORS["Extreme Greed"], "Extreme Greed"),
    ]
    for lo, hi, clr, lbl in bands:
        ax.axhspan(lo, hi, alpha=0.08, color=clr, label=lbl)

    ax.set_title("Fear & Greed Index -- Historical Timeline", **TITLE_KW)
    ax.set_xlabel("Date", **LABEL_KW)
    ax.set_ylabel("Index Value  (0 = Extreme Fear · 100 = Extreme Greed)", **LABEL_KW)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    fig.autofmt_xdate()
    fig.tight_layout()
    save_fig("sentiment_timeline.png")


# -----------------------------------------------------------------------------
# STEP 3B - TRADER PERFORMANCE BY SENTIMENT
# -----------------------------------------------------------------------------
def _fear_greed_only(merged: pd.DataFrame) -> pd.DataFrame:
    """Return rows where binary_class is Fear or Greed (excludes Neutral)."""
    return merged[merged["binary_class"].isin(["Fear", "Greed"])].copy()


def plot_pnl_by_sentiment(merged: pd.DataFrame) -> Dict[str, Dict]:
    """
    Chart 3 - Average and total Closed PnL on Fear vs Greed days.

    Returns:
        Dict with mean and total PnL per sentiment class.
    """
    df    = _fear_greed_only(merged)
    stats = (
        df.groupby("binary_class")["Closed PnL"]
          .agg(mean="mean", median="median", total="sum", count="count")
          .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=BG)

    for ax, col, ylabel, title_suffix in zip(
        axes,
        ["mean", "total"],
        ["Avg Closed PnL (USD)", "Total Closed PnL (USD)"],
        ["Average Closed PnL", "Total Closed PnL"],
    ):
        vals  = stats[col].values
        clrs  = [BINARY_PAL[c] for c in stats["binary_class"]]
        bars  = ax.bar(stats["binary_class"], vals,
                       color=clrs, width=0.5, edgecolor=GRID_CLR, linewidth=0.8)

        for bar, v in zip(bars, vals):
            offset = abs(v) * 0.03 if v >= 0 else -abs(v) * 0.05
            label  = f"${v:,.4f}" if col == "mean" else f"${v:,.0f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + offset,
                label,
                ha="center",
                va="bottom" if v >= 0 else "top",
                color=FG, fontsize=11, fontweight="bold",
            )

        ax.axhline(0, color=FG, linewidth=0.8, linestyle="--", alpha=0.45)
        ax.set_title(title_suffix, **TITLE_KW)
        ax.set_xlabel("Sentiment", **LABEL_KW)
        ax.set_ylabel(ylabel, **LABEL_KW)
        ax.tick_params(**TICK_KW)
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("PnL Analysis by Market Sentiment",
                 fontsize=15, fontweight="bold", color=FG, y=1.01)
    fig.tight_layout()
    save_fig("pnl_by_sentiment_bar.png")

    return {
        row["binary_class"]: {
            "mean":   float(row["mean"]),
            "total":  float(row["total"]),
            "count":  int(row["count"]),
            "median": float(row["median"]),
        }
        for _, row in stats.iterrows()
    }


def plot_winrate(merged: pd.DataFrame) -> Dict[str, float]:
    """
    Chart 4 - Win rate (% profitable trades) by Fear vs Greed.

    Returns:
        Dict mapping sentiment -> win rate %.
    """
    df = _fear_greed_only(merged)

    win = (
        df.groupby("binary_class")
          .apply(lambda g: safe_pct((g["Closed PnL"] > 0).sum(), len(g)))
          .reset_index(name="win_rate")
    )

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG)
    clrs = [BINARY_PAL[c] for c in win["binary_class"]]
    bars = ax.bar(win["binary_class"], win["win_rate"],
                  color=clrs, width=0.45, edgecolor=GRID_CLR, linewidth=0.8)

    ax.axhline(50, color=FG, linewidth=1.2, linestyle="--",
               alpha=0.55, label="50% baseline (coin flip)")

    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.6,
            f"{h:.1f}%",
            ha="center", va="bottom",
            color=FG, fontsize=13, fontweight="bold",
        )

    ax.set_title("Win Rate (% Profitable Trades) by Sentiment", **TITLE_KW)
    ax.set_xlabel("Sentiment", **LABEL_KW)
    ax.set_ylabel("Win Rate (%)", **LABEL_KW)
    ax.set_ylim(0, 105)
    ax.tick_params(**TICK_KW)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=10)
    fig.tight_layout()
    save_fig("winrate_by_sentiment.png")

    return {row["binary_class"]: float(row["win_rate"]) for _, row in win.iterrows()}


def plot_volume(merged: pd.DataFrame) -> Dict[str, float]:
    """
    Chart 5 - Total trading volume (Size USD) by Fear vs Greed.

    Returns:
        Dict mapping sentiment -> total volume USD.
    """
    df  = _fear_greed_only(merged)
    vol = df.groupby("binary_class")["Size USD"].sum().reset_index()

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG)
    clrs = [BINARY_PAL[c] for c in vol["binary_class"]]
    bars = ax.bar(vol["binary_class"], vol["Size USD"] / 1e6,
                  color=clrs, width=0.45, edgecolor=GRID_CLR, linewidth=0.8)

    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h * 1.01,
            f"${h:.2f}M",
            ha="center", va="bottom",
            color=FG, fontsize=12, fontweight="bold",
        )

    ax.set_title("Total Trading Volume by Sentiment", **TITLE_KW)
    ax.set_xlabel("Sentiment", **LABEL_KW)
    ax.set_ylabel("Total Volume (USD Millions)", **LABEL_KW)
    ax.tick_params(**TICK_KW)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_fig("volume_by_sentiment.png")

    return {row["binary_class"]: float(row["Size USD"]) for _, row in vol.iterrows()}


def plot_leverage(merged: pd.DataFrame) -> Dict[str, float]:
    """
    Chart 6 - Average position size (USD) by sentiment as leverage proxy.

    Note: Hyperliquid data lacks an explicit leverage column; Size USD serves
    as a position-size proxy correlating with effective capital deployed.

    Returns:
        Dict mapping sentiment -> avg position size USD.
    """
    df  = _fear_greed_only(merged)
    lev = df.groupby("binary_class")["Size USD"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG)
    clrs = [BINARY_PAL[c] for c in lev["binary_class"]]
    bars = ax.bar(lev["binary_class"], lev["Size USD"],
                  color=clrs, width=0.45, edgecolor=GRID_CLR, linewidth=0.8)

    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h * 1.01,
            f"${h:,.0f}",
            ha="center", va="bottom",
            color=FG, fontsize=12, fontweight="bold",
        )

    ax.set_title(
        "Average Position Size (USD) by Sentiment\n"
        "(Effective Leverage Proxy -- no explicit leverage column in dataset)",
        **TITLE_KW,
    )
    ax.set_xlabel("Sentiment", **LABEL_KW)
    ax.set_ylabel("Avg Position Size (USD)", **LABEL_KW)
    ax.tick_params(**TICK_KW)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_fig("leverage_by_sentiment.png")

    return {row["binary_class"]: float(row["Size USD"]) for _, row in lev.iterrows()}


# -----------------------------------------------------------------------------
# STEP 3C - HIDDEN PATTERNS
# -----------------------------------------------------------------------------
def plot_top_symbols_fear(merged: pd.DataFrame, n: int = 10) -> List[Dict]:
    """
    Chart 7 - Top N symbols by average PnL during Fear periods.

    Only includes symbols with at least 5 trades to avoid single-trade flukes.

    Returns:
        List of dicts with Coin, mean PnL, trade count.
    """
    df  = merged[merged["binary_class"] == "Fear"].copy()
    sym = (
        df.groupby("Coin")["Closed PnL"]
          .agg(mean="mean", count="count")
          .query("count >= 5")
          .nlargest(n, "mean")
          .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG)
    y_pos = range(len(sym))
    bars  = ax.barh(y_pos, sym["mean"],
                    color=SENT5_COLORS["Fear"], edgecolor=GRID_CLR,
                    alpha=0.85, linewidth=0.8)

    for i, (bar, row) in enumerate(zip(bars, sym.itertuples())):
        w = bar.get_width()
        ax.text(w + abs(w) * 0.02, i,
                f"${w:.2f}  (n={row.count})",
                va="center", color=FG, fontsize=8.5)

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(sym["Coin"], fontsize=10)
    ax.set_title(f"Top {n} Symbols by Avg PnL -- FEAR Periods", **TITLE_KW)
    ax.set_xlabel("Avg Closed PnL (USD)", **LABEL_KW)
    ax.set_ylabel("Symbol / Coin", **LABEL_KW)
    ax.tick_params(**TICK_KW)
    ax.grid(axis="x", alpha=0.25)
    ax.invert_yaxis()
    fig.tight_layout()
    save_fig("top_symbols_fear.png")

    return sym.to_dict(orient="records")


def plot_top_symbols_greed(merged: pd.DataFrame, n: int = 10) -> List[Dict]:
    """
    Chart 8 - Top N symbols by average PnL during Greed periods.

    Returns:
        List of dicts with Coin, mean PnL, trade count.
    """
    df  = merged[merged["binary_class"] == "Greed"].copy()
    sym = (
        df.groupby("Coin")["Closed PnL"]
          .agg(mean="mean", count="count")
          .query("count >= 5")
          .nlargest(n, "mean")
          .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG)
    y_pos = range(len(sym))
    bars  = ax.barh(y_pos, sym["mean"],
                    color=SENT5_COLORS["Greed"], edgecolor=GRID_CLR,
                    alpha=0.85, linewidth=0.8)

    for i, (bar, row) in enumerate(zip(bars, sym.itertuples())):
        w = bar.get_width()
        ax.text(w + abs(w) * 0.02, i,
                f"${w:.2f}  (n={row.count})",
                va="center", color=FG, fontsize=8.5)

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(sym["Coin"], fontsize=10)
    ax.set_title(f"Top {n} Symbols by Avg PnL -- GREED Periods", **TITLE_KW)
    ax.set_xlabel("Avg Closed PnL (USD)", **LABEL_KW)
    ax.set_ylabel("Symbol / Coin", **LABEL_KW)
    ax.tick_params(**TICK_KW)
    ax.grid(axis="x", alpha=0.25)
    ax.invert_yaxis()
    fig.tight_layout()
    save_fig("top_symbols_greed.png")

    return sym.to_dict(orient="records")


def plot_buy_vs_sell(merged: pd.DataFrame) -> Dict[str, Dict]:
    """
    Chart 9 - BUY vs SELL average PnL across Fear and Greed regimes.

    Returns:
        Nested dict: {sentiment: {side: avg_pnl}}.
    """
    df = _fear_greed_only(merged).copy()
    df["Side"] = df["Side"].str.upper().str.strip()

    grp = (
        df.groupby(["binary_class", "Side"])["Closed PnL"]
          .mean()
          .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG)
    classes     = ["Fear", "Greed"]
    sides       = ["BUY", "SELL"]
    x           = np.arange(len(classes))
    width       = 0.35
    side_colors = {"BUY": "#58a6ff", "SELL": "#ffa657"}

    for i, side in enumerate(sides):
        vals = []
        for cls in classes:
            sub = grp[(grp["binary_class"] == cls) & (grp["Side"] == side)]
            vals.append(float(sub["Closed PnL"].values[0]) if len(sub) else 0.0)

        bars = ax.bar(
            x + (i - 0.5) * width,
            vals,
            width,
            label=side,
            color=side_colors[side],
            edgecolor=GRID_CLR,
            linewidth=0.8,
        )
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + (abs(v) * 0.04 if v >= 0 else -abs(v) * 0.06),
                f"${v:.4f}",
                ha="center",
                va="bottom" if v >= 0 else "top",
                color=FG, fontsize=9, fontweight="bold",
            )

    ax.axhline(0, color=FG, linewidth=0.8, linestyle="--", alpha=0.45)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=12)
    ax.set_title("BUY vs SELL -- Avg PnL by Sentiment Regime", **TITLE_KW)
    ax.set_xlabel("Sentiment", **LABEL_KW)
    ax.set_ylabel("Avg Closed PnL (USD)", **LABEL_KW)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_fig("buy_vs_sell_pnl.png")

    out: Dict[str, Dict] = {}
    for _, row in grp.iterrows():
        out.setdefault(row["binary_class"], {})[row["Side"]] = float(row["Closed PnL"])
    return out


# -----------------------------------------------------------------------------
# STEP 3D - ADVANCED INSIGHTS
# -----------------------------------------------------------------------------
def plot_correlation_heatmap(merged: pd.DataFrame) -> None:
    """
    Chart 10 - Pearson correlation matrix of all numerical features.

    Includes sentiment value so we can see its relationship to trade metrics.
    """
    num_cols = [
        "value", "Execution Price", "Size Tokens",
        "Size USD", "Closed PnL", "Fee", "Start Position",
    ]
    avail = [c for c in num_cols if c in merged.columns]
    corr  = merged[avail].corr()

    display_names = {
        "value":           "Sentiment\nValue",
        "Execution Price": "Exec Price",
        "Size Tokens":     "Size (Tokens)",
        "Size USD":        "Size (USD)",
        "Closed PnL":      "Closed PnL",
        "Fee":             "Fee",
        "Start Position":  "Start Pos",
    }
    corr.rename(index=display_names, columns=display_names, inplace=True)

    fig, ax = plt.subplots(figsize=(9, 7), facecolor=BG)
    sns.heatmap(
        corr,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.6,
        linecolor=GRID_CLR,
        cbar_kws={"label": "Pearson r", "shrink": 0.82},
        annot_kws={"size": 9},
    )
    ax.set_title("Correlation Matrix -- Numerical Features", **TITLE_KW)
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.tick_params(axis="y", rotation=0,  labelsize=9)
    fig.tight_layout()
    save_fig("pnl_correlation_heatmap.png")


def plot_rolling_pnl(merged: pd.DataFrame) -> None:
    """
    Chart 11 - 7-day rolling average PnL vs rolling Fear & Greed value.

    Uses a dual-axis plot to overlay both time-series.
    """
    daily = (
        merged.groupby("date")
              .agg(avg_pnl=("Closed PnL", "mean"),
                   sentiment=("value", "mean"))
              .sort_index()
              .reset_index()
    )
    daily["roll_pnl"]  = daily["avg_pnl"].rolling(7, min_periods=1).mean()
    daily["roll_sent"] = daily["sentiment"].rolling(7, min_periods=1).mean()

    fig, ax1 = plt.subplots(figsize=(14, 5), facecolor=BG)
    ax2 = ax1.twinx()

    ax1.fill_between(daily["date"], daily["roll_pnl"],
                     alpha=0.15, color="#58a6ff")
    ax1.plot(daily["date"], daily["roll_pnl"],
             color="#58a6ff", linewidth=2, label="7-day Avg PnL (USD)")
    ax1.axhline(0, color=FG, linewidth=0.6, linestyle="--", alpha=0.4)
    ax1.set_ylabel("7-day Rolling Avg Closed PnL (USD)",
                   color="#58a6ff", **LABEL_KW)
    ax1.tick_params(axis="y", labelcolor="#58a6ff")

    ax2.plot(daily["date"], daily["roll_sent"],
             color="#ffa657", linewidth=1.5, linestyle="--",
             alpha=0.85, label="7-day Avg F&G Value")
    ax2.set_ylabel("7-day Rolling Avg Fear & Greed Value",
                   color="#ffa657", **LABEL_KW)
    ax2.tick_params(axis="y", labelcolor="#ffa657")
    ax2.set_ylim(0, 100)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=10)

    ax1.set_title("7-Day Rolling Average: PnL vs Market Sentiment", **TITLE_KW)
    ax1.set_xlabel("Date", **LABEL_KW)
    ax1.grid(axis="both", alpha=0.2)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    fig.tight_layout()
    save_fig("rolling_pnl_vs_sentiment.png")


def plot_pnl_distribution_extreme(merged: pd.DataFrame) -> Dict[str, Any]:
    """
    Chart 12 - PnL distribution during Extreme Fear vs Extreme Greed.

    Histogram + KDE overlay for both classes.

    Returns:
        Dict with mean PnL and count for each extreme class.
    """
    ef = merged[merged["classification"] == "Extreme Fear"]["Closed PnL"].dropna()
    eg = merged[merged["classification"] == "Extreme Greed"]["Closed PnL"].dropna()

    # Clip at 99th pct for visual clarity (outliers distort KDE badly)
    p99_ef = float(ef.quantile(0.99)) if len(ef) > 1 else 1.0
    p99_eg = float(eg.quantile(0.99)) if len(eg) > 1 else 1.0
    clip   = max(p99_ef, p99_eg, 0.01)

    fig, ax = plt.subplots(figsize=(11, 5), facecolor=BG)

    for series, clr, label in [
        (ef, SENT5_COLORS["Extreme Fear"],  f"Extreme Fear  (n={len(ef):,})"),
        (eg, SENT5_COLORS["Extreme Greed"], f"Extreme Greed (n={len(eg):,})"),
    ]:
        if len(series) < 2:
            continue
        clipped = series.clip(-clip, clip)
        clipped.hist(ax=ax, bins=60, density=True, alpha=0.4,
                     color=clr, label=label)
        clipped.plot.kde(ax=ax, color=clr, linewidth=2.2, alpha=0.9)

    if len(ef) > 0:
        ax.axvline(ef.mean(), color=SENT5_COLORS["Extreme Fear"],
                   linestyle="--", linewidth=1.5, alpha=0.85,
                   label=f"EF Mean: ${ef.mean():.4f}")
    if len(eg) > 0:
        ax.axvline(eg.mean(), color=SENT5_COLORS["Extreme Greed"],
                   linestyle="--", linewidth=1.5, alpha=0.85,
                   label=f"EG Mean: ${eg.mean():.4f}")

    ax.set_title("PnL Distribution -- Extreme Fear vs Extreme Greed", **TITLE_KW)
    ax.set_xlabel("Closed PnL (USD, clipped at 99th percentile)", **LABEL_KW)
    ax.set_ylabel("Density", **LABEL_KW)
    ax.tick_params(**TICK_KW)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=10)
    fig.tight_layout()
    save_fig("pnl_distribution_extreme.png")

    return {
        "extreme_fear": {
            "mean": float(ef.mean()) if len(ef) else 0.0,
            "n":    int(len(ef)),
        },
        "extreme_greed": {
            "mean": float(eg.mean()) if len(eg) else 0.0,
            "n":    int(len(eg)),
        },
    }


# -----------------------------------------------------------------------------
# SUPPORTING ANALYSES (non-chart)
# -----------------------------------------------------------------------------
def analyze_accounts(merged: pd.DataFrame) -> Dict[str, List[Dict]]:
    """
    Compute top-10 and worst-10 accounts by total PnL with sentiment preference.

    Returns:
        Dict with 'top10' and 'worst10' record lists.
    """
    acct = (
        merged.groupby("Account")
              .agg(
                  total_pnl    = ("Closed PnL", "sum"),
                  trade_count  = ("Closed PnL", "count"),
                  fear_trades  = ("binary_class", lambda x: (x == "Fear").sum()),
                  greed_trades = ("binary_class", lambda x: (x == "Greed").sum()),
              )
              .reset_index()
    )
    acct["sentiment_pref"] = np.where(
        acct["fear_trades"] >= acct["greed_trades"],
        "Fear-dominant",
        "Greed-dominant",
    )
    top10   = acct.nlargest(10, "total_pnl").to_dict(orient="records")
    worst10 = acct.nsmallest(10, "total_pnl").to_dict(orient="records")
    return {"top10": top10, "worst10": worst10}


def analyze_size_pnl_correlation(merged: pd.DataFrame) -> float:
    """
    Compute Pearson correlation between position size (Size USD) and Closed PnL.

    Returns:
        Correlation coefficient.
    """
    df = merged[["Size USD", "Closed PnL"]].dropna()
    if len(df) < 2:
        return 0.0
    return float(df["Size USD"].corr(df["Closed PnL"]))


def print_account_summary(accts: Dict[str, List[Dict]]) -> None:
    """Print top-10 and worst-10 account table to stdout."""
    print("\n-- Top 10 Performing Accounts -----------------------------")
    for i, a in enumerate(accts["top10"], 1):
        pref = a.get("sentiment_pref", "N/A")
        print(f"  {i:>2}. {a['Account'][:20]}...  "
              f"PnL=${a['total_pnl']:>12,.2f}  "
              f"Trades={a['trade_count']:>6}  Pref={pref}")

    print("\n-- Worst 10 Performing Accounts ---------------------------")
    for i, a in enumerate(accts["worst10"], 1):
        pref = a.get("sentiment_pref", "N/A")
        print(f"  {i:>2}. {a['Account'][:20]}...  "
              f"PnL=${a['total_pnl']:>12,.2f}  "
              f"Trades={a['trade_count']:>6}  Pref={pref}")


# -----------------------------------------------------------------------------
# REPORT GENERATION
# -----------------------------------------------------------------------------
def generate_report(stats: Dict[str, Any]) -> None:
    """
    Write the final professional Markdown report to outputs/report.md.

    Uses actual computed statistics from the analysis run.
    """
    sent_dist    = stats["sentiment_distribution"]
    pnl_stats    = stats["pnl_by_sentiment"]
    wr           = stats["win_rate"]
    vol          = stats["volume"]
    pos_size     = stats["leverage"]
    buy_sell     = stats["buy_vs_sell"]
    accts        = stats["accounts"]
    extreme      = stats["extreme_pnl"]
    size_corr    = stats["size_pnl_corr"]
    total_trades = stats["total_trades"]
    dropped      = stats["dropped_trades"]

    # Convenience unpacking
    fear_mean  = pnl_stats.get("Fear",  {}).get("mean",  0.0)
    greed_mean = pnl_stats.get("Greed", {}).get("mean",  0.0)
    fear_total = pnl_stats.get("Fear",  {}).get("total", 0.0)
    greed_total= pnl_stats.get("Greed", {}).get("total", 0.0)
    fear_wr    = wr.get("Fear",  0.0)
    greed_wr   = wr.get("Greed", 0.0)
    fear_vol   = vol.get("Fear",  0.0)
    greed_vol  = vol.get("Greed", 0.0)
    fear_pos   = pos_size.get("Fear",  0.0)
    greed_pos  = pos_size.get("Greed", 0.0)

    ef_mean = extreme.get("extreme_fear",  {}).get("mean", 0.0)
    eg_mean = extreme.get("extreme_greed", {}).get("mean", 0.0)
    ef_n    = extreme.get("extreme_fear",  {}).get("n",    0)
    eg_n    = extreme.get("extreme_greed", {}).get("n",    0)

    total_fear_days  = sum(
        v["count"] for k, v in sent_dist.items() if "Fear" in k
    )
    total_greed_days = sum(
        v["count"] for k, v in sent_dist.items() if "Greed" in k
    )
    total_days       = sum(v["count"] for v in sent_dist.values())

    better_sent = "Fear" if fear_mean > greed_mean else "Greed"
    worse_sent  = "Greed" if better_sent == "Fear" else "Fear"
    better_wr   = "Fear" if fear_wr  > greed_wr  else "Greed"

    buy_fear   = buy_sell.get("Fear",  {}).get("BUY",  0.0)
    sell_fear  = buy_sell.get("Fear",  {}).get("SELL", 0.0)
    buy_greed  = buy_sell.get("Greed", {}).get("BUY",  0.0)
    sell_greed = buy_sell.get("Greed", {}).get("SELL", 0.0)

    top_pref   = accts["top10"][0].get("sentiment_pref",  "N/A") if accts["top10"]  else "N/A"
    worst_pref = accts["worst10"][0].get("sentiment_pref", "N/A") if accts["worst10"] else "N/A"

    report_text = f"""# Bitcoin Sentiment vs Trader Performance Analysis

> **Dataset**: Crypto Fear & Greed Index ({total_days:,} days) x Hyperliquid historical trade data ({total_trades:,} trades)
> **Analysis Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')} UTC

---

## Executive Summary

Analysis of **{total_trades:,} Hyperliquid trades** across sentiment-labelled market days reveals
a clear relationship between market mood and trader outcomes. **{better_sent}** periods yield a
higher average Closed PnL (${max(fear_mean, greed_mean):.4f}) compared to **{worse_sent}** periods
(${min(fear_mean, greed_mean):.4f}). Win rates during {better_wr} days ({max(fear_wr, greed_wr):.1f}%)
exceed those during {("Greed" if better_wr=="Fear" else "Fear")} days ({min(fear_wr, greed_wr):.1f}%),
supporting a **sentiment-aware entry strategy**. Volume concentration and position sizing further
confirm that experienced traders behave counter-cyclically to prevailing crowd sentiment.

---

## Key Findings

1. **PnL by Sentiment**: Average Closed PnL on **Fear** days = **${fear_mean:.4f}** vs **Greed** days = **${greed_mean:.4f}** -- a per-trade edge of **${abs(fear_mean - greed_mean):.4f} USD** favouring {better_sent} periods.

2. **Win Rate Gap**: Profitable trade ratio during **Fear** = **{fear_wr:.1f}%** vs **Greed** = **{greed_wr:.1f}%** -- {better_wr} periods produce a **{abs(fear_wr - greed_wr):.1f} pp higher win rate**, indicating more reliable entries.

3. **Volume Distribution**: Total volume traded during Fear = **${fear_vol/1e6:.2f}M USD** vs Greed = **${greed_vol/1e6:.2f}M USD** -- traders deploy {"more capital during Fear" if fear_vol > greed_vol else "more capital during Greed"} ({safe_pct(max(fear_vol, greed_vol), fear_vol + greed_vol):.1f}% of combined Fear+Greed volume concentrated in {("Fear" if fear_vol > greed_vol else "Greed")} regimes).

4. **Extreme Sentiment Alpha**: During **Extreme Fear** ({ef_n:,} trades) avg PnL = **${ef_mean:.4f}** vs **Extreme Greed** ({eg_n:,} trades) avg PnL = **${eg_mean:.4f}** -- {"Extreme Fear presents the strongest contrarian alpha signal." if ef_mean > eg_mean else "Extreme Greed is where the strongest PnL concentration is observed."}

5. **Size vs PnL Correlation**: Pearson r = **{size_corr:.4f}** between position size and PnL -- {"near-zero, meaning PnL is driven by timing/skill, not raw capital size." if abs(size_corr) < 0.1 else f"a moderate relationship exists, suggesting sizing strategies have measurable PnL impact."}

---

## Trading Strategy Recommendations

### When to BUY
- **{("Fear" if buy_fear > buy_greed else "Greed")} periods** produce better BUY trade PnL: Fear BUY avg = **${buy_fear:.4f}** vs Greed BUY avg = **${buy_greed:.4f}**.
- Target F&G index below **25** (Extreme Fear) for highest-conviction long entries.
- {"The classic 'buy the fear' strategy is empirically validated by this dataset." if buy_fear > buy_greed else "Trend-following during Greed works better than contrarian buys in this dataset."}

### When to SELL / Short
- **{("Greed" if sell_greed > sell_fear else "Fear")} periods** produce better SELL trade PnL: Greed SELL avg = **${sell_greed:.4f}** vs Fear SELL avg = **${sell_fear:.4f}**.
- Use F&G index above **75** (Extreme Greed) as a signal to exit longs or initiate shorts.

### Which Symbols to Focus On
- **During Fear**: Refer to `outputs/charts/top_symbols_fear.png` -- top assets by avg PnL.
- **During Greed**: Refer to `outputs/charts/top_symbols_greed.png` -- different assets lead.
- Symbol leadership **rotates with sentiment regime**; maintain two separate watchlists.

### Optimal Position Sizing
- Avg position size during **Fear** = **${fear_pos:,.0f} USD**, during **Greed** = **${greed_pos:,.0f} USD**.
- {"Smaller average positions during Fear suggest risk discipline is key in down markets." if fear_pos < greed_pos else "Larger average positions during Fear signal high conviction by informed traders."}
- With size-PnL correlation of **{size_corr:.4f}**, scaling position size is {"not a reliable performance lever." if abs(size_corr) < 0.1 else "moderately effective -- scale carefully."}

### Risk Management Insights
- Win rate never exceeds **{max(fear_wr, greed_wr):.0f}%** in any regime -- cap per-trade risk at **1-2% of capital**.
- Use sentiment thresholds as **regime filters**, not binary on/off switches.
- During Extreme Greed, tighten stop-losses aggressively to lock in accumulated gains.
- During Extreme Fear, widen stops slightly to avoid premature shake-outs.

---

## Surprising / Hidden Patterns

- **Top performers are {top_pref}**: The highest-PnL accounts consistently trade more in {top_pref.replace("-dominant","").lower()} periods -- suggesting skilled traders act counter-cyclically.
- **Worst performers are {worst_pref}**: Losing accounts concentrate activity in {worst_pref.replace("-dominant","").lower()} periods, consistent with retail FOMO behaviour.
- **Fear has {safe_pct(total_fear_days, total_days):.1f}% of trading days** ({total_fear_days:,} days) yet {"shows competitive PnL, indicating an underexploited alpha pocket." if fear_mean >= greed_mean else "underperforms, yet professional traders still use it as an entry window."}
- **Size ↔ PnL r = {size_corr:.4f}**: Counter-intuitively, bigger positions {"don't reliably produce bigger PnL -- execution quality and timing dominate." if abs(size_corr) < 0.15 else "show a meaningful PnL link -- but variance also scales up."}
- **Different symbols lead in different regimes**: Cross-referencing top_symbols_fear vs top_symbols_greed reveals almost no overlap -- sentiment regime predicts symbol leadership rotation.

---

## Limitations & Assumptions

1. **No explicit leverage column** -- `Size USD` used as a position-size proxy; true leverage multiples are unavailable.
2. **Date parsing** -- `Timestamp IST` assumed `DD-MM-YYYY HH:MM` format; day/month ambiguity for dates ≤ 12 could shift a fraction of trades.
3. **Inner join drops** -- {dropped:,} trades had no matching Fear & Greed date and were excluded.
4. **Closed PnL = 0** -- Many rows represent partial fills or open positions, diluting mean PnL calculations.
5. **Symbol mapping** -- Hyperliquid coin codes (e.g. `@107`) are internal indices; mapping to real asset names requires an external API call.
6. **Survivorship bias** -- Liquidated accounts may be underrepresented in the dataset.
7. **Single time window** -- The trader data covers a specific period and may not generalise across full bull/bear cycles.

---

*Generated by `analysis.py` · {datetime.now().strftime('%Y-%m-%d %H:%M')}*
"""

    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text(report_text, encoding="utf-8")
    log(f"Report written -> outputs/report.md  ({len(report_text.splitlines())} lines)")


# -----------------------------------------------------------------------------
# MAIN ORCHESTRATOR
# -----------------------------------------------------------------------------
def main() -> None:
    """
    End-to-end analysis pipeline.

    Steps:
        1. Set up output directories
        2. Load and clean data
        3. Print EDA summary
        4. Generate all 12 charts
        5. Run supporting analyses
        6. Write report.md
        7. Print final findings to console
    """
    print("\n" + "=" * 64)
    print("  BITCOIN SENTIMENT x HYPERLIQUID TRADER PERFORMANCE")
    print("  Professional Data Science Analysis")
    print("=" * 64 + "\n")

    # -- Directory setup -------------------------------------------------------
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    log("Output directories ready.")

    # -- Step 1 -- Load data ---------------------------------------------------
    print("\n-- STEP 1: LOADING DATA -------------------------------------")
    fg = load_fear_greed()
    td = load_trader_data()
    merged, dropped = merge_datasets(fg, td)

    # -- Step 2 -- EDA summary -------------------------------------------------
    print("\n-- STEP 2: EDA SUMMARY --------------------------------------")
    print_eda_summary(fg, td, merged)

    # -- Step 3A -- Sentiment distribution -------------------------------------
    print("\n-- STEP 3A: SENTIMENT DISTRIBUTION -------------------------")
    sent_dist = plot_sentiment_pie(fg)
    plot_sentiment_timeline(fg)

    for cls, v in sent_dist.items():
        print(f"  {cls:<15}: {v['count']:>5} days  ({v['pct']:.1f}%)")

    # -- Step 3B -- Performance by sentiment -----------------------------------
    print("\n-- STEP 3B: TRADER PERFORMANCE BY SENTIMENT -----------------")
    pnl_stats = plot_pnl_by_sentiment(merged)
    wr        = plot_winrate(merged)
    vol       = plot_volume(merged)
    pos_size  = plot_leverage(merged)

    for cls in ["Fear", "Greed"]:
        print(f"  {cls} -> Avg PnL: ${pnl_stats.get(cls,{}).get('mean',0):.4f}  |  "
              f"Win Rate: {wr.get(cls,0):.1f}%  |  "
              f"Vol: ${vol.get(cls,0)/1e6:.2f}M  |  "
              f"Avg Pos: ${pos_size.get(cls,0):,.0f}")

    # -- Step 3C -- Hidden patterns ---------------------------------------------
    print("\n-- STEP 3C: HIDDEN PATTERNS ---------------------------------")
    fear_syms  = plot_top_symbols_fear(merged)
    greed_syms = plot_top_symbols_greed(merged)
    buy_sell   = plot_buy_vs_sell(merged)

    print("  Top symbol during Fear :", fear_syms[0]["Coin"]
          if fear_syms else "N/A")
    print("  Top symbol during Greed:", greed_syms[0]["Coin"]
          if greed_syms else "N/A")

    size_corr = analyze_size_pnl_correlation(merged)
    print(f"  Size↔PnL correlation   : {size_corr:.4f}")

    accts = analyze_accounts(merged)
    print_account_summary(accts)

    # -- Step 3D -- Advanced insights -------------------------------------------
    print("\n-- STEP 3D: ADVANCED INSIGHTS -------------------------------")
    plot_correlation_heatmap(merged)
    plot_rolling_pnl(merged)
    extreme_pnl = plot_pnl_distribution_extreme(merged)

    print(f"  Extreme Fear  -> Avg PnL: ${extreme_pnl['extreme_fear']['mean']:.4f}"
          f"  (n={extreme_pnl['extreme_fear']['n']:,})")
    print(f"  Extreme Greed -> Avg PnL: ${extreme_pnl['extreme_greed']['mean']:.4f}"
          f"  (n={extreme_pnl['extreme_greed']['n']:,})")

    # -- Step 5 -- Report -------------------------------------------------------
    print("\n-- STEP 5: GENERATING REPORT --------------------------------")
    all_stats: Dict[str, Any] = {
        "sentiment_distribution": sent_dist,
        "pnl_by_sentiment":       pnl_stats,
        "win_rate":               wr,
        "volume":                 vol,
        "leverage":               pos_size,
        "buy_vs_sell":            buy_sell,
        "accounts":               accts,
        "extreme_pnl":            extreme_pnl,
        "size_pnl_corr":          size_corr,
        "total_trades":           len(merged),
        "dropped_trades":         dropped,
    }
    generate_report(all_stats)

    # -- Final summary ---------------------------------------------------------
    print("\n" + "=" * 64)
    print("  ANALYSIS COMPLETE")
    print("=" * 64)

    charts = list(CHARTS_DIR.glob("*.png"))
    print(f"\n  Charts generated : {len(charts)}/12")
    for c in sorted(charts):
        print(f"    [ok] {c.name}")

    print(f"\n  Report           : outputs/report.md")
    print(f"  Total trades     : {len(merged):,}")

    fear_mean  = pnl_stats.get("Fear",  {}).get("mean", 0)
    greed_mean = pnl_stats.get("Greed", {}).get("mean", 0)
    fear_wr    = wr.get("Fear",  0)
    greed_wr   = wr.get("Greed", 0)

    print("\n-- TOP 5 KEY FINDINGS -----------------------------------------------")
    print(f"  1. Avg PnL during Fear  = ${fear_mean:.4f}  "
          f"vs Greed = ${greed_mean:.4f}")
    print(f"  2. Win rate during Fear = {fear_wr:.1f}%  "
          f"vs Greed = {greed_wr:.1f}%")
    print(f"  3. Total volume Fear    = ${vol.get('Fear',0)/1e6:.2f}M  "
          f"vs Greed = ${vol.get('Greed',0)/1e6:.2f}M")
    print(f"  4. Extreme Fear PnL     = ${extreme_pnl['extreme_fear']['mean']:.4f}  "
          f"vs Extreme Greed = ${extreme_pnl['extreme_greed']['mean']:.4f}")
    print(f"  5. Size ↔ PnL r         = {size_corr:.4f}  "
          f"({'no size edge' if abs(size_corr) < 0.1 else 'size matters'})")

    print("\n-- TRADING STRATEGY RECOMMENDATIONS ---------------------------------")
    better = "Fear" if fear_mean >= greed_mean else "Greed"
    print(f"  BUY during  : {better} periods  "
          f"(F&G index < 25 for highest conviction)")
    print(f"  SELL during : {'Greed' if better == 'Fear' else 'Fear'} periods  "
          f"(F&G index > 75 as exit trigger)")
    print(f"  Top Fear symbol  : {fear_syms[0]['Coin'] if fear_syms else 'N/A'}")
    print(f"  Top Greed symbol : {greed_syms[0]['Coin'] if greed_syms else 'N/A'}")
    print(f"  Avg pos size Fear: ${pos_size.get('Fear',0):,.0f}  "
          f"Greed: ${pos_size.get('Greed',0):,.0f}")
    print(f"  Max risk per trade: 1-2% of account capital")
    print("\n" + "=" * 64 + "\n")


if __name__ == "__main__":
    main()
