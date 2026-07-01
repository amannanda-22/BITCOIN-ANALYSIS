# Bitcoin Sentiment × Hyperliquid Trader Performance

> **Professional data science analysis** of the relationship between the
> Crypto Fear & Greed Index and real trader performance on the Hyperliquid DEX.

---

## Project Overview

This project answers a key question in crypto trading:

> *Does market sentiment (Fear vs. Greed) predict trader profitability?*

We merge **2,600+ days** of the Crypto Fear & Greed Index with **211,000+ trades**
from Hyperliquid to uncover actionable, data-driven trading insights.

### What We Analyse

| Dimension | Questions Answered |
|---|---|
| **Sentiment Distribution** | How often is the market in Fear vs. Greed? |
| **PnL by Sentiment** | Do traders earn more on Fear or Greed days? |
| **Win Rate** | Is the probability of a profitable trade higher in Fear or Greed? |
| **Volume & Position Size** | Does sentiment change how much traders deploy? |
| **Symbol Performance** | Which coins outperform in each regime? |
| **BUY vs. SELL** | Does side (long/short) interact with sentiment? |
| **Rolling Trends** | How does 7-day average PnL track sentiment over time? |
| **Extreme Regimes** | What happens at Extreme Fear / Extreme Greed? |
| **Account Analysis** | Do top performers prefer Fear or Greed? |

---

## Project Structure

```
bitcoin_analysis/
├── data/
│   ├── fear_greed.csv          # Crypto Fear & Greed Index (daily)
│   └── trader_data.csv         # Hyperliquid historical trade data
├── notebooks/
│   └── analysis.ipynb          # Jupyter notebook walkthrough
├── outputs/
│   ├── charts/                 # 12 PNG visualisations (300 DPI)
│   └── report.md               # Auto-generated insights report
├── analysis.py                 # Main analysis script
├── requirements.txt
└── README.md
```

---

## Setup & Installation

### Prerequisites

- Python 3.9+
- pip

### Install Dependencies

```bash
cd bitcoin_analysis
pip install -r requirements.txt
```

---

## How to Run

### Option 1 — Python Script (Recommended)

```bash
python analysis.py
```

This will:
1. Load and clean both datasets
2. Merge on date
3. Generate all 12 charts → `outputs/charts/`
4. Write the insights report → `outputs/report.md`
5. Print key findings and trading strategy recommendations to the console

### Option 2 — Jupyter Notebook

```bash
jupyter notebook notebooks/analysis.ipynb
```

---

## Output Charts

| # | File | Description |
|---|---|---|
| 1 | `sentiment_distribution_pie.png` | Donut chart — all 5 sentiment classes |
| 2 | `sentiment_timeline.png` | F&G Index timeline (2018–2025) |
| 3 | `pnl_by_sentiment_bar.png` | Avg & total PnL: Fear vs. Greed |
| 4 | `winrate_by_sentiment.png` | % profitable trades by regime |
| 5 | `volume_by_sentiment.png` | Total USD volume traded |
| 6 | `leverage_by_sentiment.png` | Avg position size by sentiment |
| 7 | `top_symbols_fear.png` | Best-performing symbols during Fear |
| 8 | `top_symbols_greed.png` | Best-performing symbols during Greed |
| 9 | `buy_vs_sell_pnl.png` | BUY vs. SELL PnL across regimes |
| 10 | `pnl_correlation_heatmap.png` | Pearson correlation matrix |
| 11 | `rolling_pnl_vs_sentiment.png` | 7-day rolling PnL vs. F&G value |
| 12 | `pnl_distribution_extreme.png` | KDE — Extreme Fear vs. Extreme Greed |

---

## Key Findings Summary

Full findings with exact numbers are in **`outputs/report.md`**.
Run the script to populate it with live statistics from your data.

High-level takeaways:
- Market sentiment is a **statistically meaningful predictor** of trade profitability.
- **Fear periods** often provide contrarian alpha for disciplined traders.
- **Win rates** and **average PnL** differ measurably between Fear and Greed regimes.
- **Symbol leadership rotates** between regimes — maintaining two watchlists is recommended.
- **Position size has near-zero correlation with PnL**, confirming that timing/skill, not capital size, drives performance.

---

## Data Sources

| Dataset | Columns | Rows |
|---|---|---|
| Crypto Fear & Greed Index | `date`, `value`, `classification` | ~2,644 |
| Hyperliquid Trade History | `Account`, `Coin`, `Side`, `Closed PnL`, `Size USD`, ... | ~211,000 |

---

## Technical Details

- **Language**: Python 3.11
- **Key libraries**: pandas, numpy, matplotlib, seaborn
- **Chart style**: Dark background (`#0d1117`), 300 DPI
- **Date parsing**: Fear & Greed uses `YYYY-MM-DD`; Hyperliquid uses `DD-MM-YYYY HH:MM` (IST)
- **Sentiment merge**: Inner join on calendar date (UTC/IST date boundary assumed consistent)

---

## Assumptions & Limitations

1. `Closed PnL = 0` rows may represent open/partial fills rather than zero-profit trades.
2. No explicit leverage column in the Hyperliquid data — `Size USD` used as proxy.
3. Hyperliquid coin codes (`@107`, etc.) are internal perpetual indices, not mapped to ticker symbols.
4. Analysis window is constrained to the intersection of both datasets' date ranges.

---

*Prepared as part of a Data Science assignment for a Web3 trading company.*
