"""
Final deep dive: What specifically makes these 603xxx stocks rank so high?

Key findings so far:
1. NOT simply low-vol + small-cap (0% overlap with simple rule)
2. 93% of persistent top-20 are SSE stocks with 603/601 prefix
3. Target stocks have NEGATIVE momentum but stay at top
4. Volatility is below-average but not extreme (11-47th percentile)
5. Training returns are mixed (-25% to +20%)

New hypotheses to test:
A. IPO recency effect: 603xxx stocks are relatively recent IPOs, 
   short data history → model overfits to limited samples
B. GP factors (13 genetic-programming factors) may capture specific patterns
C. Factor profile similarity: these stocks may share a unique combination
   that maps to a specific attention pattern the model learned
D. The ~994 stocks in the signal file vs ~2906 in daily dir means the 
   model operates on a FILTERED universe - these might dominate within that filter
"""
import sys
sys.path.insert(0, '/home/airst/Workspace/vnpy')

import polars as pl
import numpy as np
from datetime import datetime
from pathlib import Path
from collections import Counter
import json
import warnings
warnings.filterwarnings('ignore')

TARGET_STOCKS = ["603273.SSE", "603325.SSE", "603209.SSE", "603202.SSE"]
DAILY_DIR = Path("/home/airst/Workspace/vnpy/core/alpha_db/daily")

# Load signal data
signal_df = pl.read_parquet("/home/airst/Workspace/vnpy/core/alpha_db/signal/ashare_mlp_signal_v15.parquet")

print("=" * 80)
print("SECTION 1: DATA HISTORY LENGTH (IPO recency effect)")
print("=" * 80)

# Get data length for ALL stocks in the signal file
signal_stocks = signal_df["vt_symbol"].unique().to_list()
print(f"Stocks in signal file: {len(signal_stocks)}")

# Check how many days of data each stock has
stock_data_lengths = {}
for sym in signal_stocks:
    try:
        df = pl.read_parquet(DAILY_DIR / f"{sym}.parquet")
        stock_data_lengths[sym] = df.height
    except:
        pass

# Get persistent top-20
period_sig = signal_df.filter(
    (pl.col("datetime") >= datetime(2026, 4, 1)) & 
    (pl.col("datetime") <= datetime(2026, 5, 28))
)
dates = period_sig["datetime"].unique().sort().to_list()
top20_counts = Counter()
for dt in dates:
    day = period_sig.filter(pl.col("datetime") == dt).sort("total_score", descending=True)
    top20 = day.head(20)["vt_symbol"].to_list()
    for s in top20:
        top20_counts[s] += 1

frequent_top20 = [s for s, c in top20_counts.most_common(50) if c > len(dates) * 0.5]

print(f"\nData length for persistent top-20 vs all stocks:")
print(f"{'Stock':<15} {'Days':>6} {'IPO est':>12}")
print("─" * 40)
for stock in frequent_top20:
    days = stock_data_lengths.get(stock, 0)
    # Estimate IPO date from data length (approximate)
    print(f"  {stock:<13} {days:>6}", end="")
    if stock in TARGET_STOCKS:
        print(" *** TARGET ***", end="")
    print()

# Overall distribution
all_lengths = list(stock_data_lengths.values())
top20_lengths = [stock_data_lengths.get(s, 0) for s in frequent_top20]
print(f"\nAll stocks: mean={np.mean(all_lengths):.0f}, median={np.median(all_lengths):.0f}")
print(f"Top-20 stocks: mean={np.mean(top20_lengths):.0f}, median={np.median(top20_lengths):.0f}")

# What's the 10th percentile of data length?
p10 = np.percentile(all_lengths, 10)
p25 = np.percentile(all_lengths, 25)
print(f"10th percentile length: {p10:.0f} days")
print(f"25th percentile length: {p25:.0f} days")

# How many top-20 stocks are in bottom quartile of data length?
short_history = [s for s in frequent_top20 if stock_data_lengths.get(s, 0) < p25]
print(f"\nTop-20 stocks with SHORT history (<{p25:.0f} days): {len(short_history)}/{len(frequent_top20)}")
for s in short_history:
    print(f"  {s}: {stock_data_lengths.get(s, 0)} days")


print("\n\n" + "=" * 80)
print("SECTION 2: SIGNAL STICKINESS WITHIN MODEL WINDOW")
print("=" * 80)

# The model was last retrained on 2026-03-02. Its prediction window is ~90 days.
# Within this window, model weights are FIXED. Only factor values change.
# If factors are "stable" for these stocks, the signal won't change much.

# Check: how volatile is the total_score for these stocks vs others?
print("\nSignal (total_score) volatility during Apr-May 2026:")
print(f"{'Stock':<15} {'Score_Std':>10} {'Score_Range':>12} {'Score_Mean':>11}")
print("─" * 50)

all_score_stds = []
for stock in signal_stocks:
    stock_sig = period_sig.filter(pl.col("vt_symbol") == stock)
    if stock_sig.height >= 10:
        scores = stock_sig["total_score"].to_numpy()
        all_score_stds.append(np.std(scores))

for stock in frequent_top20[:15]:
    stock_sig = period_sig.filter(pl.col("vt_symbol") == stock).sort("datetime")
    if stock_sig.height > 0:
        scores = stock_sig["total_score"].to_numpy()
        marker = " ***" if stock in TARGET_STOCKS else ""
        print(f"  {stock:<13} {np.std(scores):>10.4f} {scores.max()-scores.min():>12.4f} {np.mean(scores):>11.4f}{marker}")

print(f"\n  All stocks: mean_std={np.mean(all_score_stds):.4f}, median_std={np.median(all_score_stds):.4f}")


print("\n\n" + "=" * 80)
print("SECTION 3: GP FACTOR CONTRIBUTIONS (checking gp_factors.json)")
print("=" * 80)

# Load GP factors registry
gp_path = "/home/airst/Workspace/vnpy/core/alpha/gp_factors.json"
with open(gp_path, 'r') as f:
    gp_data = json.load(f)

gp_factors = gp_data.get("factors", [])
validated = [f for f in gp_factors if f.get("status") == "validated"]
print(f"Validated GP factors: {len(validated)}")
for i, gp in enumerate(validated):
    print(f"  {i+1}. {gp.get('name', 'unnamed')}: {gp.get('expression', 'N/A')[:80]}")
    if 'ic_stats' in gp:
        print(f"     IC stats: {gp['ic_stats']}")


print("\n\n" + "=" * 80)
print("SECTION 4: SCORE TRAJECTORY & MODEL WINDOW BOUNDARY")
print("=" * 80)

# Show the full score history for target stocks - when did they start climbing?
# Look at the broader signal history
broad_period = signal_df.filter(
    (pl.col("datetime") >= datetime(2026, 1, 1)) & 
    (pl.col("datetime") <= datetime(2026, 5, 28))
)

print("\nScore trajectory (weekly samples) from Jan 2026:")
print(f"{'Date':<12}", end="")
for s in TARGET_STOCKS:
    short = s.split(".")[0][-4:]
    print(f" {short:>7}", end="")
print()
print("─" * 45)

broad_dates = broad_period["datetime"].unique().sort().to_list()
for dt in broad_dates[::5]:  # Weekly
    dt_str = dt.strftime("%Y-%m-%d") if hasattr(dt, 'strftime') else str(dt)
    print(f"{dt_str:<12}", end="")
    for s in TARGET_STOCKS:
        row = broad_period.filter((pl.col("datetime") == dt) & (pl.col("vt_symbol") == s))
        if row.height > 0:
            score = row["total_score"][0]
            print(f" {score:>7.3f}", end="")
        else:
            print(f" {'N/A':>7}", end="")
    print()


# Model retrain date is 2026-03-02
# Check if there's a sudden jump in scores around that date
print("\n\nModel retrain boundary: 2026-03-02")
print("Checking score jump around retrain date:")
boundary_period = signal_df.filter(
    (pl.col("datetime") >= datetime(2026, 2, 15)) & 
    (pl.col("datetime") <= datetime(2026, 3, 15))
)

for stock in TARGET_STOCKS:
    stock_sig = boundary_period.filter(pl.col("vt_symbol") == stock).sort("datetime")
    if stock_sig.height > 0:
        print(f"\n{stock}:")
        for row in stock_sig.iter_rows(named=True):
            dt_str = row['datetime'].strftime("%Y-%m-%d")
            print(f"  {dt_str}: score={row['total_score']:.4f}, signal={row['final_signal']:.4f}")


print("\n\n" + "=" * 80)
print("SECTION 5: CROSS-SECTION ON 2026-05-07 - RAW SCORE DISTRIBUTION BY PREFIX")
print("=" * 80)

# Check if 603xxx stocks systematically score higher
sig_day = signal_df.filter(pl.col("datetime") == datetime(2026, 5, 7)).sort("total_score", descending=True)

# Add prefix column
sig_day = sig_day.with_columns(
    pl.col("vt_symbol").str.slice(0, 3).alias("prefix"),
    pl.col("vt_symbol").str.split(".").list.last().alias("exchange"),
)

# Score distribution by prefix
print("\nMean total_score by stock code prefix (top prefixes by count):")
prefix_stats = sig_day.group_by("prefix").agg([
    pl.col("total_score").mean().alias("mean_score"),
    pl.col("total_score").median().alias("median_score"),
    pl.col("total_score").std().alias("std_score"),
    pl.len().alias("count"),
]).sort("mean_score", descending=True)

print(f"{'Prefix':<8} {'Count':>6} {'Mean_Score':>11} {'Median':>8} {'Std':>8}")
print("─" * 45)
for row in prefix_stats.iter_rows(named=True):
    print(f"  {row['prefix']:<6} {row['count']:>6} {row['mean_score']:>11.4f} {row['median_score']:>8.4f} {row['std_score']:>8.4f}")


# List ALL 603xxx stocks and their scores on this date
print("\n\nAll 603xxx stocks in signal (sorted by score):")
stocks_603 = sig_day.filter(pl.col("prefix") == "603").sort("total_score", descending=True)
print(f"Total 603xxx stocks: {stocks_603.height}")
print(f"\nTop-20 of 603xxx:")
for i, row in enumerate(stocks_603.head(20).iter_rows(named=True)):
    marker = " ***" if row["vt_symbol"] in TARGET_STOCKS else ""
    print(f"  {i+1:>3}. {row['vt_symbol']:<13} score={row['total_score']:.4f}{marker}")

print(f"\nBottom-5 of 603xxx:")
for row in stocks_603.tail(5).iter_rows(named=True):
    print(f"  {row['vt_symbol']:<13} score={row['total_score']:.4f}")


print("\n\n" + "=" * 80)
print("SECTION 6: CHECK DATA AVAILABILITY IN TRAINING WINDOW")
print("=" * 80)

# Critical: the model trains on 600 days ending ~2026-03-01.
# If a stock has only ~200-300 days of history, it has MUCH less training data.
# This could lead to the model learning spurious patterns from limited samples.

print("\nHow much training data did each target stock contribute?")
print("(Model window: ~2024-03-01 to 2026-03-01, 600 training days)")

for stock in frequent_top20[:15]:
    try:
        df = pl.read_parquet(DAILY_DIR / f"{stock}.parquet")
        train_data = df.filter(
            (pl.col("datetime") >= datetime(2024, 3, 1)) & 
            (pl.col("datetime") <= datetime(2026, 3, 1))
        )
        total_data = df.height
        marker = " ***" if stock in TARGET_STOCKS else ""
        
        # Check if data starts AFTER the training window start
        first_date = df["datetime"].min()
        first_str = first_date.strftime("%Y-%m-%d") if hasattr(first_date, 'strftime') else str(first_date)
        
        print(f"  {stock:<13} total={total_data:>5}d, in_train={train_data.height:>4}d, first_date={first_str}{marker}")
    except Exception as e:
        print(f"  {stock:<13} Error: {e}")


print("\n\nDONE.")
