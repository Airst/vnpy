"""
Analyze why certain stocks consistently rank at top of V15 signal.

Examines:
1. Daily signal scores and ranks for the target stocks
2. Cross-sectional signal distribution (spread between ranks)
3. Raw model output (total_score) vs rank-normalized signal
4. Stocks that replaced these when they dropped out of top-5
"""
import polars as pl
import numpy as np
from datetime import datetime

# Load signal parquet
SIGNAL_PATH = "/home/airst/Workspace/vnpy/core/alpha_db/signal/ashare_mlp_signal_v15.parquet"
TARGET_STOCKS = ["603273.SSE", "603325.SSE", "603209.SSE", "603202.SSE"]
START_DATE = datetime(2026, 4, 1)
END_DATE = datetime(2026, 5, 28)

print("=" * 80)
print("LOADING SIGNAL DATA")
print("=" * 80)

df = pl.read_parquet(SIGNAL_PATH)
print(f"Total rows: {len(df):,}")
print(f"Columns: {df.columns}")
print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
print(f"Unique stocks: {df['vt_symbol'].n_unique()}")

# Check if target stocks exist with the right suffix
sample_symbols = df['vt_symbol'].unique().sort().to_list()[:20]
print(f"\nSample symbols: {sample_symbols}")

# Check for our target stocks
for s in TARGET_STOCKS:
    count = df.filter(pl.col("vt_symbol") == s).height
    if count == 0:
        # Try alternative format
        alt = s.replace(".SSE", ".SH")
        count2 = df.filter(pl.col("vt_symbol") == alt).height
        if count2 > 0:
            print(f"  {s} not found, but {alt} has {count2} rows")
        else:
            print(f"  {s} NOT FOUND (also tried {alt})")
    else:
        print(f"  {s}: {count} rows")

# Detect symbol format
if df.filter(pl.col("vt_symbol").str.contains("SSE")).height == 0:
    # Try .SH format
    TARGET_STOCKS = [s.replace(".SSE", ".SH") for s in TARGET_STOCKS]
    print(f"\nAdjusted target stocks to: {TARGET_STOCKS}")

# Filter to our date range
period_df = df.filter(
    (pl.col("datetime") >= START_DATE) & (pl.col("datetime") <= END_DATE)
)
print(f"\nPeriod rows: {len(period_df):,}")
print(f"Period dates: {period_df['datetime'].n_unique()} trading days")

# Compute cross-sectional rank per day (on total_score = raw model output)
period_df = period_df.with_columns([
    pl.col("total_score").rank(method="average", descending=True).over("datetime").alias("rank_desc"),
    pl.col("total_score").count().over("datetime").alias("n_stocks"),
])

print("\n" + "=" * 80)
print("SECTION 1: DAILY SIGNAL SCORES AND RANKS FOR TARGET STOCKS")
print("=" * 80)

target_df = period_df.filter(pl.col("vt_symbol").is_in(TARGET_STOCKS))
if target_df.height == 0:
    print("WARNING: No data found for target stocks in this period!")
    print("Available symbols containing '603273':")
    matches = df.filter(pl.col("vt_symbol").str.contains("603273"))
    print(matches['vt_symbol'].unique())
else:
    # Show daily ranks
    for stock in TARGET_STOCKS:
        stock_df = target_df.filter(pl.col("vt_symbol") == stock).sort("datetime")
        if stock_df.height == 0:
            print(f"\n{stock}: NO DATA")
            continue
        print(f"\n{'─' * 60}")
        print(f"Stock: {stock}")
        print(f"{'─' * 60}")
        print(f"{'Date':<12} {'total_score':>12} {'final_signal':>14} {'rank_desc':>10} {'n_stocks':>10}")
        for row in stock_df.iter_rows(named=True):
            dt = row['datetime'].strftime("%Y-%m-%d") if hasattr(row['datetime'], 'strftime') else str(row['datetime'])
            print(f"{dt:<12} {row['total_score']:>12.6f} {row['final_signal']:>14.6f} {int(row['rank_desc']):>10} {int(row['n_stocks']):>10}")
        
        # Summary stats
        days_top5 = stock_df.filter(pl.col("rank_desc") <= 5).height
        days_top10 = stock_df.filter(pl.col("rank_desc") <= 10).height
        days_top20 = stock_df.filter(pl.col("rank_desc") <= 20).height
        total_days = stock_df.height
        print(f"\nSummary: Top-5: {days_top5}/{total_days} days ({100*days_top5/total_days:.1f}%)")
        print(f"         Top-10: {days_top10}/{total_days} days ({100*days_top10/total_days:.1f}%)")
        print(f"         Top-20: {days_top20}/{total_days} days ({100*days_top20/total_days:.1f}%)")


print("\n" + "=" * 80)
print("SECTION 2: CROSS-SECTIONAL SIGNAL DISTRIBUTION (total_score)")
print("=" * 80)

# For each day, show the spread between top ranks
dates_in_period = period_df['datetime'].unique().sort().to_list()

print(f"\n{'Date':<12} {'Rank1':>8} {'Rank2':>8} {'Rank3':>8} {'Rank5':>8} {'Rank10':>8} {'Rank20':>8} {'Median':>8} {'Rank1-5':>8} {'Rank1-20':>9}")
print("─" * 105)

spread_data = []
for dt in dates_in_period:
    day_df = period_df.filter(pl.col("datetime") == dt).sort("total_score", descending=True)
    scores = day_df['total_score'].to_list()
    n = len(scores)
    if n < 20:
        continue
    rank1 = scores[0]
    rank2 = scores[1]
    rank3 = scores[2]
    rank5 = scores[4]
    rank10 = scores[9]
    rank20 = scores[19]
    median = scores[n // 2]
    
    dt_str = dt.strftime("%Y-%m-%d") if hasattr(dt, 'strftime') else str(dt)
    spread_1_5 = rank1 - rank5
    spread_1_20 = rank1 - rank20
    print(f"{dt_str:<12} {rank1:>8.4f} {rank2:>8.4f} {rank3:>8.4f} {rank5:>8.4f} {rank10:>8.4f} {rank20:>8.4f} {median:>8.4f} {spread_1_5:>8.4f} {spread_1_20:>9.4f}")
    spread_data.append({
        "date": dt, "rank1": rank1, "rank5": rank5, "rank10": rank10, 
        "rank20": rank20, "median": median, "spread_1_5": spread_1_5, "spread_1_20": spread_1_20
    })

if spread_data:
    spread_df = pl.DataFrame(spread_data)
    print(f"\nAverage spread Rank1-Rank5:  {spread_df['spread_1_5'].mean():.4f}")
    print(f"Average spread Rank1-Rank10: {(spread_df['rank1'] - spread_df['rank10']).mean():.4f}")
    print(f"Average spread Rank1-Rank20: {spread_df['spread_1_20'].mean():.4f}")
    print(f"Average Rank1 total_score:   {spread_df['rank1'].mean():.4f}")
    print(f"Average Median total_score:  {spread_df['median'].mean():.4f}")
    print(f"Average Rank1-Median spread: {(spread_df['rank1'] - spread_df['median']).mean():.4f}")


print("\n" + "=" * 80)
print("SECTION 3: RAW MODEL OUTPUT (total_score) DISTRIBUTION")
print("=" * 80)

# Check if total_score is compressed or if specific stocks are true outliers
for dt in dates_in_period[:5]:  # Sample 5 days
    day_df = period_df.filter(pl.col("datetime") == dt).sort("total_score", descending=True)
    scores = day_df['total_score'].to_numpy()
    dt_str = dt.strftime("%Y-%m-%d") if hasattr(dt, 'strftime') else str(dt)
    
    print(f"\n{dt_str}: N={len(scores)}")
    print(f"  Top-5 scores: {scores[:5]}")
    print(f"  Percentiles: p99={np.percentile(scores, 99):.4f}, p95={np.percentile(scores, 95):.4f}, "
          f"p75={np.percentile(scores, 75):.4f}, p50={np.percentile(scores, 50):.4f}, "
          f"p25={np.percentile(scores, 25):.4f}, p5={np.percentile(scores, 5):.4f}")
    print(f"  Std: {np.std(scores):.4f}, Range: {scores.max()-scores.min():.4f}")
    
    # How far is rank 1 from rank 5 in std-dev units?
    std = np.std(scores)
    if std > 0:
        print(f"  Rank1 z-score: {(scores[0] - np.mean(scores))/std:.2f}")
        print(f"  Rank5 z-score: {(scores[4] - np.mean(scores))/std:.2f}")
        print(f"  Rank1 vs Rank5 gap in stdevs: {(scores[0] - scores[4])/std:.3f}")

    # Check where target stocks are in this day
    day_targets = day_df.filter(pl.col("vt_symbol").is_in(TARGET_STOCKS))
    if day_targets.height > 0:
        print(f"  Target stocks on this day:")
        for row in day_targets.iter_rows(named=True):
            z = (row['total_score'] - np.mean(scores)) / std if std > 0 else 0
            print(f"    {row['vt_symbol']}: score={row['total_score']:.4f}, rank={int(row['rank_desc'])}, z={z:.2f}")


print("\n" + "=" * 80)
print("SECTION 4: RANK NORMALIZATION EFFECT")
print("=" * 80)

# The rank normalization formula: ((rank/count) - 0.5) * 3.46
# For rank 998/998: (1.0 - 0.5) * 3.46 = 1.73
# For rank 997/998: (997/998 - 0.5) * 3.46 ≈ 1.727
# For rank 993/998: (993/998 - 0.5) * 3.46 ≈ 1.711
print("Rank normalization: final_signal = ((ascending_rank / count) - 0.5) * 3.46")
print("\nTheoretical final_signal for top ranks (out of 998 stocks):")
for rank_asc in [998, 997, 996, 995, 994, 993, 990, 980]:
    signal = ((rank_asc / 998) - 0.5) * 3.46
    rank_desc = 999 - rank_asc
    print(f"  Ascending rank {rank_asc} (desc rank {rank_desc}): signal = {signal:.4f}")

# Compare actual total_score differentiation vs final_signal differentiation
print("\nKey question: Does rank normalization COMPRESS the top?")
print("Let's compare raw score ratio vs signal ratio for top stocks:")
for dt in dates_in_period[:5]:
    day_df = period_df.filter(pl.col("datetime") == dt).sort("total_score", descending=True)
    dt_str = dt.strftime("%Y-%m-%d") if hasattr(dt, 'strftime') else str(dt)
    top5 = day_df.head(5)
    
    raw_scores = top5['total_score'].to_list()
    signals = top5['final_signal'].to_list()
    
    if len(raw_scores) >= 5:
        raw_range = raw_scores[0] - raw_scores[4]
        sig_range = signals[0] - signals[4] if len(signals) >= 5 else 0
        print(f"  {dt_str}: raw_score range top5 = {raw_range:.6f}, signal range top5 = {sig_range:.6f}")


print("\n" + "=" * 80)
print("SECTION 5: STOCKS THAT REPLACED TARGETS IN TOP-5")
print("=" * 80)

# Find dates where target stocks were in top-5, then find dates where they dropped out
# and show which stocks took their place

# Track top-5 composition over time
top5_history = []
for dt in dates_in_period:
    day_df = period_df.filter(pl.col("datetime") == dt).sort("total_score", descending=True)
    top5_stocks = day_df.head(5)['vt_symbol'].to_list()
    top5_history.append({"date": dt, "top5": top5_stocks})

# Find transitions: when a target stock leaves top-5
print("\nTop-5 composition over time:")
print(f"{'Date':<12} {'Top 5 stocks'}")
print("─" * 80)
for entry in top5_history:
    dt_str = entry['date'].strftime("%Y-%m-%d") if hasattr(entry['date'], 'strftime') else str(entry['date'])
    stocks = entry['top5']
    # Highlight target stocks
    display = []
    for s in stocks:
        if s in TARGET_STOCKS:
            display.append(f"*{s}*")
        else:
            display.append(s)
    print(f"{dt_str:<12} {', '.join(display)}")

# Summary: how often each stock appears in top-5
print("\n\nTop-5 appearance frequency (entire period):")
from collections import Counter
all_top5 = []
for entry in top5_history:
    all_top5.extend(entry['top5'])
counter = Counter(all_top5)
for stock, count in counter.most_common(20):
    marker = " *** TARGET ***" if stock in TARGET_STOCKS else ""
    print(f"  {stock}: {count}/{len(top5_history)} days ({100*count/len(top5_history):.1f}%){marker}")


print("\n" + "=" * 80)
print("SECTION 6: SIGNAL PERSISTENCE / AUTOCORRELATION")
print("=" * 80)

# Since the model is retrained every 90 days, within a window the model weights are fixed
# Check if the same stocks stay at top across the entire window
# Find the retrain boundary
print("\nChecking if signal is from a single model window (same weights)...")
print("If yes, then the ranking is purely a function of daily factor values with fixed weights")

# Look at rank autocorrelation for target stocks
for stock in TARGET_STOCKS:
    stock_df = period_df.filter(pl.col("vt_symbol") == stock).sort("datetime")
    if stock_df.height < 2:
        continue
    ranks = stock_df['rank_desc'].to_numpy()
    scores = stock_df['total_score'].to_numpy()
    print(f"\n{stock}:")
    print(f"  Rank range: {ranks.min()} - {ranks.max()} (mean: {ranks.mean():.1f})")
    print(f"  Score range: {scores.min():.4f} - {scores.max():.4f} (std: {scores.std():.4f})")
    if len(ranks) > 1:
        rank_changes = np.diff(ranks)
        print(f"  Daily rank changes: mean={rank_changes.mean():.1f}, max_jump={np.abs(rank_changes).max()}")

print("\n\nDONE.")
