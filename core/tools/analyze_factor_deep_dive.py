"""
Deep dive: Common structural properties of persistently top-ranked stocks.

Key findings from prior analysis:
- These stocks have LOW volatility (inv_vol_20 high) 
- SMALL cap (size_ln_cap very low percentile)
- NEGATIVE momentum but the model still ranks them #1

This suggests the model has learned a "low-vol small-cap" factor that dominates.
Let's investigate:
1. What industry/sector are these stocks in?
2. Do they share a concept board?
3. What does the GP factor miner produce for them?
4. Cross-check: are ALL low-vol small-caps ranked high, or is it specific to these?
"""
import sys
sys.path.insert(0, '/home/airst/Workspace/vnpy')

import polars as pl
import numpy as np
from datetime import datetime
from pathlib import Path
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

TARGET_STOCKS = ["603273.SSE", "603325.SSE", "603209.SSE", "603202.SSE"]
DAILY_DIR = Path("/home/airst/Workspace/vnpy/core/alpha_db/daily")

# Load signal data to get full top-20 composition
signal_df = pl.read_parquet("/home/airst/Workspace/vnpy/core/alpha_db/signal/ashare_mlp_signal_v15.parquet")

print("=" * 80)
print("SECTION 1: STOCK BASIC INFO (Industry, IPO date, etc)")
print("=" * 80)

# Load stock info
from data_manager.ts_downloader.stock_info_manager import StockInfoManager
info_mgr = StockInfoManager()

# Load info for target stocks + comparison stocks
all_focus_stocks = TARGET_STOCKS + ["603344.SSE", "603082.SSE", "603004.SSE", "601033.SSE", "603210.SSE"]
try:
    all_info = info_mgr.load_data(all_focus_stocks)
    if all_info is not None and not all_info.empty:
        info_df = pl.from_pandas(all_info)
        print(f"Stock info loaded: {info_df.shape}")
        print(f"Columns: {info_df.columns}")
        print(info_df)
    else:
        print("No stock info returned")
except Exception as e:
    print(f"Could not load stock info: {e}")

# Also check frequent top-5 stocks 
print("\n\n" + "=" * 80)
print("SECTION 2: TOP-20 COMPOSITION ANALYSIS (Shared characteristics)")
print("=" * 80)

# Get the top-20 stocks on multiple dates in our period
period_sig = signal_df.filter(
    (pl.col("datetime") >= datetime(2026, 4, 1)) & 
    (pl.col("datetime") <= datetime(2026, 5, 28))
)

# Count top-20 appearances
top20_counts = Counter()
dates = period_sig["datetime"].unique().sort().to_list()
for dt in dates:
    day = period_sig.filter(pl.col("datetime") == dt).sort("total_score", descending=True)
    top20 = day.head(20)["vt_symbol"].to_list()
    for s in top20:
        top20_counts[s] += 1

print(f"\nStocks appearing in top-20 more than 50% of days ({len(dates)} trading days):")
frequent_top20 = [s for s, c in top20_counts.most_common(50) if c > len(dates) * 0.5]
for stock, count in top20_counts.most_common(50):
    if count > len(dates) * 0.5:
        marker = " *** TARGET ***" if stock in TARGET_STOCKS else ""
        print(f"  {stock}: {count}/{len(dates)} days ({100*count/len(dates):.0f}%){marker}")

# Check if these stocks share a common prefix pattern (603xxx = SSE main board small-caps)
print(f"\n\nExchange/Code prefix distribution of frequent top-20 stocks:")
prefixes = Counter()
exchanges = Counter()
for stock in frequent_top20:
    code = stock.split(".")[0]
    exchange = stock.split(".")[1]
    prefixes[code[:3]] += 1
    exchanges[exchange] += 1

print(f"  Exchange: {dict(exchanges)}")
print(f"  Code prefix (3-digit): {dict(prefixes)}")


# ============================================================================
print("\n\n" + "=" * 80)
print("SECTION 3: IS THE MODEL LEARNING 'LOW-VOL SMALL-CAP' AS DOMINANT SIGNAL?")
print("=" * 80)

# Hypothesis: the model has learned that low-volatility + small-cap = high alpha
# This would explain why declining stocks stay at top (their vol is still low historically)
# Let's test: rank stocks by inv_vol_20 * (1/size_ln_cap) and see overlap with model's top-20

from data_manager.ts_downloader.daily_basic_manager import DailyBasicManager

# Load price and basic data for the analysis date
all_symbols = [f.stem for f in DAILY_DIR.glob("*.parquet")]

# Load data for a specific date
test_date = datetime(2026, 5, 7)
print(f"\nTest date: {test_date.strftime('%Y-%m-%d')}")

# Load price data with lookback for volatility calc
dfs = []
for sym in all_symbols:
    try:
        df = pl.read_parquet(DAILY_DIR / f"{sym}.parquet")
        df = df.filter(
            (pl.col("datetime") >= datetime(2026, 3, 1)) & 
            (pl.col("datetime") <= datetime(2026, 5, 28))
        )
        if df.height >= 20:  # Need at least 20 days for vol calc
            df = df.with_columns(pl.lit(sym).alias("vt_symbol"))
            dfs.append(df)
    except Exception:
        pass

price_df = pl.concat(dfs).sort(["vt_symbol", "datetime"])

# Compute factors
price_df = price_df.with_columns([
    (pl.col("close") / pl.col("close").shift(1).over("vt_symbol") - 1).alias("ret_1d"),
])
price_df = price_df.with_columns([
    pl.col("ret_1d").rolling_std(20).over("vt_symbol").alias("vol_20d"),
])

# Load basic for size
db_manager = DailyBasicManager()
basic_pd = db_manager.load_data(all_symbols, "20260301", "20260528")
if not basic_pd.empty:
    basic_df = pl.from_pandas(basic_pd)
    basic_df = basic_df.with_columns(pl.col("datetime").cast(pl.Datetime("us")))
    basic_df = basic_df.select(["vt_symbol", "datetime", "total_mv"])
    price_df = price_df.join(basic_df, on=["vt_symbol", "datetime"], how="left")
    price_df = price_df.sort(["vt_symbol", "datetime"])
    price_df = price_df.with_columns(pl.col("total_mv").forward_fill().over("vt_symbol"))

# Get values on test date
day_df = price_df.filter(pl.col("datetime") == test_date)
day_df = day_df.filter(pl.col("vol_20d").is_not_null() & pl.col("total_mv").is_not_null())

# Compute rank scores: higher inv_vol + lower size = higher score
day_df = day_df.with_columns([
    (1.0 / (pl.col("vol_20d") + 1e-8)).alias("inv_vol"),
    pl.col("total_mv").log().alias("ln_cap"),
])

# Cross-sectional rank
n = day_df.height
day_df = day_df.with_columns([
    pl.col("inv_vol").rank().alias("inv_vol_rank"),
    (n + 1 - pl.col("ln_cap").rank()).alias("small_cap_rank"),  # Higher rank = smaller
])

# Combined: low-vol small-cap score
day_df = day_df.with_columns([
    (pl.col("inv_vol_rank") + pl.col("small_cap_rank")).alias("lowvol_smallcap_score"),
])

day_df = day_df.sort("lowvol_smallcap_score", descending=True)

# Get model's top-20
sig_day = signal_df.filter(pl.col("datetime") == test_date).sort("total_score", descending=True)
model_top20 = sig_day.head(20)["vt_symbol"].to_list()
model_top50 = sig_day.head(50)["vt_symbol"].to_list()

# Our simple rule's top-20
rule_top20 = day_df.head(20)["vt_symbol"].to_list()
rule_top50 = day_df.head(50)["vt_symbol"].to_list()

overlap_20 = len(set(model_top20) & set(rule_top20))
overlap_50 = len(set(model_top50) & set(rule_top50))

print(f"\nModel's Top-20 vs Simple 'Low-Vol + Small-Cap' rule's Top-20:")
print(f"  Overlap: {overlap_20}/20 ({100*overlap_20/20:.0f}%)")
print(f"\nModel's Top-50 vs Simple rule's Top-50:")
print(f"  Overlap: {overlap_50}/50 ({100*overlap_50/50:.0f}%)")

print(f"\nModel's top-20:")
for s in model_top20:
    in_rule = "✓" if s in rule_top50 else " "
    target = "***" if s in TARGET_STOCKS else ""
    # Get rule rank
    rule_rank_row = day_df.filter(pl.col("vt_symbol") == s)
    rule_rank = "N/A"
    if rule_rank_row.height > 0:
        # Find position in sorted df
        rule_sorted = day_df["vt_symbol"].to_list()
        if s in rule_sorted:
            rule_rank = str(rule_sorted.index(s) + 1)
    print(f"  {s} [rule_rank={rule_rank:>4}] {in_rule} {target}")

print(f"\nSimple rule's top-20 (Low-Vol + Small-Cap):")
for s in rule_top20:
    in_model = "✓" if s in model_top50 else " "
    target = "***" if s in TARGET_STOCKS else ""
    print(f"  {s} {in_model} {target}")


# ============================================================================
print("\n\n" + "=" * 80)
print("SECTION 4: MODEL'S TRAINING WINDOW CONTEXT")
print("=" * 80)

# The model was trained on data ending ~2026-03-01 (train window: ~2024-03 to 2026-03)
# During training, what was the performance of these stocks?
# If they had HIGH returns in the training period, the model may have learned
# to associate their factor profile with high scores

print("\nPrice performance of target stocks DURING training window:")
print("(Model trained on ~600 days ending ~2026-03-01)")
print("(Label = future 5-day beta-neutral return, ranked cross-sectionally)")

for stock in TARGET_STOCKS + ["603344.SSE", "603082.SSE"]:
    try:
        df = pl.read_parquet(DAILY_DIR / f"{stock}.parquet")
        
        # Training period performance
        train_data = df.filter(
            (pl.col("datetime") >= datetime(2024, 3, 1)) & 
            (pl.col("datetime") <= datetime(2026, 3, 1))
        ).sort("datetime")
        
        if train_data.height > 0:
            closes = train_data["close"].to_numpy()
            total_ret = (closes[-1] / closes[0] - 1) * 100
            
            # Also check recent performance just before prediction window
            recent = df.filter(
                (pl.col("datetime") >= datetime(2025, 12, 1)) & 
                (pl.col("datetime") <= datetime(2026, 3, 1))
            ).sort("datetime")
            
            if recent.height > 0:
                recent_closes = recent["close"].to_numpy()
                recent_ret = (recent_closes[-1] / recent_closes[0] - 1) * 100
            else:
                recent_ret = float('nan')
            
            print(f"\n{stock}:")
            print(f"  Training period return (2024-03 to 2026-03): {total_ret:+.1f}%")
            print(f"  Recent 3-month (2025-12 to 2026-03): {recent_ret:+.1f}%")
            print(f"  Data points in training: {train_data.height} days")
        else:
            print(f"\n{stock}: Insufficient data in training period")
    except Exception as e:
        print(f"\n{stock}: Error - {e}")


# ============================================================================
print("\n\n" + "=" * 80)
print("SECTION 5: CHECKING IF THE ISSUE IS 'STALE MODEL' (Fixed weights, changing market)")
print("=" * 80)

# The model was trained 2026-03-02. Predictions start from that date.
# During Apr-May, these stocks declined. But the model sees DAILY factor values.
# Key question: which factors are INPUT to the model that WOULD change with price drops?

# Factors that SHOULD change with declining prices:
# - mom_5d, mom_20d, mom_60d → become negative ✓ (confirmed: they ARE negative)
# - bias_5/10/20 → become negative ✓ (confirmed)
# - drawdown_20d → become large negative ✓ (confirmed)
# - rebound_20d → decrease ✓ (confirmed)
# - rsi_14 → decrease

# Factors that DON'T change much:
# - volatility_20d → depends on magnitude of daily returns, not direction
# - inv_vol_20/60 → stays high if decline is gradual/smooth
# - size_ln_cap → barely changes (market cap ≈ price × shares, but shares constant)
# - ep_ratio → 1/PE, changes slowly
# - turnover_mean_20d → depends on trading activity

# KEY INSIGHT: if the model heavily weights LOW VOLATILITY, and these stocks
# are declining SMOOTHLY (low daily return variance), then inv_vol stays high
# despite the price decline!

print("\nKey hypothesis: Model weights 'low volatility' very heavily.")
print("These stocks decline SMOOTHLY → daily return std stays low → inv_vol stays high")
print("\nLet's verify: daily return distribution during the decline:")

for stock in TARGET_STOCKS:
    df = pl.read_parquet(DAILY_DIR / f"{stock}.parquet")
    period = df.filter(
        (pl.col("datetime") >= datetime(2026, 4, 1)) & 
        (pl.col("datetime") <= datetime(2026, 5, 28))
    ).sort("datetime")
    
    if period.height > 1:
        closes = period["close"].to_numpy()
        rets = np.diff(closes) / closes[:-1]
        
        print(f"\n{stock}:")
        print(f"  Total return: {(closes[-1]/closes[0]-1)*100:.1f}%")
        print(f"  Daily return std: {np.std(rets)*100:.2f}%")
        print(f"  Daily return mean: {np.mean(rets)*100:.3f}%")
        print(f"  Max daily loss: {np.min(rets)*100:.2f}%")
        print(f"  Max daily gain: {np.max(rets)*100:.2f}%")
        print(f"  Days down: {np.sum(rets < 0)}/{len(rets)}")
        print(f"  Avg absolute daily return: {np.mean(np.abs(rets))*100:.2f}%")

# Compare to a random sample of stocks
print("\n\nComparison: average daily return std for ALL stocks in same period:")
all_stds = []
for sym in all_symbols[:500]:  # Sample 500
    try:
        df = pl.read_parquet(DAILY_DIR / f"{sym}.parquet")
        period = df.filter(
            (pl.col("datetime") >= datetime(2026, 4, 1)) & 
            (pl.col("datetime") <= datetime(2026, 5, 28))
        ).sort("datetime")
        if period.height > 10:
            closes = period["close"].to_numpy()
            rets = np.diff(closes) / closes[:-1]
            all_stds.append(np.std(rets))
    except:
        pass

if all_stds:
    all_stds = np.array(all_stds)
    target_stds = []
    for stock in TARGET_STOCKS:
        df = pl.read_parquet(DAILY_DIR / f"{stock}.parquet")
        period = df.filter(
            (pl.col("datetime") >= datetime(2026, 4, 1)) & 
            (pl.col("datetime") <= datetime(2026, 5, 28))
        ).sort("datetime")
        if period.height > 1:
            closes = period["close"].to_numpy()
            rets = np.diff(closes) / closes[:-1]
            target_stds.append(np.std(rets))
    
    print(f"  Cross-section daily std: mean={np.mean(all_stds)*100:.2f}%, median={np.median(all_stds)*100:.2f}%")
    print(f"  Target stocks daily std: {[f'{s*100:.2f}%' for s in target_stds]}")
    print(f"  Target stocks percentile in volatility distribution: ", end="")
    for ts in target_stds:
        pct = np.sum(all_stds <= ts) / len(all_stds)
        print(f"{pct:.2f} ", end="")
    print()


print("\n\nDONE.")
