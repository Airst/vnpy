"""
Factor Analysis: Why do 603273/603325/603209/603202 persistently rank at top?

Approach:
- Load raw price + daily_basic data for ALL stocks in the CSI 2000 pool
- Compute the most important factors manually (matching factor_calculator logic)
- Compare target stocks' factor values vs cross-sectional distribution
- Identify which factor dimensions are anomalously high for these stocks
"""
import sys
sys.path.insert(0, '/home/airst/Workspace/vnpy')

import polars as pl
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
TARGET_STOCKS = ["603273.SSE", "603325.SSE", "603209.SSE", "603202.SSE"]
# Also include consistently top-ranked non-target stocks for comparison
COMPARISON_STOCKS = ["603344.SSE", "603082.SSE", "603004.SSE", "601033.SSE", "603210.SSE"]
ALL_FOCUS = TARGET_STOCKS + COMPARISON_STOCKS

ANALYSIS_DATE = "2026-05-07"  # A date when all 4 targets are in top-5
PERIOD_START = "2026-04-01"
PERIOD_END = "2026-05-28"

DAILY_DIR = Path("/home/airst/Workspace/vnpy/core/alpha_db/daily")

# Load daily basic data
from data_manager.ts_downloader.daily_basic_manager import DailyBasicManager

print("=" * 80)
print("LOADING DATA FOR ALL STOCKS IN POOL")
print("=" * 80)

# Get all stock symbols from daily directory
all_symbols = [f.stem for f in DAILY_DIR.glob("*.parquet")]
print(f"Total stocks with daily data: {len(all_symbols)}")

# Load price data for all stocks (need ~120 days lookback for 60d factors)
lookback_start = "2025-12-01"  # ~5 months before analysis period

print(f"Loading price data from {lookback_start} to {PERIOD_END}...")
dfs = []
for sym in all_symbols:
    try:
        df = pl.read_parquet(DAILY_DIR / f"{sym}.parquet")
        df = df.filter(
            (pl.col("datetime") >= datetime(2025, 12, 1)) & 
            (pl.col("datetime") <= datetime(2026, 5, 28))
        )
        if df.height > 0:
            df = df.with_columns(pl.lit(sym).alias("vt_symbol"))
            dfs.append(df)
    except Exception:
        pass

price_df = pl.concat(dfs)
print(f"Price data: {price_df.shape[0]:,} rows, {price_df['vt_symbol'].n_unique()} stocks")

# Load daily basic data
print("Loading daily basic data...")
db_manager = DailyBasicManager()
basic_pd = db_manager.load_data(all_symbols, "20251201", "20260528")
if not basic_pd.empty:
    basic_df = pl.from_pandas(basic_pd)
    basic_df = basic_df.with_columns(pl.col("datetime").cast(pl.Datetime("us")))
    # Keep key columns
    basic_cols = ["vt_symbol", "datetime", "turnover_rate", "pe", "pb", "ps", "dv_ratio", "total_mv"]
    basic_df = basic_df.select([c for c in basic_cols if c in basic_df.columns])
    
    # Join
    price_df = price_df.join(basic_df, on=["vt_symbol", "datetime"], how="left")
    price_df = price_df.sort(["vt_symbol", "datetime"])
    # Forward fill fundamentals
    for col in ["turnover_rate", "pe", "pb", "ps", "dv_ratio", "total_mv"]:
        if col in price_df.columns:
            price_df = price_df.with_columns(
                pl.col(col).forward_fill().over("vt_symbol")
            )
    print(f"After joining basic data: {price_df.shape}")

print(f"\nFinal columns: {price_df.columns}")


# ============================================================================
# COMPUTE KEY FACTORS (matching v15_factor_calculator logic as closely as possible)
# ============================================================================
print("\n" + "=" * 80)
print("COMPUTING FACTORS")
print("=" * 80)

# Sort properly
price_df = price_df.sort(["vt_symbol", "datetime"])

# Helper: rolling computations
def add_rolling_factors(df):
    """Add key factors that the model uses, computed in polars."""
    
    # Return calculations
    df = df.with_columns([
        (pl.col("close") / pl.col("close").shift(1).over("vt_symbol") - 1).alias("ret_1d"),
        (pl.col("close") / pl.col("close").shift(5).over("vt_symbol") - 1).alias("mom_5d"),
        (pl.col("close") / pl.col("close").shift(20).over("vt_symbol") - 1).alias("mom_20d"),
        (pl.col("close") / pl.col("close").shift(60).over("vt_symbol") - 1).alias("mom_60d"),
        (pl.col("close") / pl.col("close").shift(120).over("vt_symbol") - 1).alias("mom_120d"),
    ])
    
    # MA biases
    df = df.with_columns([
        (pl.col("close") / pl.col("close").rolling_mean(5).over("vt_symbol") - 1).alias("bias_5"),
        (pl.col("close") / pl.col("close").rolling_mean(10).over("vt_symbol") - 1).alias("bias_10"),
        (pl.col("close") / pl.col("close").rolling_mean(20).over("vt_symbol") - 1).alias("bias_20"),
        (pl.col("close") / pl.col("close").rolling_mean(60).over("vt_symbol") - 1).alias("bias_60"),
    ])
    
    # Volatility
    df = df.with_columns([
        pl.col("ret_1d").rolling_std(20).over("vt_symbol").alias("volatility_20d"),
        pl.col("ret_1d").rolling_std(60).over("vt_symbol").alias("volatility_60d"),
    ])
    
    # Turnover features
    if "turnover_rate" in df.columns:
        df = df.with_columns([
            pl.col("turnover_rate").rolling_mean(5).over("vt_symbol").alias("turnover_mean_5d"),
            pl.col("turnover_rate").rolling_mean(20).over("vt_symbol").alias("turnover_mean_20d"),
        ])
    
    # Size
    if "total_mv" in df.columns:
        df = df.with_columns([
            pl.col("total_mv").log().alias("size_ln_cap"),
        ])
    
    # Valuation
    if "pe" in df.columns:
        df = df.with_columns([
            (1.0 / (pl.col("pe") + 1e-4)).alias("ep_ratio"),
        ])
    if "pb" in df.columns:
        df = df.with_columns([
            (1.0 / (pl.col("pb") + 1e-4)).alias("val_pb"),
        ])
    
    # Drawdown / Distance from highs/lows
    df = df.with_columns([
        (pl.col("close") / pl.col("close").rolling_max(20).over("vt_symbol") - 1).alias("drawdown_20d"),
        (pl.col("close") / pl.col("low").rolling_min(20).over("vt_symbol") - 1).alias("rebound_20d"),
    ])
    
    # Inverse volatility (low vol anomaly)
    df = df.with_columns([
        (1.0 / (pl.col("volatility_20d") + 1e-4)).alias("inv_vol_20"),
        (1.0 / (pl.col("volatility_60d") + 1e-4)).alias("inv_vol_60"),
    ])
    
    # Volume ratio
    df = df.with_columns([
        (pl.col("volume") / pl.col("volume").rolling_mean(20).over("vt_symbol")).alias("volume_ratio"),
    ])
    
    return df

price_df = add_rolling_factors(price_df)
print("Factors computed.")

# ============================================================================
# CROSS-SECTIONAL ANALYSIS ON SPECIFIC DATE
# ============================================================================
print("\n" + "=" * 80)
print(f"CROSS-SECTIONAL FACTOR ANALYSIS ON {ANALYSIS_DATE}")
print("=" * 80)

analysis_dt = datetime.strptime(ANALYSIS_DATE, "%Y-%m-%d")
day_df = price_df.filter(pl.col("datetime") == analysis_dt)
print(f"Stocks on {ANALYSIS_DATE}: {day_df.height}")

# Factor columns to analyze
factor_cols = ["mom_5d", "mom_20d", "mom_60d", "mom_120d", 
               "bias_5", "bias_10", "bias_20", "bias_60",
               "volatility_20d", "volatility_60d", "inv_vol_20", "inv_vol_60",
               "turnover_mean_5d", "turnover_mean_20d",
               "size_ln_cap", "ep_ratio", "val_pb",
               "drawdown_20d", "rebound_20d", "volume_ratio"]

# Filter to available cols
factor_cols = [c for c in factor_cols if c in day_df.columns]

# Compute cross-sectional z-scores
print(f"\nCross-sectional Z-scores for target stocks (factor model sees Z-scored inputs):")
print(f"{'Factor':<20}", end="")
for s in ALL_FOCUS:
    short = s.split(".")[0][-4:]
    print(f" {short:>7}", end="")
print(f" {'Median':>7} {'Mean':>7} {'Std':>7}")
print("─" * (20 + 8 * (len(ALL_FOCUS) + 3)))

factor_zscores = {}
for factor in factor_cols:
    vals = day_df[factor].to_numpy()
    valid_mask = ~np.isnan(vals) & np.isfinite(vals)
    if valid_mask.sum() < 10:
        continue
    valid_vals = vals[valid_mask]
    mean = np.mean(valid_vals)
    std = np.std(valid_vals)
    median = np.median(valid_vals)
    
    print(f"{factor:<20}", end="")
    
    stock_zscores = {}
    for s in ALL_FOCUS:
        row = day_df.filter(pl.col("vt_symbol") == s)
        if row.height > 0:
            v = row[factor][0]
            if v is not None and not np.isnan(v) and std > 0:
                z = (v - mean) / std
                z_clipped = np.clip(z, -3, 3)
                stock_zscores[s] = z_clipped
                print(f" {z_clipped:>7.2f}", end="")
            else:
                stock_zscores[s] = np.nan
                print(f" {'N/A':>7}", end="")
        else:
            stock_zscores[s] = np.nan
            print(f" {'N/A':>7}", end="")
    
    print(f" {median:>7.4f} {mean:>7.4f} {std:>7.4f}")
    factor_zscores[factor] = stock_zscores


# ============================================================================
# IDENTIFY TOP FACTORS FOR TARGET STOCKS
# ============================================================================
print("\n" + "=" * 80)
print("TOP FACTORS DRIVING TARGET STOCKS TO HIGH SCORES")
print("=" * 80)

print("\nFor each target stock, factors with |z-score| > 1.0 (extreme values):")
for stock in TARGET_STOCKS:
    short = stock.split(".")[0]
    extremes = []
    for factor, zscores in factor_zscores.items():
        z = zscores.get(stock, np.nan)
        if not np.isnan(z) and abs(z) > 1.0:
            extremes.append((factor, z))
    
    extremes.sort(key=lambda x: abs(x[1]), reverse=True)
    print(f"\n{stock}:")
    for factor, z in extremes:
        direction = "HIGH" if z > 0 else "LOW"
        print(f"  {factor:<25} z={z:>6.2f}  ({direction})")


# ============================================================================
# TIME SERIES: FACTOR EVOLUTION FOR TARGET STOCKS
# ============================================================================
print("\n" + "=" * 80)
print("FACTOR EVOLUTION OVER TIME (Cross-sectional percentile rank)")
print("=" * 80)

# For key factors, show how target stocks' percentile ranks evolve
key_factors = ["mom_5d", "mom_20d", "bias_20", "volatility_20d", "inv_vol_20", 
               "turnover_mean_20d", "size_ln_cap", "drawdown_20d"]
key_factors = [f for f in key_factors if f in price_df.columns]

period_df = price_df.filter(
    (pl.col("datetime") >= datetime(2026, 4, 1)) & 
    (pl.col("datetime") <= datetime(2026, 5, 28))
)

dates = period_df["datetime"].unique().sort().to_list()

print(f"\nShowing percentile rank (0=lowest, 1=highest) for target stocks across key factors:")
for factor in key_factors:
    print(f"\n{'─'*60}")
    print(f"Factor: {factor}")
    print(f"{'Date':<12}", end="")
    for s in TARGET_STOCKS:
        short = s.split(".")[0][-4:]
        print(f" {short:>8}", end="")
    print(f" {'CS_Med':>8}")
    
    for dt in dates[::5]:  # Every 5 trading days to keep output manageable
        day_data = period_df.filter(pl.col("datetime") == dt)
        vals = day_data[factor].drop_nulls().to_numpy()
        valid_vals = vals[np.isfinite(vals)]
        if len(valid_vals) < 10:
            continue
        
        dt_str = dt.strftime("%Y-%m-%d") if hasattr(dt, 'strftime') else str(dt)
        print(f"{dt_str:<12}", end="")
        
        for s in TARGET_STOCKS:
            row = day_data.filter(pl.col("vt_symbol") == s)
            if row.height > 0:
                v = row[factor][0]
                if v is not None and not np.isnan(v) and np.isfinite(v):
                    pct = np.sum(valid_vals <= v) / len(valid_vals)
                    print(f" {pct:>8.3f}", end="")
                else:
                    print(f" {'N/A':>8}", end="")
            else:
                print(f" {'N/A':>8}", end="")
        
        print(f" {np.median(valid_vals):>8.4f}")


# ============================================================================
# SHARED CHARACTERISTICS: What do all persistent top stocks have in common?
# ============================================================================
print("\n" + "=" * 80)
print("COMMON CHARACTERISTICS OF PERSISTENTLY TOP-RANKED STOCKS")
print("=" * 80)

# For each date, compute the average factor percentile of top-5 stocks vs rest
print("\nAverage cross-sectional percentile of TOP-5 stocks vs BOTTOM-50% across key factors:")
print(f"\n{'Factor':<25} {'Top5_pctile':>12} {'Bottom50_pctile':>15} {'Gap':>8}")
print("─" * 65)

# Use a mid-period date
mid_dt = datetime(2026, 4, 28)
day_data = period_df.filter(pl.col("datetime") == mid_dt)

if day_data.height > 0:
    # Get top-5 by cross-referencing with signal data
    signal_df = pl.read_parquet("/home/airst/Workspace/vnpy/core/alpha_db/signal/ashare_mlp_signal_v15.parquet")
    sig_day = signal_df.filter(pl.col("datetime") == mid_dt).sort("total_score", descending=True)
    
    if sig_day.height > 0:
        top5_stocks = sig_day.head(5)["vt_symbol"].to_list()
        bottom_stocks = sig_day.tail(sig_day.height // 2)["vt_symbol"].to_list()
        
        print(f"\nTop-5 stocks on {mid_dt.strftime('%Y-%m-%d')}: {top5_stocks}")
        
        for factor in key_factors:
            day_vals = day_data.select(["vt_symbol", factor]).drop_nulls()
            all_vals = day_vals[factor].to_numpy()
            valid_all = all_vals[np.isfinite(all_vals)]
            
            top5_data = day_vals.filter(pl.col("vt_symbol").is_in(top5_stocks))
            bot_data = day_vals.filter(pl.col("vt_symbol").is_in(bottom_stocks))
            
            if top5_data.height > 0 and bot_data.height > 0 and len(valid_all) > 0:
                top5_vals = top5_data[factor].to_numpy()
                bot_vals = bot_data[factor].to_numpy()
                
                # Compute percentiles
                top5_pcts = [np.sum(valid_all <= v) / len(valid_all) for v in top5_vals if np.isfinite(v)]
                bot_pcts = [np.sum(valid_all <= v) / len(valid_all) for v in bot_vals if np.isfinite(v)]
                
                if top5_pcts and bot_pcts:
                    avg_top5 = np.mean(top5_pcts)
                    avg_bot = np.mean(bot_pcts)
                    print(f"{factor:<25} {avg_top5:>12.3f} {avg_bot:>15.3f} {avg_top5-avg_bot:>8.3f}")


# ============================================================================
# RAW PRICE ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("RAW PRICE DATA FOR TARGET STOCKS (Apr-May 2026)")
print("=" * 80)

for stock in TARGET_STOCKS:
    stock_data = price_df.filter(
        (pl.col("vt_symbol") == stock) & 
        (pl.col("datetime") >= datetime(2026, 4, 1)) & 
        (pl.col("datetime") <= datetime(2026, 5, 28))
    ).sort("datetime")
    
    if stock_data.height > 0:
        closes = stock_data["close"].to_numpy()
        start_price = closes[0]
        end_price = closes[-1]
        max_price = closes.max()
        min_price = closes.min()
        total_ret = (end_price / start_price - 1) * 100
        max_dd = ((closes / np.maximum.accumulate(closes)) - 1).min() * 100
        
        print(f"\n{stock}:")
        print(f"  Price: {start_price:.2f} → {end_price:.2f} (Total: {total_ret:+.1f}%)")
        print(f"  Range: {min_price:.2f} - {max_price:.2f}")
        print(f"  Max Drawdown: {max_dd:.1f}%")
        
        # Show if turnover/volume is anomalous
        if "turnover_rate" in stock_data.columns:
            tr = stock_data["turnover_rate"].to_numpy()
            tr_valid = tr[~np.isnan(tr)]
            if len(tr_valid) > 0:
                print(f"  Avg Turnover Rate: {np.mean(tr_valid):.2f}%")
        if "total_mv" in stock_data.columns:
            mv = stock_data["total_mv"].to_numpy()
            mv_valid = mv[~np.isnan(mv)]
            if len(mv_valid) > 0:
                print(f"  Market Cap (avg): {np.mean(mv_valid)/10000:.1f} 亿")


print("\n\nDONE.")
