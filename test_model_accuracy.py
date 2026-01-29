import sys
import argparse
from datetime import datetime, timedelta
import polars as pl
from vnpy.alpha.lab import AlphaLab
from pathlib import Path

# Setup paths
project_root = Path.cwd()
ALPHA_DB_PATH = "core/alpha_db"
lab_path = project_root / ALPHA_DB_PATH

def evaluate_accuracy(signal_name, top_n_list=[5, 10]):
    print(f"Initializing AlphaLab at {lab_path}...")
    lab = AlphaLab(str(lab_path))
    
    print(f"Loading signals for {signal_name}...")
    signal_df = lab.load_signal(signal_name)
    
    if signal_df is None or signal_df.is_empty():
        print("No signals found.")
        return

    # Check columns
    print(f"Signal Columns: {signal_df.columns}")
    
    score_col = "total_score"
    if score_col not in signal_df.columns:
        if "score" in signal_df.columns:
            score_col = "score"
        elif "final_signal" in signal_df.columns:
            # If total_score is missing, final_signal is a proxy (rank-based)
            score_col = "final_signal"
        else:
            print("Cannot find score column (total_score, score, or final_signal)")
            return
            
    print(f"Using score column for ranking: {score_col}")
    
    # Get date range
    min_date = signal_df["datetime"].min()
    max_date = signal_df["datetime"].max()
    
    print(f"Signal Range: {min_date} to {max_date}")
    
    # Load Market Data (Close Price)
    symbols = signal_df["vt_symbol"].unique().to_list()
    
    start_str = min_date.strftime("%Y-%m-%d")
    end_str = max_date.strftime("%Y-%m-%d")
    
    print(f"Loading Bar Data for {len(symbols)} symbols...")
    
    # Load bars with extension to calculate future returns
    price_df = lab.load_bar_df(
        vt_symbols=symbols,
        interval="d",
        start=start_str,
        end=end_str,
        extended_days=30 # Extend to get future 5 days
    )
    
    if price_df is None or price_df.is_empty():
        print("No price data found.")
        return
        
    price_df = price_df.select(["datetime", "vt_symbol", "close"])
    
    # Calculate Future 5-Day Return
    # Sort by symbol, datetime
    price_df = price_df.sort(["vt_symbol", "datetime"])
    
    # Ret_5 = (Close_t+5 / Close_t) - 1
    # Shift -5 means looking 5 rows ahead (future)
    price_df = price_df.with_columns([
        ((pl.col("close").shift(-5).over("vt_symbol") / pl.col("close")) - 1).alias("ret_5d")
    ])
    
    # Binary Label: Rise (>0) = 1, Fall/Flat (<=0) = 0
    # Also keep raw return to calculate Avg Return of Top N
    price_df = price_df.with_columns([
        (pl.col("ret_5d") > 0).cast(pl.Int32).alias("actual_label")
    ])
    
    # Join Signal and Price
    # Ensure join keys match.
    merged_df = signal_df.join(price_df, on=["datetime", "vt_symbol"], how="inner")
    
    # Drop rows where future return is null (end of dataset)
    merged_df = merged_df.drop_nulls(subset=["actual_label"])
    
    if merged_df.is_empty():
        print("Merged dataframe is empty (no overlapping dates with future returns).")
        return

    print(f"Evaluated Days: {merged_df['datetime'].n_unique()}")
    
    # --- Statistics ---
    
    # Group by Date
    # Sort by Date, Score (desc)
    merged_df = merged_df.sort(["datetime", pl.col(score_col)], descending=[False, True])
    
    # Add rank per day
    merged_df = merged_df.with_columns([
        pl.col(score_col).rank(method="ordinal", descending=True).over("datetime").alias("daily_rank")
    ])
    
    results = []
    
    print("\n" + "="*50)
    print(f"Evaluation Report: {signal_name}")
    print(f"Metric: Next 5-Day Return > 0 (Accuracy)")
    print("="*50)

    for n in top_n_list:
        # Filter Top N
        top_n_df = merged_df.filter(pl.col("daily_rank") <= n)
        
        # Calculate daily metrics
        daily_stats = top_n_df.group_by("datetime").agg([
            (pl.col("actual_label").sum() / pl.count()).alias("accuracy"),
            (pl.col("ret_5d").mean()).alias("avg_return")
        ])
        
        # Aggregate over all days
        mean_acc = daily_stats["accuracy"].mean()
        std_acc = daily_stats["accuracy"].std()
        
        mean_ret = daily_stats["avg_return"].mean()
        
        print(f"\n[Top {n}]")
        print(f"  Accuracy (Daily Mean): {mean_acc:.2%} (Std: {std_acc:.2f})")
        print(f"  Avg 5-Day Return:      {mean_ret:.2%}")
        
        # --- Monthly Statistics ---
        print(f"\n  [Top {n} - Monthly Statistics]")
        print(f"  {'Month':<10} | {'Accuracy':<10} | {'Avg Return':<10}")
        print(f"  {'-'*36}")
        
        # Add Month Column (truncate to month start)
        daily_stats = daily_stats.with_columns([
            pl.col("datetime").dt.truncate("1mo").alias("month")
        ])
        
        monthly_stats = daily_stats.group_by("month").agg([
            pl.col("accuracy").mean(),
            pl.col("avg_return").mean()
        ]).sort("month")
        
        for row in monthly_stats.iter_rows(named=True):
            m_str = row["month"].strftime("%Y-%m")
            acc = row["accuracy"]
            ret = row["avg_return"]
            print(f"  {m_str:<10} | {acc:<10.2%} | {ret:<10.2%}")

        results.append({
            "Top N": n,
            "Accuracy": mean_acc,
            "Avg Return": mean_ret
        })
        
    print("="*50)
    
    # Also evaluate "Bottom N" (Short signal check - should have low accuracy)
    print("\n[Bottom 5 Check - Should be low]")
    # Rank ascending for bottom
    merged_df = merged_df.with_columns([
        pl.col(score_col).rank(method="ordinal", descending=False).over("datetime").alias("daily_rank_asc")
    ])
    bot_n_df = merged_df.filter(pl.col("daily_rank_asc") <= 5)
    
    daily_stats_bot = bot_n_df.group_by("datetime").agg([
        (pl.col("actual_label").sum() / pl.count()).alias("accuracy"),
        (pl.col("ret_5d").mean()).alias("avg_return")
    ])
    print(f"  Accuracy (Daily Mean): {daily_stats_bot['accuracy'].mean():.2%}")
    print(f"  Avg 5-Day Return:      {daily_stats_bot['avg_return'].mean():.2%}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--signal", default="ashare_mlp_signal_v4", help="Signal name")
    args = parser.parse_args()
    
    evaluate_accuracy(args.signal)
