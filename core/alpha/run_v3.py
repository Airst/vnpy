import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
import polars as pl

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.alpha.engine import AlphaEngine
from core.alpha.ashare_factors_v3 import AShareFactorCalculatorV3

def run():
    print("Initializing Alpha Engine for V3...")
    engine = AlphaEngine()
    
    # Initialize V3 Calculator
    calculator = AShareFactorCalculatorV3(engine)
    
    # Calculate Factors & Predict
    # Use a shorter range for testing if needed, or full range
    start_date = "2025-09-01" 
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    print(f"Running V3 Calculator from {start_date} to {end_date}...")
    
    start_time = datetime.now()
    
    signal_df = calculator.calculate_all_factors(
        start_date=start_date,
        end_date=end_date
    )
    
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()
    
    print(f"\nCalculation completed in {elapsed:.2f} seconds.")
    
    if signal_df is not None:
        # Display Stats
        stats = signal_df.group_by("datetime").agg([
            pl.col("final_signal").mean().alias("mean_signal"),
            pl.col("final_signal").std().alias("std_signal"),
            pl.count().alias("stock_count")
        ]).sort("datetime")
        
        print(f"\nSignal Statistics:")
        print(f"Time Range: {stats['datetime'].min()} to {stats['datetime'].max()}")
        print(f"Avg Daily Stocks: {stats['stock_count'].mean():.0f}")
        print(f"Signal Mean: {stats['mean_signal'].mean():.4f}")
        print(f"Signal Std: {stats['std_signal'].mean():.4f}")
        
        # Save to CSV for inspection
        output_path = "./ashare_factor_signals_v3.csv"
        signal_df.write_csv(output_path)
        print(f"\nSignals saved to: {output_path}")
        
        # Also print last few signals
        print("\nLast 5 signals:")
        print(signal_df.tail(5))
    else:
        print("No signals generated.")

if __name__ == "__main__":
    run()
