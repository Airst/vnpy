
import sys
import os
import polars as pl
import torch
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.alpha.v8_factor_calculator import V8FactorCalculator

def test_v8_calculation():
    print("Testing V8FactorCalculator...")
    
    # Create dummy data
    dates = [f"2023-01-{i:02d}" for i in range(1, 30)]
    symbols = ["StockA", "StockB"]
    
    data = []
    for sym in symbols:
        for i, d in enumerate(dates):
            close = 10.0 + i * 0.1 if sym == "StockA" else 20.0 - i * 0.1
            # Add a limit up for StockA
            if sym == "StockA" and i == 20:
                close = close * 1.10 # Limit up
                
            row = {
                "vt_symbol": sym,
                "datetime": d,
                "open": close * 0.99,
                "high": close * 1.05,
                "low": close * 0.95,
                "close": close,
                "volume": 1000.0,
                "turnover": 10000.0 * close,
                "turnover_rate": 0.01,
                "pe": 20.0,
                "pb": 2.0,
                "ps": 3.0,
                "dv_ratio": 0.02,
                "total_mv": 1e9,
                "industry": "Tech",
                # Concept cols (need to exist as V8 expects them or adds them)
                "concept_mom_5d": 0.01,
                "concept_mom_10d": 0.02,
                "concept_mom_20d": 0.03,
                "concept_mom_20d_max": 0.05,
                "concept_mom_20d_min": -0.01,
                "concept_mom_20d_std": 0.01,
                "concept_turnover_20d": 100.0,
                "concept_vol_20d": 0.02,
                "concept_count": 5,
                "concept_daily_ret": 0.01,
                "concept_hot_ratio": 0.5,
                "concept_top3_mean": 0.04,
                "concept_cohesion": 0.8
            }
            data.append(row)
            
    df = pl.DataFrame(data)
    
    calc = V8FactorCalculator()
    try:
        res = calc.calculate_features(df)
        print("Calculation successful!")
        print("Result columns:", res.columns)
        
        # Check specific new columns
        if "zt_count_20d" in res.columns:
            print("zt_count_20d found.")
            # Check values
            zt_vals = res.select(["vt_symbol", "zt_count_20d", "dragon_score"]).filter(pl.col("vt_symbol") == "StockA").tail(5)
            print(zt_vals)
        else:
            print("Error: zt_count_20d NOT found.")
            
    except Exception as e:
        print(f"Calculation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_v8_calculation()
