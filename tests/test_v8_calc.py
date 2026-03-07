
import sys
import os
import polars as pl
import torch
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.alpha.v8_factor_calculator import V8FactorCalculator

def test_v8_calculation():
    print("Testing V8FactorCalculator (Dynamic Mapping & Financials)...")
    
    # Create dummy data
    dates = [f"2023-01-{i:02d}" for i in range(1, 30)]
    symbols = ["StockA", "StockB"]
    
    data = []
    for sym in symbols:
        for i, d in enumerate(dates):
            close = 10.0 + i * 0.1 if sym == "StockA" else 20.0 - i * 0.1
            # Add a strong move for StockA to test momentum
            if sym == "StockA" and i == 20:
                close = close * 1.10 
                
            # Give different financial values to StockA and StockB to test normalization
            roe_val = 0.20 if sym == "StockA" else 0.10
            np_yoy_val = 0.30 if sym == "StockA" else 0.10
            
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
                # Concept cols
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
                "concept_cohesion": 0.8,
                "concept_acc_5_mean": 0.01,
                "concept_rank_score_mean": 0.02,
                # Financial Indicator cols (Varying)
                "roe": roe_val,
                "roa": roe_val * 0.5,
                "roic": roe_val * 0.8,
                "gross_margin": 0.35 if sym == "StockA" else 0.25,
                "netprofit_margin": 0.12,
                "netprofit_yoy": np_yoy_val,
                "tr_yoy": np_yoy_val * 0.8,
                "eps": 0.5,
                "dt_eps": 0.48,
                "total_revenue_ps": 5.0,
                "revenue_ps": 4.8,
                "current_ratio": 1.5,
                "quick_ratio": 1.2,
                "assets_turn": 0.8,
                # Label
                "label": 0.5 
            }
            data.append(row)
            
    df = pl.DataFrame(data)
    
    calc = V8FactorCalculator()
    try:
        res = calc.calculate_features(df)
        print("Calculation successful!")
        print("Result columns count:", len(res.columns))
        
        # Check specific new factors
        check_factors = ["mom_20d", "dragon_score", "label"]
        for f in check_factors:
            if f in res.columns:
                print(f"Factor '{f}' found.")
            else:
                print(f"Error: Factor '{f}' NOT found.")
        
        # Display sample output
        print("\nSample values for StockA (last 5 days):")
        print(res.filter(pl.col("vt_symbol") == "StockA").select(["datetime", "dragon_score", "mom_20d"]).tail(5))
            
    except Exception as e:
        print(f"Calculation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_v8_calculation()
