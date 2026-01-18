import polars as pl
import numpy as np
import torch
from datetime import datetime, timedelta

from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
from core.alpha.v7_factor_calculator import V7FactorCalculator

def test_v7_calculator():
    # Create dummy data
    n_days = 100
    n_stocks = 10
    dates = []
    symbols = []
    opens = []
    highs = []
    lows = []
    closes = []
    volumes = []
    turnovers = []
    mvs = []
    
    start_date = datetime(2023, 1, 1)
    for s in range(n_stocks):
        sym = f"S{s}"
        price = 10.0
        for d in range(n_days):
            current_date = start_date + timedelta(days=d)
            dates.append(current_date)
            symbols.append(sym)
            
            change = np.random.normal(0, 0.02)
            price *= (1 + change)
            
            o = price * (1 + np.random.normal(0, 0.01))
            c = price
            h = max(o, c) * (1 + abs(np.random.normal(0, 0.01)))
            l = min(o, c) * (1 - abs(np.random.normal(0, 0.01)))
            v = np.random.uniform(1000, 10000)
            t = v * price
            mv = 1e9 * price # Market Cap
            
            opens.append(o)
            highs.append(h)
            lows.append(l)
            closes.append(c)
            volumes.append(v)
            turnovers.append(t)
            mvs.append(mv)
            
    df = pl.DataFrame({
        "datetime": dates,
        "vt_symbol": symbols,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
        "turnover": turnovers,
        "turnover_rate": np.random.uniform(0.01, 0.05, len(dates)),
        "pe": np.random.uniform(10, 50, len(dates)),
        "pb": np.random.uniform(1, 5, len(dates)),
        "ps": np.random.uniform(1, 10, len(dates)),
        "dv_ratio": np.random.uniform(0, 0.05, len(dates)),
        "total_mv": mvs
    })
    
    # Cast datetime to proper type if needed, Polars usually prefers proper datetime objects or int64 timestamp
    # df = df.with_columns(pl.col("datetime").cast(pl.Datetime))

    calc = V7FactorCalculator()
    res = calc.calculate_features(df)
    
    print("Columns:", res.columns)
    assert "dragon_score" in res.columns
    assert "label" in res.columns
    
    # Check if dragon_score is not all NaN
    ds = res["dragon_score"].to_numpy()
    # First few might be NaN due to rolling windows (e.g. 20 days)
    # Check later values
    valid_ds = ds[~np.isnan(ds)]
    print(f"Valid Dragon Score Count: {len(valid_ds)}/{len(ds)}")
    assert len(valid_ds) > 0
    
    print("Dragon Score Sample:", valid_ds[:10])

if __name__ == "__main__":
    test_v7_calculator()
