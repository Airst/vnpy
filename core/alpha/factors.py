import polars as pl
from vnpy.alpha.dataset import (
    process_cs_norm,
    process_cs_rank_norm,
    process_fill_na,
    process_drop_na
)
from core.alpha.engine import AlphaEngine

# Try to import GPU libraries
try:
    import cudf
    import cupy as cp
    HAS_GPU = True
except ImportError:
    HAS_GPU = False


def calculate_momentum_factors(engine: AlphaEngine, start_date=None, end_date=None):
    """
    Sample alpha research workflow:
    1. Load data from AlphaLab
    2. Calculate Momentum (ROC) and Volatility
    3. Normalize
    4. Save as 'momentum_signal'
    """
    symbols = engine.selector.get_candidate_symbols()
    if not symbols:
        return

    # 1. Load Data (with some buffer for calculation)
    if not start_date:
        start_date = "2020-01-01"
    if not end_date:
        end_date = "2030-01-01" 

    df = engine.lab.load_bar_df(
        vt_symbols=symbols,
        interval="d",
        start=start_date,
        end=end_date,
        extended_days=60
    )
    
    if df is None or df.is_empty():
        print("No data loaded for factor calculation.")
        return

    print("Data loaded, calculating factors...")

    # GPU Calculation Path
    if HAS_GPU:
        print("GPU detected (cuDF). Using Graphics Card for calculation.")
        try:
            # Polars -> Arrow -> cuDF
            gdf = cudf.DataFrame.from_arrow(df.to_arrow()) # type: ignore
            
            # Sort
            gdf = gdf.sort_values(["vt_symbol", "datetime"])
            
            # Calculations using cuDF (Pandas-like API)
            # Group by symbol for time-series ops
            # Note: cuDF groupby apply/rolling can be slower than vectorized ops if not careful,
            # but basic shifts are fine.
            
            # 1. Pct Change & ROC
            # Shift is per group
            # Define a function to apply or use groupby shift
            
            # Efficient way in cuDF for simple lags without explicit loop:
            # Ensure sorted, then shift. But need to handle symbol boundaries.
            # Groupby shift is supported.
            
            gdf["close_lag_1"] = gdf.groupby("vt_symbol")["close"].shift(1)
            gdf["close_lag_20"] = gdf.groupby("vt_symbol")["close"].shift(20)
            
            gdf["pct_change"] = (gdf["close"] / gdf["close_lag_1"]) - 1
            gdf["roc_20"] = (gdf["close"] / gdf["close_lag_20"]) - 1
            
            # 2. Volatility (Rolling Std)
            # cuDF rolling is supported
            gdf["vol_20"] = gdf.groupby("vt_symbol")["pct_change"].rolling(20).std().reset_index(0, drop=True)
            
            # 3. Drop NAs
            gdf = gdf.dropna(subset=["roc_20", "vol_20"])
            
            if len(gdf) == 0:
                print("Data empty after drop na (GPU).")
                return

            # 4. Cross-sectional Normalization
            # Group by datetime
            
            # Rank Norm for ROC_20
            # (rank - 0.5) / count is approx what we want (0 centered? vnpy uses specific logic)
            # vnpy logic: ((rank / count) - 0.5) * 3.46
            
            gdf["rank"] = gdf.groupby("datetime")["roc_20"].rank(method="average")
            gdf["count"] = gdf.groupby("datetime")["roc_20"].transform("count")
            gdf["roc_20"] = ((gdf["rank"] / gdf["count"]) - 0.5) * 3.46
            
            # Z-Score for VOL_20
            # (x - mean) / std
            gdf["mean"] = gdf.groupby("datetime")["vol_20"].transform("mean")
            gdf["std"] = gdf.groupby("datetime")["vol_20"].transform("std")
            gdf["vol_20"] = (gdf["vol_20"] - gdf["mean"]) / gdf["std"]
            
            # Clip outliers (optional, usually done in robust zscore)
            gdf["vol_20"] = gdf["vol_20"].clip(-3, 3)
            
            # 5. Score
            gdf["score"] = gdf["roc_20"] - gdf["vol_20"]
            
            # Select columns
            cols = ["datetime", "vt_symbol", "score", "roc_20", "vol_20", "close"]
            gdf = gdf[cols]
            
            # Convert back to Polars
            # cuDF -> Arrow -> Polars
            signal_df = pl.from_arrow(gdf.to_arrow())
            
            engine.lab.save_signal("momentum_alpha", signal_df)
            print("Signal 'momentum_alpha' saved (GPU calculation completed).")
            return

        except Exception as e:
            print(f"GPU calculation failed: {e}. Falling back to CPU.")
            # Fallback will continue below
    else:
        print("cuDF not found. Using CPU (Polars) for calculation.")

