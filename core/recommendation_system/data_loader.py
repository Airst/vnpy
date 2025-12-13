import json
from datetime import datetime
from functools import partial
import polars as pl

from vnpy.alpha.lab import AlphaLab
from vnpy.trader.constant import Interval
from vnpy.alpha.dataset.processor import process_drop_na, process_robust_zscore_norm, process_fill_na

from features import RecommendationDataset

def get_vt_symbols(config_path: str) -> list[str]:
    """Read symbols from config"""
    with open(config_path, "r", encoding="utf-8") as f:
        dl_config = json.load(f)
        
    vt_symbols = []
    for task in dl_config.get("downloads", []):
        s = task["symbol"]
        e = task["exchange"]
        # Ensure format symbol.exchange
        if "." in s:
            code, suffix = s.split(".")
            vt_symbols.append(f"{code}.{e}")
        else:
            vt_symbols.append(f"{s}.{e}")
    return vt_symbols

def load_raw_data(lab: AlphaLab, vt_symbols: list[str], start_str: str, end_str: str) -> pl.DataFrame:
    """Load raw bar data DataFrame"""
    print(f"Loading data for {len(vt_symbols)} symbols...")
    
    # Load bar dataframe
    # Need to load enough data to cover all periods
    # Start from train start
    df = lab.load_bar_df(
        vt_symbols=vt_symbols,
        interval=Interval.DAILY,
        start=start_str,
        end=end_str,
        extended_days=60 # For calculation window
    )
    return df # type: ignore

def create_dataset(
    df: pl.DataFrame,
    train_period: tuple[str, str],
    valid_period: tuple[str, str],
    test_period: tuple[str, str]
) -> RecommendationDataset:
    """Prepare and process the RecommendationDataset"""
    print("Preparing dataset...")
    dataset = RecommendationDataset(
        df=df,
        train_period=train_period,
        valid_period=valid_period,
        test_period=test_period
    )
    
    # Add processors
    dataset.add_processor("learn", process_drop_na) # type: ignore
    dataset.add_processor("learn", process_robust_zscore_norm)  # type: ignore
    
    dataset.add_processor("infer", process_robust_zscore_norm)  # type: ignore
    dataset.add_processor("infer", partial(process_fill_na, fill_value=0)) #type: ignore  Fill NaNs with 0 (mean) after norm for inference
    
    dataset.prepare_data()
    dataset.process_data()
    
    return dataset
