import json
from datetime import datetime, timedelta
import polars as pl
from vnpy.alpha.lab import AlphaLab
from vnpy.alpha.dataset import Segment
from vnpy.trader.constant import Interval

from experiment_config import CONFIGS
from data_loader import get_vt_symbols, get_dataset

def predict_daily(config_name: str = "default_mlp", force_reload: bool = False) -> pl.DataFrame:
    if config_name not in CONFIGS:
        raise ValueError(f"Error: Config '{config_name}' not found.")

    cfg = CONFIGS[config_name]
    lab_path = "core/alpha_db"
    lab = AlphaLab(lab_path)
    
    # 1. Load Model
    model_name = f"{config_name}_model"
    print(f"Loading model: {model_name}...")
    try:
        model = lab.load_model(model_name)
    except FileNotFoundError:
        print(f"Model {model_name} not found. Please run backtest first to train it.")
        return pl.DataFrame()

    # 2. Load Data for Inference
    # We must load full training history to ensure normalization stats (median/IQR) are consistent with training.
    
    print("Loading history data to ensure consistent normalization...")
    config_path = "core/data_download/download_config.json"
    vt_symbols = get_vt_symbols(config_path)
            
    # Load from start of TRAINING period up to LATEST available
    train_start_str = cfg["dataset"]["train_period"][0]
    # Use today as end date for prediction
    end_date = datetime.now()
    
    # We need a temporary dataset name for prediction or reuse the training one if it covers enough?
    # Actually, we should use the same dataset name to leverage cache, but extend it to today.
    dataset_name = cfg["dataset"].get("name", "rec_dataset")
    
    # Dummy periods for inference dataset
    # We want train_period to match training config to get same stats if we re-calc?
    # Actually get_dataset handles loading and ensuring coverage.
    
    train_period = cfg["dataset"]["train_period"]
    valid_period = cfg["dataset"]["valid_period"]
    
    # Test period: Look back 7 days to ensure we get the latest trading day if today is weekend/holiday
    test_start = end_date - timedelta(days=7)
    test_period = (test_start.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    
    dataset = get_dataset(
        lab,
        dataset_name,
        vt_symbols,
        train_period,
        valid_period,
        test_period,
        force_reload=force_reload
    )
    
    # 3. Predict
    print("Predicting...")
    # The 'test' segment in dataset corresponds to the test_period passed to get_dataset?
    # Yes, get_dataset updates the periods on the dataset object.
    
    segment = Segment.TEST
    pred = model.predict(dataset, segment)
    
    # 4. Show Results
    infer_df = dataset.fetch_infer(segment)
    
    if infer_df.is_empty():
        print("No data found for prediction period.")
        return pl.DataFrame()

    # Create result dataframe
    result_df = infer_df.select(["datetime", "vt_symbol"]).with_columns(
        pl.Series(name="signal", values=pred)
    )
    
    # Filter to keep only the LATEST date for each symbol
    max_date = result_df["datetime"].max()
    result_df = result_df.filter(pl.col("datetime") == max_date)
    
    # Join with close price from original dataset
    orig_data = dataset.df.select(["datetime", "vt_symbol", "close"])
    final_df = result_df.join(orig_data, on=["datetime", "vt_symbol"], how="left")
    
    return final_df
