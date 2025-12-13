import json
from datetime import datetime
import polars as pl
from vnpy.alpha.lab import AlphaLab
from vnpy.alpha.dataset import Segment
from vnpy.trader.constant import Interval

from experiment_config import CONFIGS
from data_loader import get_vt_symbols, load_raw_data, create_dataset

def predict_daily(config_name: str = "default_mlp") -> pl.DataFrame:
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
    end_date = datetime.now()
    
    df_full = load_raw_data(lab, vt_symbols, train_start_str, end_date.strftime("%Y-%m-%d")) # Actually load_raw_data expects YYYY-MM-DD usually, checking implementation...
    # load_raw_data uses lab.load_bar_df which takes str or datetime.
    # In data_loader.py I used start_str directly.
    # In run_prediction.py orig code: datetime.strptime(train_start_str, "%Y-%m-%d")
    # AlphaLab.load_bar_df documentation is not visible but typically handles both.
    # Let's check data_loader.py again.
    # It passes start=start_str directly.
    
    # In run_prediction.py, it was:
    # df_full = lab.load_bar_df(..., start=datetime.strptime(train_start_str, "%Y-%m-%d"), end=end_date, ...)
    # My data_loader.py wrapper takes string.
    # I should pass string.
    
    if df_full is None or df_full.is_empty():
        print("No data loaded.")
        return pl.DataFrame()
        
    latest_date = df_full["datetime"].max()
    print(f"Latest data date: {latest_date}")
    
    # Define Periods
    # Train: Same as config
    # Test: Only the latest date
    
    test_start = latest_date
    test_end = latest_date
    
    # Re-use dataset creation logic
    dataset = create_dataset(
        df=df_full,
        train_period=cfg["dataset"]["train_period"],
        valid_period=cfg["dataset"]["valid_period"],
        test_period=(test_start.strftime("%Y-%m-%d"), test_end.strftime("%Y-%m-%d"))
    )
    
    # 3. Predict
    print("Predicting...")
    segment = Segment.TEST
    pred = model.predict(dataset, segment)
    
    # 4. Show Results
    infer_df = dataset.fetch_infer(segment)
    
    # Create result dataframe
    result_df = infer_df.select(["datetime", "vt_symbol"]).with_columns(
        pl.Series(name="signal", values=pred)
    )
    
    # Join with close price from original dataset
    orig_data = dataset.df.select(["datetime", "vt_symbol", "close"])
    final_df = result_df.join(orig_data, on=["datetime", "vt_symbol"], how="left")
    
    return final_df
