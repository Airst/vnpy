
import json
from datetime import datetime
import polars as pl
from vnpy.alpha.lab import AlphaLab
from vnpy.alpha.model.models.lasso_model import LassoModel
from vnpy.alpha.dataset.processor import process_drop_na, process_robust_zscore_norm
from vnpy.alpha.dataset import Segment
from vnpy.alpha.strategy import BacktestingEngine
from vnpy.trader.constant import Interval

from features import RecommendationDataset
from strategy import RecStrategy

def run_backtest():
    # 1. Configuration
    lab_path = "core/alpha_db"
    config_path = "core/data_download/download_config.json"
    
    # Define periods
    train_period = ("2022-12-07", "2023-12-31")
    valid_period = ("2024-01-01", "2024-06-30")
    test_period  = ("2024-07-01", "2025-12-08") # Adjust end date as needed

    lab = AlphaLab(lab_path)
    
    # 2. Load Data
    # Read symbols from config
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
        
    vt_symbols = []
    for task in config.get("downloads", []):
        s = task["symbol"]
        e = task["exchange"]
        # Ensure format symbol.exchange
        if "." in s:
            code, suffix = s.split(".")
            vt_symbols.append(f"{code}.{e}")
        else:
            vt_symbols.append(f"{s}.{e}")

    print(f"Loading data for {len(vt_symbols)} symbols...")
    
    # Load bar dataframe
    # Need to load enough data to cover all periods
    # Start from train start
    df = lab.load_bar_df(
        vt_symbols=vt_symbols,
        interval=Interval.DAILY,
        start=train_period[0],
        end=test_period[1],
        extended_days=60 # For calculation window
    )
    
    if df is None or df.is_empty():
        print("No data loaded. Please run ingest_data.py first.")
        return

    # 3. Prepare Dataset
    print("Preparing dataset...")
    dataset = RecommendationDataset(
        df=df,
        train_period=train_period,
        valid_period=valid_period,
        test_period=test_period
    )
    
    # Add processors
    dataset.add_processor("learn", process_drop_na)
    dataset.add_processor("learn", process_robust_zscore_norm)
    dataset.add_processor("infer", process_robust_zscore_norm)
    
    dataset.prepare_data()
    dataset.process_data()
    
    # 4. Train Model
    print("Training Lasso Model...")
    model = LassoModel(alpha=0.0001)
    model.fit(dataset)
    model.detail()
    
    # 5. Predict
    print("Generating Signals...")
    # Predict on Test period
    segment = Segment.TEST
    pred = model.predict(dataset, segment)
    
    # Construct signal DataFrame
    infer_df = dataset.fetch_infer(segment)
    signal_df = infer_df.select(["datetime", "vt_symbol"]).with_columns(
        pl.Series(name="signal", values=pred)
    )
    
    # 6. Backtest
    print("Running Backtest...")
    engine = BacktestingEngine(lab)
    
    engine.set_parameters(
        vt_symbols=vt_symbols,
        interval=Interval.DAILY,
        start=datetime.strptime(test_period[0], "%Y-%m-%d"),
        end=datetime.strptime(test_period[1], "%Y-%m-%d"),
        capital=1_000_000,
        risk_free=0.02
    )
    
    engine.add_strategy(
        strategy_class=RecStrategy,
        setting={"top_k": 1},
        signal_df=signal_df
    )
    
    engine.load_data()
    engine.run_backtesting()
    engine.calculate_result()
    engine.calculate_statistics()
    
    # 7. Visualization
    print("Displaying Chart...")
    engine.show_chart()

if __name__ == "__main__":
    run_backtest()
