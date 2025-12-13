
import json
from datetime import datetime
from functools import partial
import polars as pl
import torch
from vnpy.alpha.lab import AlphaLab
from vnpy.alpha.model.models.mlp_model import MlpModel
from vnpy.alpha.dataset.processor import process_drop_na, process_robust_zscore_norm, process_fill_na
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
    test_period  = ("2024-07-01", "2025-12-12") # Adjust end date as needed

    # Check GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

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

    dataset_name = "rec_dataset"
    dataset = lab.load_dataset(dataset_name)

    if dataset:
        print(f"Loaded cached dataset: {dataset_name}")
        print("To force refresh, delete the cache file in core/alpha_db/dataset/")
    else:
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
        dataset.add_processor("infer", partial(process_fill_na, fill_value=0)) # Fill NaNs with 0 (mean) after norm
        
        dataset.prepare_data()
        dataset.process_data()
        
        print(f"Saving dataset to cache: {dataset_name}")
        lab.save_dataset(dataset_name, dataset)
    
    # 4. Train Model
    print(f"Training MLP Model on {device}...")
    
    # learn_df columns: datetime, vt_symbol, feature1, feature2, ..., label
    feature_count = len(dataset.learn_df.columns) - 3 # datetime, vt_symbol, label
    print(f"Feature count: {feature_count}")

    model = MlpModel(
        input_size=feature_count,
        hidden_sizes=(265,),
        lr=0.001,
        n_epochs=50,
        batch_size=4096,
        device=device,
        seed=42 # For reproducibility
    )
    
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
        setting={"top_k": 5, "stop_loss": 0.10},
        signal_df=signal_df
    )
    
    engine.load_data()
    engine.run_backtesting()
    
    # Ensure capital is set for statistics
    engine.capital = 1_000_000
    
    engine.calculate_result()
    engine.calculate_statistics()
    
    # 7. Visualization
    print("Displaying Chart...")
    # engine.show_chart() # Replaced with save to html
    
    if engine.daily_df is not None:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        df = engine.daily_df
        fig = make_subplots(
            rows=4,
            cols=1,
            subplot_titles=["Balance", "Drawdown", "Daily Pnl", "Pnl Distribution"],
            vertical_spacing=0.06
        )

        balance_line = go.Scatter(
            x=df["date"],
            y=df["balance"],
            mode="lines",
            name="Balance"
        )
        drawdown_scatter = go.Scatter(
            x=df["date"],
            y=df["drawdown"],
            fillcolor="red",
            fill='tozeroy',
            mode="lines",
            name="Drawdown"
        )
        pnl_bar = go.Bar(y=df["net_pnl"], name="Daily Pnl")
        pnl_histogram = go.Histogram(x=df["net_pnl"], nbinsx=100, name="Days")

        fig.add_trace(balance_line, row=1, col=1)
        fig.add_trace(drawdown_scatter, row=2, col=1)
        fig.add_trace(pnl_bar, row=3, col=1)
        fig.add_trace(pnl_histogram, row=4, col=1)

        fig.update_layout(height=1000, width=1000)
        
        filename = "backtest_chart.html"
        fig.write_html(filename)
        print(f"Chart saved to {filename}")
    else:
        print("No daily result to show.")

if __name__ == "__main__":
    run_backtest()
    