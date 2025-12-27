from datetime import datetime
import polars as pl
from vnpy.alpha.lab import AlphaLab
from vnpy.alpha.dataset import Segment, AlphaDataset
from vnpy.alpha.model import AlphaModel
from vnpy.alpha.strategy import BacktestingEngine
from vnpy.trader.constant import Interval

from experiment_config import CONFIGS
from strategy import RecStrategy
from data_loader import get_vt_symbols

def run_backtesting(config_name: str, model: AlphaModel, dataset: AlphaDataset) -> BacktestingEngine:
    cfg = CONFIGS[config_name]
    lab_path = "core/alpha_db"
    lab = AlphaLab(lab_path)
    
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
    
    strat_cfg = cfg["strategy"]
    capital = strat_cfg.get("capital", 1_000_000)
    
    test_period = cfg["dataset"]["test_period"]
    
    # Get symbols
    config_path = "core/data_download/download_config.json"
    vt_symbols = get_vt_symbols(config_path)
    
    engine.set_parameters(
        vt_symbols=vt_symbols,
        interval=Interval.DAILY,
        start=datetime.strptime(test_period[0], "%Y-%m-%d"),
        end=datetime.strptime(test_period[1], "%Y-%m-%d"),
        capital=capital,
        risk_free=0.02
    )
    
    # Get strategy class
    if strat_cfg["class_name"] == "RecStrategy":
        strategy_class = RecStrategy
    else:
        # Fallback or dynamic import
        strategy_class = RecStrategy 

    engine.add_strategy(
        strategy_class=strategy_class,
        setting=strat_cfg["setting"],
        signal_df=signal_df
    )
    
    engine.load_data()
    engine.run_backtesting()
    
    # Ensure capital is set for statistics
    engine.capital = capital
    
    engine.calculate_result()
    engine.calculate_statistics()
    
    return engine
