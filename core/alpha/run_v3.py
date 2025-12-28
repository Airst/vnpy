import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
import polars as pl

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.alpha.engine import AlphaEngine
from core.alpha.factor_calculator import FactorCalculator
from mlp_signals import MLPSignals
from core.selector.selector import FundamentalSelector

def run(only_factor: bool = False):
    print("Initializing Alpha Engine for V3...")
    engine = AlphaEngine(
        factor_calculator=FactorCalculator(),
        mlp_signals=MLPSignals(),
        selector=FundamentalSelector(),
        signal_name="ashare_mlp_signal_v3",
        start_date="2020-12-28",
        end_date=datetime.now().strftime("%Y-%m-%d")
    )
    
    signal_df = engine.calculate_factors()

    engine.analyze_factor_performance(signal_df)

    if not only_factor:
        signal_df = engine.calculate_signals(signal_df)

        engine.save_factors(signal_df)

if __name__ == "__main__":
    run()

