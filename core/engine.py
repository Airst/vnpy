import os
import importlib
import inspect
from datetime import datetime
from typing import List, Dict, Type
import numpy as np
import pandas as pd

from vnpy_portfoliostrategy.backtesting import BacktestingEngine
from vnpy.trader.constant import Interval
from core.strategies.base import RecommendationStrategy
from core.selector import FundamentalSelector

STRATEGY_PATH = "core/strategies"

class RecommendationEngine:
    def __init__(self):
        self.strategies: Dict[str, Type[RecommendationStrategy]] = {}
        self.selector = FundamentalSelector()
        self.load_strategies()

    def load_strategies(self):
        """Load all strategies from the strategies directory."""
        if not os.path.exists(STRATEGY_PATH):
            return

        for filename in os.listdir(STRATEGY_PATH):
            if filename.endswith(".py") and filename != "__init__.py":
                module_name = f"core.strategies.{filename[:-3]}"
                try:
                    module = importlib.import_module(module_name)
                    for name, obj in inspect.getmembers(module):
                        if (inspect.isclass(obj) and 
                            issubclass(obj, RecommendationStrategy) and 
                            obj is not RecommendationStrategy):
                            self.strategies[name] = obj
                except Exception as e:
                    print(f"Failed to load strategy from {filename}: {e}")

    def get_strategies(self) -> List[str]:
        return list(self.strategies.keys())

    def get_candidate_symbols(self) -> List[str]:
        return self.selector.get_candidate_symbols()

    def get_data_range(self):
        return self.selector.get_data_range()

    def run_backtest(self, 
                     strategy_name: str, 
                     start: datetime, 
                     end: datetime, 
                     interval: str = "d", 
                     capital: int = 1_000_000, 
                     rate: float = 2/10000, 
                     slippage: float = 0.002, 
                     size: int = 1, 
                     pricetick: float = 0.01, 
                     setting: dict = {},
                     vt_symbols: List[str] = None): # type: ignore
        
        if strategy_name not in self.strategies:
            raise ValueError(f"Strategy {strategy_name} not found")

        strategy_cls = self.strategies[strategy_name]
        
        if vt_symbols is None:
            symbols = self.selector.get_candidate_symbols()
        else:
            symbols = vt_symbols
        
        engine = BacktestingEngine()
        engine.set_parameters(
            vt_symbols=symbols,
            interval=Interval(interval),
            start=start,
            end=end,
            rates={s: rate for s in symbols},
            slippages={s: slippage for s in symbols},
            sizes={s: size for s in symbols},
            priceticks={s: pricetick for s in symbols},
            capital=capital
        )
        
        # Pass start/end date to strategy setting for preloading data
        strategy_setting = setting.copy()
        strategy_setting["start_date"] = start
        strategy_setting["end_date"] = end
        strategy_setting["capital"] = capital
        
        engine.add_strategy(strategy_cls, strategy_setting)
        
        engine.load_data()
        engine.run_backtesting()
        engine.calculate_result()
        stats = engine.calculate_statistics()
        
        # Convert numpy types to python types for JSON serialization
        sanitized_stats = {}
        for k, v in stats.items():
            if isinstance(v, (np.integer, np.floating)):
                if np.isnan(v) or np.isinf(v):
                    sanitized_stats[k] = 0
                else:
                    sanitized_stats[k] = v.item()
            elif isinstance(v, np.ndarray):
                sanitized_stats[k] = np.nan_to_num(v).tolist()
            else:
                sanitized_stats[k] = v

        # Extract daily data for charts
        daily_data = []
        df = engine.daily_df
        if df is not None:
            # Handle NaN/Inf in DataFrame
            df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
            for dt, row in df.iterrows():
                daily_data.append({
                    "date": dt.strftime("%Y-%m-%d"), # type: ignore
                    "balance": float(row["balance"]),
                    "drawdown": float(row["drawdown"]),
                })

        # Extract trades for table
        trades = []
        for trade in engine.trades.values():
            trades.append({
                "date": trade.datetime.strftime("%Y-%m-%d %H:%M:%S"), # type: ignore
                "symbol": trade.vt_symbol,
                "direction": trade.direction.value, # type: ignore
                "price": float(trade.price),
                "volume": float(trade.volume),
                "pnl": 0  # Portfolio engine doesn't track individual trade pnl in engine.trades
            })

        return {
            "statistics": sanitized_stats,
            "daily_data": daily_data,
            "trades": trades
        }

    def run_prediction(self, strategy_name: str, setting: dict = {}) -> List[Dict]:
        """
        Run strategy on historical data up to today for all candidate symbols and return predictions.
        """
        if strategy_name not in self.strategies:
            raise ValueError(f"Strategy {strategy_name} not found")

        strategy_cls = self.strategies[strategy_name]
        symbols = self.selector.get_candidate_symbols()
        
        # Determine range: we need enough history for indicators (e.g. 60 days)
        # In production, this should be dynamic or based on strategy requirements
        end = datetime.now()
        start = end - pd.Timedelta(days=100) # Buffer for 60-day MA
        
        engine = BacktestingEngine()
        engine.set_parameters(
            vt_symbols=symbols,
            interval=Interval.DAILY,
            start=start,
            end=end,
            rates={s: 0 for s in symbols},
            slippages={s: 0 for s in symbols},
            sizes={s: 1 for s in symbols},
            priceticks={s: 0.01 for s in symbols},
            capital=1_000_000
        )
        
        strategy_setting = setting.copy()
        strategy_setting["start_date"] = start
        strategy_setting["end_date"] = end
        
        engine.add_strategy(strategy_cls, strategy_setting)
        
        engine.load_data()
        engine.run_backtesting()
        
        # Retrieve the strategy instance
        # In vnpy_portfoliostrategy BacktestingEngine, the strategy is stored in .strategy
        strategy_instance = engine.strategy
        
        results = []
        if strategy_instance:
            for vt_symbol in symbols:
                prediction = strategy_instance.get_prediction(vt_symbol)
                # Get last price if available
                last_price = 0
                if vt_symbol in engine.history_data and engine.history_data[vt_symbol]:
                    last_price = engine.history_data[vt_symbol][-1].close_price
                
                results.append({
                    "symbol": vt_symbol,
                    "prediction": prediction,
                    "last_price": last_price,
                    "score": getattr(strategy_instance, "scores", {}).get(vt_symbol, 0)
                })
                
        return results