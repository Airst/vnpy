import os
import importlib
import inspect
from datetime import datetime
from typing import List, Dict, Type
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path

from vnpy_portfoliostrategy import StrategyTemplate
from vnpy_portfoliostrategy.backtesting import BacktestingEngine
from vnpy.trader.constant import Interval
from vnpy.alpha.lab import AlphaLab
from core.selector import FundamentalSelector

STRATEGY_PATH = "core/strategies"

class CoreService:
    def __init__(self):
        self.strategies: Dict[str, Type[StrategyTemplate]] = {}
        self.selector = FundamentalSelector()
        self.lab = AlphaLab("core/alpha_db")
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
                            issubclass(obj, StrategyTemplate) and 
                            obj is not StrategyTemplate):
                            self.strategies[name] = obj
                except Exception as e:
                    print(f"Failed to load strategy from {filename}: {e}")

    def get_strategies(self) -> List[str]:
        return list(self.strategies.keys())

    def get_signals(self) -> List[str]:
        """Get list of available signals from core/alpha_db/signal directory."""
        signal_path = "core/alpha_db/signal"
        if not os.path.exists(signal_path):
            return []
            
        signals = []
        for filename in os.listdir(signal_path):
            if filename.endswith(".parquet"):
                signals.append(os.path.splitext(filename)[0])
        return sorted(signals)

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
        
        for trade in engine.active_limit_orders.values():
            trades.append({
                "date": "下个交易日",
                "symbol": trade.vt_symbol,
                "direction": trade.direction.value, # type: ignore
                "price": float(trade.price),
                "volume": float(trade.volume),
                "pnl": 0  # Portfolio engine doesn't track individual trade pnl in engine.trades
            })

        # Calculate PnL for trades (FIFO)
        # Tracks buy positions: {symbol: [{"price": price, "volume": volume}, ...]}
        position_tracker = {}
        
        for trade in trades:
            symbol = trade["symbol"]
            direction = trade["direction"]
            price = trade["price"]
            volume = trade["volume"]
            
            if symbol not in position_tracker:
                position_tracker[symbol] = []
            
            # Assuming "多" is Buy/Long and "空" is Sell/Short/Close
            if direction == "多":
                position_tracker[symbol].append({"price": price, "volume": volume})
            elif direction == "空":
                realized_pnl = 0.0
                remaining_sell_volume = volume
                
                # Match against buy queue
                while remaining_sell_volume > 0 and position_tracker[symbol]:
                    buy_trade = position_tracker[symbol][0]
                    # Determine volume to match
                    match_volume = min(remaining_sell_volume, buy_trade["volume"])
                    
                    # Calculate PnL for this chunk
                    pnl_chunk = (price - buy_trade["price"]) * match_volume
                    realized_pnl += pnl_chunk
                    
                    # Update remaining volumes
                    remaining_sell_volume -= match_volume
                    buy_trade["volume"] -= match_volume
                    
                    # Remove depleted buy trades
                    if buy_trade["volume"] <= 1e-6:
                        position_tracker[symbol].pop(0)
                
                trade["pnl"] = round(realized_pnl, 2)

        return {
            "statistics": sanitized_stats,
            "daily_data": daily_data,
            "trades": trades
        }

    def get_signals_data(self, 
                         signal_name: str, 
                         start_date: datetime, 
                         end_date: datetime, 
                         vt_symbols: List[str] = None) -> Dict:
        """
        Get signal data for plotting.
        If vt_symbols is not provided, returns top 5 stocks by signal strength on the last day.
        """
        try:
            df = self.lab.load_signal(signal_name)
            
            if df is None or df.is_empty():
                return {"error": f"No signal data found for {signal_name}"}

            # Filter by date range using Polars
            df = df.filter(
                (pl.col("datetime") >= start_date) & 
                (pl.col("datetime") <= end_date)
            )

            if df.is_empty():
                return {"series": [], "dates": []}

            # Normalize column names if needed (similar to strategy logic)
            # We want a standard 'score' column
            if "final_signal" in df.columns:
                df = df.with_columns(pl.col("final_signal").alias("score"))
            elif "total_score" in df.columns:
                df = df.with_columns(pl.col("total_score").alias("score"))
            elif "score" not in df.columns:
                # Fallback: check other columns or error
                 return {"error": "Score column not found in signal data"}

            # Determine symbols to show
            target_symbols = []
            if vt_symbols:
                target_symbols = vt_symbols
            else:
                # Find last date
                last_date = df["datetime"].max()
                # Get top 5 on last date
                last_day_df = df.filter(pl.col("datetime") == last_date)
                top_5 = last_day_df.sort("score", descending=True).head(5)
                target_symbols = top_5["vt_symbol"].to_list()
            
            if not target_symbols:
                return {"series": [], "dates": []}

            # Prepare data for frontend
            # Format: 
            # dates: [d1, d2, ...]
            # series: [ {name: symbol1, data: [v1, v2, ...]}, ... ]
            
            # Get unique sorted dates from the filtered dataframe
            dates = sorted(df["datetime"].unique().to_list())
            date_strs = [d.strftime("%Y-%m-%d") for d in dates]
            
            series = []
            
            for symbol in target_symbols:
                # Filter for this symbol
                symbol_df = df.filter(pl.col("vt_symbol") == symbol)
                
                # Create a map of date -> score
                score_map = {}
                for row in symbol_df.iter_rows(named=True):
                    d_str = row["datetime"].strftime("%Y-%m-%d")
                    score_map[d_str] = row.get("score", 0) # Use .get with alias created above or existing column
                    
                # Align with master date list, fill missing with null or 0? 
                # Better null for charts to show gaps, or 0 if appropriate. 
                # Let's use null (None) to indicate no signal.
                data_points = []
                for d_str in date_strs:
                    data_points.append(score_map.get(d_str, None))
                    
                series.append({
                    "name": symbol,
                    "data": data_points
                })
                
            return {
                "dates": date_strs,
                "series": series
            }
            
        except Exception as e:
            print(f"Error getting signal data: {e}")
            return {"error": str(e)}
