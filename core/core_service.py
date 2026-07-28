import os
import json
import importlib
import inspect
from datetime import datetime, date, timedelta
from typing import List, Dict, Type, Optional
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path

from vnpy_portfoliostrategy import StrategyTemplate
from vnpy_portfoliostrategy.backtesting import BacktestingEngine
from vnpy.trader.constant import Interval, Direction, Offset, Status
from vnpy.trader.object import OrderData, TradeData, BarData
from vnpy.trader.utility import extract_vt_symbol
from vnpy.alpha.lab import AlphaLab
from core.selector import FundamentalSelector
from core.alpha.run_manager import RunManager

# Resolve project root
PROJECT_ROOT = Path(__file__).parent.parent
STRATEGY_PATH = PROJECT_ROOT.joinpath("core/strategies")
BACKTEST_DB_PATH = PROJECT_ROOT.joinpath("core/alpha_db/backtest")
ALPHA_DB_PATH = PROJECT_ROOT.joinpath("core/alpha_db")
SIGNAL_PATH = PROJECT_ROOT.joinpath("core/alpha_db/signal")
INDEX_PATH = PROJECT_ROOT.joinpath("core/alpha_db/index")
INDEX_NAME_MAP = {
    "000001.SH": "上证指数",
    "399001.SZ": "深证成指",
    "000300.SH": "沪深300",
}


class MOCBacktestingEngine(BacktestingEngine):
    """支持 Market-On-Close 订单的回测引擎子类"""

    def __init__(self) -> None:
        super().__init__()
        self.moc_orders: list[OrderData] = []

    def clear_data(self) -> None:
        super().clear_data()
        self.moc_orders = []

    def new_bars(self, dt) -> None:
        """重写：在 on_bars 后撮合 MOC 订单"""
        self.datetime = dt

        bars: dict[str, BarData] = {}
        for vt_symbol in self.vt_symbols:
            bar = self.history_data.get((dt, vt_symbol), None)

            if bar:
                self.bars[vt_symbol] = bar
                bars[vt_symbol] = bar
            elif vt_symbol in self.bars:
                old_bar = self.bars[vt_symbol]
                bar = BarData(
                    symbol=old_bar.symbol,
                    exchange=old_bar.exchange,
                    datetime=dt,
                    open_price=old_bar.close_price,
                    high_price=old_bar.close_price,
                    low_price=old_bar.close_price,
                    close_price=old_bar.close_price,
                    gateway_name=old_bar.gateway_name
                )
                self.bars[vt_symbol] = bar

        self.cross_limit_order()
        self.strategy.on_bars(bars)
        self._cross_moc_orders(bars)

        if self.strategy.inited:
            self.update_daily_close(self.bars, dt)

    def _cross_moc_orders(self, bars: dict[str, BarData]) -> None:
        """以当前 bar 收盘价撮合 MOC 订单"""
        for order in self.moc_orders:
            bar = bars.get(order.vt_symbol)
            if not bar or bar.volume == 0:
                order.status = Status.CANCELLED
                self.strategy.update_order(order)
                continue

            order.traded = order.volume
            order.status = Status.ALLTRADED
            self.strategy.update_order(order)

            if order.vt_orderid in self.active_limit_orders:
                self.active_limit_orders.pop(order.vt_orderid)

            self.trade_count += 1
            trade_price = bar.close_price

            trade = TradeData(
                symbol=order.symbol,
                exchange=order.exchange,
                orderid=order.orderid,
                tradeid=str(self.trade_count),
                direction=order.direction,
                offset=order.offset,
                price=trade_price,
                volume=order.volume,
                datetime=self.datetime,
                gateway_name=self.gateway_name,
            )

            self.strategy.update_trade(trade)
            self.trades[trade.vt_tradeid] = trade

        self.moc_orders.clear()

    def send_moc_order(
        self,
        strategy,
        vt_symbol: str,
        direction: Direction,
        offset: Offset,
        volume: float,
    ) -> list[str]:
        """发送 MOC 订单（当前 bar 收盘价成交）"""
        symbol, exchange = extract_vt_symbol(vt_symbol)

        self.limit_order_count += 1

        order = OrderData(
            symbol=symbol,
            exchange=exchange,
            orderid=str(self.limit_order_count),
            direction=direction,
            offset=offset,
            price=0,
            volume=volume,
            status=Status.SUBMITTING,
            datetime=self.datetime,
            gateway_name=self.gateway_name,
        )

        self.moc_orders.append(order)
        self.limit_orders[order.vt_orderid] = order

        return [order.vt_orderid]


class CoreService:
    def __init__(self):
        self.strategies: Dict[str, Type[StrategyTemplate]] = {}
        self.selector = FundamentalSelector()
        self.lab = AlphaLab(str(ALPHA_DB_PATH))
        self.run_manager = RunManager(str(ALPHA_DB_PATH))
        self.load_strategies()
        os.makedirs(BACKTEST_DB_PATH, exist_ok=True)

    def load_strategies(self):
        """Load all strategies from the strategies directory."""
        print(f"Loading strategies from: {STRATEGY_PATH}")
        if not STRATEGY_PATH.exists():
            print(f"Strategy path does not exist: {STRATEGY_PATH}")
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
                            print(f"Loaded strategy: {name}")
                except Exception as e:
                    print(f"Failed to load strategy from {filename}: {e}")

    def get_strategies(self) -> List[str]:
        return list(self.strategies.keys())

    def get_signals(self) -> List[str]:
        """Get list of available signals from core/alpha_db/signal directory."""
        if not SIGNAL_PATH.exists():
            print(f"Signal path does not exist: {SIGNAL_PATH}")
            return []
            
        signal_files = []
        for filename in os.listdir(SIGNAL_PATH):
            if filename.endswith(".parquet"):
                filepath = SIGNAL_PATH / filename
                signal_files.append((os.path.splitext(filename)[0], filepath.stat().st_mtime))
        
        # Sort by modification time ascending (latest last)
        signal_files.sort(key=lambda x: x[1])
        return [name for name, _ in signal_files]

    def get_candidate_symbols(self) -> List[str]:
        return self.selector.get_candidate_symbols()

    def search_symbols(self, keyword: str) -> List[Dict]:
        """
        Fuzzy search symbols.
        Returns list of {"value": vt_symbol, "label": "vt_symbol name"}
        """
        results = []
        
        # 1. Try StockInfoManager for name search
        try:
            from data_manager.ts_downloader.stock_info_manager import StockInfoManager
            manager = StockInfoManager()
            # This might fail if DB is not configured
            stock_data = manager.search_symbols(keyword)
            
            for item in stock_data:
                results.append({
                    "value": item["vt_symbol"],
                    "label": f"{item['vt_symbol']} {item['name']}"
                })
        except Exception as e:
            # Silence error if DB or Tushare manager is not available/configured
            pass
            
        # 2. If no results (or DB failed), and keyword is numeric, search in candidate_symbols
        # Or if we just want to augment results with local candidate symbols (which might not be in stock_basic if it's outdated)
        if not results:
             candidates = self.get_candidate_symbols()
             # Simple fuzzy match on code
             lower_kw = keyword.lower()
             
             matched = []
             for sym in candidates:
                 if lower_kw in sym.lower():
                     matched.append(sym)
                     if len(matched) >= 20:
                         break
            
             for sym in matched:
                 results.append({
                     "value": sym,
                     "label": sym # No name available
                 })
                 
        return results

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
        
        engine = MOCBacktestingEngine()
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
            elif isinstance(v, (datetime, date)):
                sanitized_stats[k] = v.strftime("%Y-%m-%d")
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

            # Calculate and log 90-day return
            print("-" * 30)
            print("90-Day Return Analysis:")
            dates = df.index.tolist()
            balances = df["balance"].tolist()
            if dates:
                start_idx = 0
                total_days = len(dates)
                while start_idx < total_days:
                    current_date = dates[start_idx]
                    target_date = current_date + timedelta(days=90)
                    end_idx = -1
                    for i in range(start_idx, total_days):
                        if dates[i] >= target_date:
                            end_idx = i
                            break
                    
                    if end_idx == -1:
                        end_idx = total_days - 1
                        
                    start_balance = balances[start_idx]
                    end_balance = balances[end_idx]
                    
                    if start_balance > 0:
                        ret = (end_balance / start_balance) - 1
                        start_str = dates[start_idx].strftime("%Y-%m-%d")
                        end_str = dates[end_idx].strftime("%Y-%m-%d")
                        print(f"[{start_str} to {end_str}]: {ret:.2%}")
                    
                    if start_idx == end_idx:
                        break
                    start_idx = end_idx
            print("-" * 30)

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
        # Tracks buy positions: {symbol: [{"idx": index, "price": price, "volume": volume}, ...]}
        position_tracker = {}
        
        for idx, trade in enumerate(trades):
            symbol = trade["symbol"]
            direction = trade["direction"]
            price = trade["price"]
            volume = trade["volume"]
            
            if symbol not in position_tracker:
                position_tracker[symbol] = []
            
            # Assuming "多" is Buy/Long and "空" is Sell/Short/Close
            if direction == "多":
                position_tracker[symbol].append({"idx": idx, "price": price, "volume": volume})
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

        # Mark uncancelled buy trades as holding
        holding_indices = set()
        for positions in position_tracker.values():
            for pos in positions:
                if pos["volume"] > 1e-6:
                    holding_indices.add(pos["idx"])
        for idx, trade in enumerate(trades):
            if idx in holding_indices:
                trade["holding"] = True

        # Compute benchmark index returns
        benchmark_data = self._compute_benchmarks(daily_data, capital)

        result = {
            "statistics": sanitized_stats,
            "daily_data": daily_data,
            "benchmarks": benchmark_data,
            "trades": trades
        }

        # Save result to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        start_str = start.strftime("%Y%m%d")
        end_str = end.strftime("%Y%m%d")
        signal_name = setting['signal_name'] if setting['signal_name'] else strategy_name
        filename = f"{signal_name}_{start_str}_{end_str}_{timestamp}.json"
        filepath = os.path.join(BACKTEST_DB_PATH, filename)
        
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=4)
            # 返回给调用方的结果携带文件名 (不写入 JSON 文件本身), 供 run manifest 登记回测引用
            result["filename"] = filename
        except Exception as e:
            print(f"Failed to save backtest result: {e}")

        # Clean up old backtest files, keep only the latest 4 per signal_name
        # (run manifest 引用的回测文件豁免, 不参与轮转删除)
        try:
            protected = self.run_manager.list_referenced_backtests()
            backtest_files = [
                f for f in os.listdir(BACKTEST_DB_PATH)
                if f.endswith(".json") and f.startswith(signal_name + "_")
            ]
            backtest_files.sort(
                key=lambda f: os.path.getmtime(os.path.join(BACKTEST_DB_PATH, f)),
                reverse=True,
            )
            for old_file in backtest_files[4:]:
                if old_file in protected:
                    continue
                os.remove(os.path.join(BACKTEST_DB_PATH, old_file))
                print(f"Deleted old backtest: {old_file}")
        except Exception as e:
            print(f"Failed to clean up old backtest files: {e}")

        return result

    def _compute_benchmarks(self, daily_data: list, capital: int) -> dict:
        """
        Compute normalized NAV benchmark returns aligned to daily_data dates.

        Reads index parquet files from INDEX_PATH, filters to the backtest date range,
        and normalizes each index to start at 1.00 (representing 1元 initial investment).
        Returns cumulative growth curve (1.00 → X.XX) so it aligns with portfolio NAV.

        Returns:
            dict mapping Chinese index name to list of {"date": str, "value": float}
            Empty dict if no index data is available.
        """
        if not INDEX_PATH.exists():
            return {}

        if not daily_data:
            return {}

        start_date_str = daily_data[0]["date"]
        end_date_str = daily_data[-1]["date"]
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

        benchmarks = {}
        for ts_code, name in INDEX_NAME_MAP.items():
            filepath = INDEX_PATH / f"{ts_code}.parquet"
            if not filepath.exists():
                continue

            try:
                df = pl.read_parquet(filepath)

                df = df.filter(
                    (pl.col("trade_date") >= start_date) &
                    (pl.col("trade_date") <= end_date)
                ).sort("trade_date")

                if df.is_empty():
                    continue

                base_close = df["close"][0]

                close_map = {}
                for row in df.iter_rows(named=True):
                    d = row["trade_date"]
                    d_str = d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)
                    close_map[d_str] = row["close"]

                series = []
                last_close = base_close
                for entry in daily_data:
                    d = entry["date"]
                    if d in close_map:
                        last_close = close_map[d]
                    nav = last_close / base_close
                    series.append({
                        "date": d,
                        "value": round(nav, 4)
                    })

                benchmarks[name] = series

            except Exception as e:
                print(f"Failed to compute benchmark for {ts_code}: {e}")
                continue

        return benchmarks

    def get_backtest_history(self) -> List[Dict]:
        """Get list of saved backtest results."""
        if not os.path.exists(BACKTEST_DB_PATH):
            return []
            
        files = []
        for filename in os.listdir(BACKTEST_DB_PATH):
            if filename.endswith(".json"):
                try:
                    # Parse filename: {signal_name}_{start}_{end}_{date}_{time}.json
                    parts = filename.replace(".json", "").split("_")
                    if len(parts) >= 4:
                        timestamp_str = parts[-2] + "_" + parts[-1]
                        end_date = parts[-3]
                        start_date = parts[-4]
                        strategy_name = "_".join(parts[:-4])
                        
                        files.append({
                            "filename": filename,
                            "strategy": strategy_name,
                            "start_date": start_date,
                            "end_date": end_date,
                            "timestamp": timestamp_str
                        })
                except Exception as e:
                    print(f"Error parsing backtest file {filename}: {e}")
                    
        # Sort by timestamp descending (yyyymmdd_hhmmss format sorts lexically)
        return sorted(files, key=lambda x: x["timestamp"], reverse=True)

    def get_backtest_result(self, filename: str) -> Dict:
        """Load a specific backtest result."""
        filepath = os.path.join(BACKTEST_DB_PATH, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Backtest result {filename} not found")
            
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # 训练轮次 (Run) 管理
    # ------------------------------------------------------------------
    def list_runs(self) -> Dict:
        """列出所有训练轮次摘要 + 当前 active run"""
        return {
            "runs": self.run_manager.list_runs(),
            "active": self.run_manager.get_active(),
        }

    def get_run_detail(self, run_id: str) -> Dict:
        """run 详情: manifest + 窗口模型清单 + 回测引用有效性"""
        detail = self.run_manager.get_run_detail(run_id)
        if detail is None:
            raise FileNotFoundError(f"Run {run_id} not found")
        # 标记回测引用文件是否仍存在
        detail["backtests"] = [
            {"filename": f, "exists": os.path.exists(os.path.join(BACKTEST_DB_PATH, f))}
            for f in detail.get("backtests", [])
        ]
        return detail

    def activate_run(self, run_id: str) -> Dict:
        """设为生产 run (信号同步到 signal/{signal_name}.parquet)"""
        self.run_manager.set_active(run_id)
        return {"active": run_id}

    def delete_run(self, run_id: str) -> Dict:
        """删除 run (禁止删除 active run)"""
        ok = self.run_manager.delete_run(run_id)
        if not ok:
            raise FileNotFoundError(f"Run {run_id} not found")
        return {"deleted": run_id}

    def _load_run_signal_scored(self, run_id: str) -> Optional[pl.DataFrame]:
        """加载 run 信号并归一化出 score 列"""
        df = self.run_manager.load_signal(run_id)
        if df is None or df.is_empty():
            return None
        if "score" not in df.columns:
            for col in ("final_signal", "total_score"):
                if col in df.columns:
                    df = df.with_columns(pl.col(col).alias("score"))
                    break
            else:
                return None
        return df

    def get_run_signal_top(self, run_id: str, date_str: str = None, n: int = 20) -> Dict:
        """某日 run 信号 Top-N 排名 (date 缺省/非交易日时取不晚于该日的最近信号日)"""
        df = self._load_run_signal_scored(run_id)
        if df is None:
            raise FileNotFoundError(f"Run {run_id} has no signal")

        dates = df["datetime"].unique().sort()
        if date_str:
            target = datetime.strptime(date_str, "%Y-%m-%d")
            valid = dates.filter(dates <= target)
            actual = valid[-1] if len(valid) else dates[0]
        else:
            actual = dates[-1]

        day_df = df.filter(pl.col("datetime") == actual).sort("score", descending=True).head(n)
        symbols = day_df["vt_symbol"].to_list()

        # 批量补股票名称 (DB 不可用时留空)
        names = {}
        try:
            from data_manager.ts_downloader.stock_info_manager import StockInfoManager
            info_df = StockInfoManager().load_data(symbols)
            if not info_df.empty:
                names = dict(zip(info_df["vt_symbol"], info_df["name"]))
        except Exception:
            pass

        return {
            "run_id": run_id,
            "date": actual.strftime("%Y-%m-%d"),
            "date_range": {
                "start": dates[0].strftime("%Y-%m-%d"),
                "end": dates[-1].strftime("%Y-%m-%d"),
            },
            "items": [
                {
                    "rank": i + 1,
                    "vt_symbol": row["vt_symbol"],
                    "name": names.get(row["vt_symbol"], ""),
                    "score": row["score"],
                }
                for i, row in enumerate(day_df.iter_rows(named=True))
            ],
        }

    def get_signals_data(self, 
                         signal_name: str, 
                         start_date: datetime, 
                         end_date: datetime, 
                         vt_symbols: List[str] = None,
                         run_id: str = None) -> Dict:
        """
        Get signal data for plotting.
        If vt_symbols is not provided, returns top 5 stocks by signal strength on the last day.
        run_id 提供时从该 run 的 signal.parquet 读取 (否则读生产信号目录)。
        """
        try:
            if run_id:
                df = self.run_manager.load_signal(run_id)
            else:
                df = self.lab.load_signal(signal_name)
            
            if df is None or df.is_empty():
                return {"error": f"No signal data found for {run_id or signal_name}"}

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
