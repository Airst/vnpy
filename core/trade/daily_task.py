import time
from datetime import datetime
from typing import Dict, List, Set, Optional

from vnpy.trader.engine import MainEngine
from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.object import BarData
from vnpy.trader.database import get_database
from vnpy_portfoliostrategy.engine import StrategyEngine

from core.selector.selector import FundamentalSelector
from core.strategies.multifactor_strategy import MultiFactorStrategy

STRATEGY_NAME = "daily_multifactor"
SIGNAL_NAME = "ashare_mlp_signal_v6"

class DailyTrader:
    def __init__(self, main_engine: MainEngine, strategy_engine: StrategyEngine):
        self.main_engine = main_engine
        self.strategy_engine = strategy_engine
        self.selector = FundamentalSelector()
        
    def run(self):
        print(f"[{datetime.now()}] [DailyTrader] Starting Daily Trading Task...")
        
        if not self.strategy_engine:
            print("[DailyTrader] Error: StrategyEngine is None. Aborting.")
            return

        # Register Class
        if "MultiFactorStrategy" not in self.strategy_engine.classes:
             self.strategy_engine.classes["MultiFactorStrategy"] = MultiFactorStrategy

        # 2. Init Engine (loads data)
        #self.strategy_engine.init_engine()

        # 3. Get or Add Strategy
        if STRATEGY_NAME not in self.strategy_engine.strategies:
            print(f"[DailyTrader] Adding strategy {STRATEGY_NAME}...")
            self.strategy_engine.add_strategy(
                class_name="MultiFactorStrategy", 
                strategy_name=STRATEGY_NAME, 
                vt_symbols=[], 
                setting={
                    "signal_name": SIGNAL_NAME,
                    "max_holdings": 5,
                    "capital": 5000000
                }
            )

        strategy = self.strategy_engine.strategies[STRATEGY_NAME]
        
        if not strategy.inited:
            print(f"[DailyTrader] Initializing strategy...")
            self.strategy_engine.init_strategy(STRATEGY_NAME)
            timeout = 10
            while not strategy.inited and timeout > 0:
                time.sleep(1)
                timeout -= 1
        
        print(f"[DailyTrader] Starting strategy...")
        self.strategy_engine.start_strategy(STRATEGY_NAME)
        
        # 4. Prepare Data & Sync Trades
        date_range = self.selector.get_data_range()
        today = date_range[1]
        print(f"[{datetime.now()}] [DailyTrader] Preparing Data for {today}...")

        # Identify Symbols
        today_str = today.strftime("%Y-%m-%d")
        signal_scores = strategy.signal_data.get(today_str, {})
        candidate_symbols = list(signal_scores.keys())
        held_symbols = list(strategy.pos_entry_price.keys())
        all_symbols = set(candidate_symbols) | set(held_symbols)
        
        # Sync Trades
        print(f"[{datetime.now()}] [DailyTrader] Syncing trades...")
        all_trades = self.main_engine.get_all_trades()
        sync_count = 0
        for trade in all_trades:
            if trade.vt_symbol in all_symbols:
                strategy.update_trade(trade)
                sync_count += 1
        print(f"[{datetime.now()}] [DailyTrader] Synced {sync_count} trades.")
        
        if not all_symbols:
            print("[DailyTrader] No symbols to process.")
            return

        # Load Bars
        print(f"[{datetime.now()}] [DailyTrader] Loading bars for {len(all_symbols)} symbols...")
        database = get_database()
        bars_dict: Dict[str, BarData] = {}
        
        for vt_symbol in all_symbols:
            try:
                symbol, exchange_str = vt_symbol.split(".")
                exchange = Exchange(exchange_str)
                bars = database.load_bar_data(
                    symbol=symbol,
                    exchange=exchange,
                    interval=Interval.DAILY,
                    start=today,
                    end=today
                )
                if bars:
                    bars_dict[vt_symbol] = bars[0]
            except Exception as e:
                print(f"[DailyTrader] Error loading bar for {vt_symbol}: {e}")

        if not bars_dict:
            print("[DailyTrader] No market data found for today.")
            return

        # 5. Trigger Strategy
        print(f"[{datetime.now()}] [DailyTrader] Triggering Strategy on_bars...")
        try:
            strategy.on_bars(bars_dict)
        except Exception as e:
            print(f"[DailyTrader] Strategy execution error: {e}")
            import traceback
            traceback.print_exc()

        print(f"[{datetime.now()}] [DailyTrader] Stop strategy succeed.")

        print(f"[{datetime.now()}] [DailyTrader] Task Completed.")
