import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import time

from vnpy_tora import ToraStockGateway
from vnpy_portfoliostrategy import PortfolioStrategyApp
from vnpy_portfoliostrategy.engine import StrategyEngine
from vnpy_portfoliostrategy.template import StrategyTemplate

from vnpy.event import EventEngine
from vnpy.trader.engine import MainEngine
from vnpy.trader.utility import load_json, get_folder_path
from vnpy.trader.object import AccountData, PositionData, OrderData, TradeData, BarData
from vnpy.trader.constant import Direction, Status, Exchange, Interval, Product, OptionType, OrderType, Offset
from vnpy.trader.database import get_database
from core.strategies.multifactor_strategy import MultiFactorStrategy
from core.selector.selector import FundamentalSelector


CONNECT_FILE = "connect_torastock.json"
GATEWAY_NAME = "TORASTOCK"
STRATEGY_NAME = "daily_multifactor"
STRATEGY_CLASS = "MultiFactorStrategy"
SIGNAL_NAME = "ashare_mlp_signal_v6"

class TradeService:
    def __init__(self):
        self.selector = FundamentalSelector()
        self.event_engine = EventEngine()
        self.event_engine.register("eLog", self.process_log_event)
        self.main_engine = MainEngine(self.event_engine)
        
        self.main_engine.add_gateway(ToraStockGateway)
            
        self.strategy_engine: StrategyEngine
        self.strategy_engine = self.main_engine.add_app(PortfolioStrategyApp)

        self.strategy: MultiFactorStrategy = self._init_strategy()
            
        self.gateway_name = GATEWAY_NAME
        self._connected = False
    
    def _init_strategy(self) -> MultiFactorStrategy:
        print(f"[{datetime.now()}] [TradeService] Starting Daily Trading Task...")

        # Register Class
        if STRATEGY_CLASS not in self.strategy_engine.classes:
             self.strategy_engine.classes[STRATEGY_CLASS] = MultiFactorStrategy

        # 2. Init Engine (loads data)
        #self.strategy_engine.init_engine()

        # 3. Get or Add Strategy
        if STRATEGY_NAME not in self.strategy_engine.strategies:
            print(f"[TradeService] Adding strategy {STRATEGY_NAME}...")
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
            print(f"[TradeService] Initializing strategy...")
            self.strategy_engine.init_strategy(STRATEGY_NAME)
            timeout = 10
            while not strategy.inited and timeout > 0:
                time.sleep(1)
                timeout -= 1
        
        print(f"[TradeService] Starting strategy...")
        self.strategy_engine.start_strategy(STRATEGY_NAME)

        return strategy #type: ignore

    def process_log_event(self, event):
        print(f"[Gateway Log] {event.data.msg}")
        
    def get_strategy_engine(self) -> Optional[StrategyEngine]:
        return self.strategy_engine

    def reset_connection(self) -> Dict[str, Any]:
        """
        Disconnect, clear all data, and re-initialize the engine.
        """
        if self.main_engine:
            self._connected = False
            self.main_engine.close()
            print("[TradeService] wait 10s for engine close...")
            time.sleep(10)
        
        print("[TradeService] Restart engine...")
        # Re-initialize everything
        self.event_engine = EventEngine()
        self.event_engine.register("eLog", self.process_log_event)
        self.main_engine = MainEngine(self.event_engine)
        self.main_engine.add_gateway(ToraStockGateway)

        self.strategy_engine = self.main_engine.add_app(PortfolioStrategyApp)

        self._init_strategy()

        res = self.connect()
        print("[TradeService] wait 10s for gateway connect...")
        time.sleep(10)

        self._rebuild_strategy_data()
            
        self._connected = True

        return res

    def connect(self) -> Dict[str, Any]:
        """
        Connect to the gateway using configuration file.
        """
        if self._connected:
            return {"status": "already_connected", "message": "Already connected"}

        setting = load_json(CONNECT_FILE)
        if not setting:
            vntrader_path = get_folder_path(".")
            setting = load_json(str(vntrader_path.joinpath(CONNECT_FILE)))
        
        if not setting:
            # Fallback for gateways that use internal config or if file missing
            print(f"Warning: {CONNECT_FILE} not found. Attempting empty connect.")
            setting = {}

        if self.main_engine.get_gateway(self.gateway_name):
            self.main_engine.connect(setting, self.gateway_name)
            self._connected = True
            return {"status": "success", "message": f"Connected to {self.gateway_name}"}
        else:
            return {"status": "error", "message": f"Gateway {self.gateway_name} not found"}

    def _to_dict(self, obj: Any) -> Dict:
        """Helper to convert VNPY objects to dicts, handling Enums and Datetimes."""
        data = {}
        for k, v in obj.__dict__.items():
            if isinstance(v, (Direction, Status, Exchange, Product, OptionType, OrderType, Offset)):
                data[k] = v.value
            elif isinstance(v, datetime):
                data[k] = v.strftime("%Y-%m-%d %H:%M:%S")
            else:
                data[k] = v
        return data

    def get_accounts(self) -> List[Dict]:
        return [self._to_dict(a) for a in self.main_engine.get_all_accounts()]

    def get_positions(self) -> List[Dict]:
        return [self._to_dict(p) for p in self.main_engine.get_all_positions()]

    def get_orders(self) -> List[Dict]:
        # Sort by time desc
        orders = self.main_engine.get_all_orders()
        orders.sort(key=lambda x: x.datetime if x.datetime else datetime.min, reverse=True)
        return [self._to_dict(o) for o in orders]

    def get_trades(self) -> List[Dict]:
        trades = self.main_engine.get_all_trades()
        trades.sort(key=lambda x: x.datetime if x.datetime else datetime.min, reverse=True)
        return [self._to_dict(t) for t in trades]
        
    def get_active_orders(self) -> List[Dict]:
        orders = self.main_engine.get_all_active_orders()
        orders.sort(key=lambda x: x.datetime if x.datetime else datetime.min, reverse=True)
        return [self._to_dict(o) for o in orders]

    def cancel_all_orders(self) -> Dict[str, Any]:
        """
        Cancel all active orders.
        """
        active_orders = self.main_engine.get_all_active_orders()
        for order in active_orders:
            req = order.create_cancel_request()
            self.main_engine.cancel_order(req, order.gateway_name)
        
        return {"status": "success", "message": f"Sent cancel requests for {len(active_orders)} orders"}
    
    def _rebuild_strategy_data(self):
        # Clear strategy data
        for key in self.strategy.variables:
            if key == 'inited' or key == 'trading':
                continue
            setattr(MultiFactorStrategy, key, {})
        
        # Sync Trades
        print(f"[{datetime.now()}] [TradeService] Syncing trades...")
        all_trades = self.main_engine.get_all_trades()
        sync_count = 0
        for trade in all_trades:
            self.strategy.update_trade(trade)
            sync_count += 1
                
        print(f"[{datetime.now()}] [TradeService] Synced {sync_count} trades.")
    
    def run_daily_trade(self):
        # 4. Prepare Data
        self.strategy.load_signals()

        date_range = self.selector.get_data_range()
        today = date_range[1]
        print(f"[{datetime.now()}] [DailyTrader] Preparing Data for {today}...")

        # Identify Symbols
        today_str = today.strftime("%Y-%m-%d")
        signal_scores = self.strategy.signal_data.get(today_str, {})
        candidate_symbols = list(signal_scores.keys())
        held_symbols = list(self.strategy.pos_entry_price.keys())
        all_symbols = set(candidate_symbols) | set(held_symbols)
        
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
            self.strategy.on_bars(bars_dict)
        except Exception as e:
            print(f"[DailyTrader] Strategy execution error: {e}")
            import traceback
            traceback.print_exc()

        print(f"[{datetime.now()}] [DailyTrader] Stop strategy succeed.")

        print(f"[{datetime.now()}] [DailyTrader] Task Completed.")

    def close(self):
        print("[TradeService] Stop all strategies...")
        self.strategy_engine.stop_all_strategies()
        print("[TradeService] Stop main engine...")
        self.main_engine.close()
