import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from vnpy.event import EventEngine
from vnpy.trader.engine import MainEngine
from vnpy.trader.utility import load_json, get_folder_path
from vnpy.trader.object import AccountData, PositionData, OrderData, TradeData
from vnpy.trader.constant import Direction, Status, Exchange, Product, OptionType, OrderType, Offset

# Try to import ToraStockGateway, handle if missing (though it should be there)
try:
    from vnpy_tora import ToraStockGateway
except ImportError:
    ToraStockGateway = None
    print("Warning: vnpy_tora not found. TradeService will not work properly.")

# Try to import PortfolioStrategyApp
try:
    from vnpy_portfoliostrategy import PortfolioStrategyApp
    from vnpy_portfoliostrategy.engine import StrategyEngine
except ImportError:
    PortfolioStrategyApp = None
    StrategyEngine = None
    print("Warning: vnpy_portfoliostrategy not found.")

CONNECT_FILE = "connect_torastock.json"
GATEWAY_NAME = "TORASTOCK"

class TradeService:
    def __init__(self):
        self.event_engine = EventEngine()
        self.event_engine.register("eLog", self.process_log_event)
        self.main_engine = MainEngine(self.event_engine)
        
        if ToraStockGateway:
            self.main_engine.add_gateway(ToraStockGateway)
            
        self.strategy_engine: Optional[StrategyEngine] = None
        if PortfolioStrategyApp:
            self.strategy_engine = self.main_engine.add_app(PortfolioStrategyApp)
            
        self.gateway_name = GATEWAY_NAME
        self._connected = False

    def process_log_event(self, event):
        print(f"[Gateway Log] {event.data.msg}")
        
    def get_strategy_engine(self) -> Optional[StrategyEngine]:
        return self.strategy_engine

    def reset_connection(self) -> Dict[str, Any]:
        """
        Disconnect, clear all data, and re-initialize the engine.
        """
        if self.main_engine:
            self.main_engine.close()
        
        # Re-initialize everything
        self.event_engine = EventEngine()
        self.event_engine.register("eLog", self.process_log_event)
        self.main_engine = MainEngine(self.event_engine)
        if ToraStockGateway:
            self.main_engine.add_gateway(ToraStockGateway)
            
        self.strategy_engine = None
        if PortfolioStrategyApp:
            self.strategy_engine = self.main_engine.add_app(PortfolioStrategyApp)
            
        self._connected = False

        return {"status": "success", "message": "Connection reset and data cleared"}

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
        orders.sort(key=lambda x: x.datetime, reverse=True)
        return [self._to_dict(o) for o in orders]

    def get_trades(self) -> List[Dict]:
        trades = self.main_engine.get_all_trades()
        trades.sort(key=lambda x: x.datetime, reverse=True)
        return [self._to_dict(t) for t in trades]
        
    def get_active_orders(self) -> List[Dict]:
        orders = self.main_engine.get_all_active_orders()
        orders.sort(key=lambda x: x.datetime, reverse=True)
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

    def close(self):
        self.main_engine.close()
