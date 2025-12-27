from vnpy_portfoliostrategy import StrategyTemplate
from vnpy.trader.object import BarData, TickData, TradeData
from typing import Dict, List

class RecommendationStrategy(StrategyTemplate):
    author = "System"
    
    def __init__(self, portfolio_engine, strategy_name, vt_symbols, setting):
        super().__init__(portfolio_engine, strategy_name, vt_symbols, setting)
        self.predictions = {} 

    def on_init(self):
        self.write_log("Strategy Initialized")
        self.load_bars(100)

    def on_start(self):
        self.write_log("Strategy Started")

    def on_stop(self):
        self.write_log("Strategy Stopped")

    def on_tick(self, tick: TickData):
        pass

    def on_bars(self, bars: Dict[str, BarData]):
        pass

    def get_prediction(self, vt_symbol: str):
        return self.predictions.get(vt_symbol, "HOLD")
