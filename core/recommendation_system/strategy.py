
from vnpy.alpha.strategy import AlphaStrategy
from vnpy.trader.constant import Direction
from vnpy.trader.object import TradeData, BarData
from vnpy.trader.utility import round_to
import polars as pl
from collections import defaultdict

class RecStrategy(AlphaStrategy):
    """
    Stock Recommendation Strategy.
    Buys top K stocks with highest predicted return.
    """
    
    top_k: int = 1                 # Hold top 10
    cash_ratio: float = 0.95        # Use 95% cash
    min_volume: int = 100           # 1 Lot
    stop_loss: float = 0.10         # 10% Stop Loss
    
    # Backtesting parameters
    open_rate: float = 0.0003
    close_rate: float = 0.0013
    price_add: float = 0.02         # Slippage/Aggressive order

    def on_init(self) -> None:
        self.holding_days = defaultdict(int)
        self.entry_prices = defaultdict(float)
        self.write_log("RecStrategy Initialized.")

    def on_bars(self, bars: dict[str, BarData]) -> None:
        # 1. Get Prediction Signals
        signal_df = self.get_signal()
        
        if signal_df.is_empty():
            return

        # Sort by signal score (descending)
        signal_df = signal_df.sort("signal", descending=True)
        
        # 2. Determine Target Portfolio
        # Select top K candidates
        target_symbols = signal_df["vt_symbol"][:self.top_k].to_list()
        
        current_pos_symbols = [s for s, pos in self.pos_data.items() if pos > 0]
        
        # Check Stop Loss
        for vt_symbol in current_pos_symbols:
            if vt_symbol in bars:
                bar = bars[vt_symbol]
                entry_price = self.entry_prices.get(vt_symbol, 0)
                if entry_price > 0 and bar.close_price < entry_price * (1 - self.stop_loss):
                    # Trigger Stop Loss
                    self.set_target(vt_symbol, 0)
                    if vt_symbol in target_symbols:
                        target_symbols.remove(vt_symbol)
        
        # Calculate available cash
        # Note: get_cash_available returns cash after T+1 settlement logic usually, 
        # but in simplistic backtest it might be immediate or next day.
        total_assets = self.get_portfolio_value()
        
        # Adjust target count if we removed some due to SL
        target_count = len(target_symbols)
        if target_count == 0:
            target_per_stock = 0
        else:
            target_per_stock = (total_assets * self.cash_ratio) / target_count

        # 3. Rebalance
        
        # Sell: Stocks not in target list
        for vt_symbol in current_pos_symbols:
            if vt_symbol not in target_symbols:
                if vt_symbol in bars:
                    self.set_target(vt_symbol, 0)
        
        # Buy/Adjust: Stocks in target list
        for vt_symbol in target_symbols:
            if vt_symbol not in bars:
                continue
                
            bar = bars[vt_symbol]
            if bar.close_price <= 0:
                continue
                
            target_vol = floor_vol(target_per_stock / bar.close_price)
            
            self.set_target(vt_symbol, target_vol)

        # Execute
        self.execute_trading(bars, self.price_add)

    def on_trade(self, trade: TradeData) -> None:
        """Update entry prices"""
        if trade.direction == Direction.LONG:
            # Weighted average entry price
            current_pos = self.pos_data[trade.vt_symbol]
            prev_pos = current_pos - trade.volume
            
            if prev_pos <= 0:
                self.entry_prices[trade.vt_symbol] = trade.price
            else:
                prev_cost = prev_pos * self.entry_prices[trade.vt_symbol]
                new_cost = prev_cost + (trade.volume * trade.price)
                self.entry_prices[trade.vt_symbol] = new_cost / current_pos
                
        elif trade.direction == Direction.SHORT:
            if self.pos_data[trade.vt_symbol] <= 0:
                self.entry_prices[trade.vt_symbol] = 0.0

def floor_vol(vol: float) -> float:
    """Floor to nearest 100"""
    return int(vol // 100) * 100
