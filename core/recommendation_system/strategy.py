
from vnpy.alpha.strategy import AlphaStrategy
from vnpy.trader.constant import Direction
from vnpy.trader.object import TradeData, BarData
import polars as pl
from collections import defaultdict


def floor_vol(vol: float) -> float:
    """Floor to nearest 100 shares"""
    return int(vol // 100) * 100


class RecStrategy(AlphaStrategy):
    """
    Stock Recommendation Strategy.
    Buys top K stocks with highest predicted return.
    """
    
    top_k: int = 5                  # Hold top 5 stocks
    cash_ratio: float = 0.95        # Use 95% of capital
    stop_loss: float = 0.10         # 10% Stop Loss threshold
    min_holding_days: int = 3       # Minimum holding days before selling
    
    # Backtesting parameters
    price_add: float = 0.5        # Slippage/Aggressive order
    fixed_capital: float = 1_000_000 # Use fixed capital for sizing

    def on_init(self) -> None:
        """Initialize strategy"""
        self.holding_days = defaultdict(int)
        self.entry_prices = defaultdict(float)
        self.write_log("RecStrategy Initialized.")

    def on_bars(self, bars: dict[str, BarData]) -> None:
        """On bar data update callback"""
        # Update holding days for positions
        for vt_symbol in list(self.pos_data.keys()):
            if self.pos_data[vt_symbol] > 0:
                self.holding_days[vt_symbol] += 1

        # 1. Get Prediction Signals
        signal_df = self.get_signal()
        
        if signal_df.is_empty():
            return

        # Sort by signal score (descending)
        signal_df = signal_df.sort("signal", descending=True)
        
        # 2. Determine Target Portfolio
        target_symbols = signal_df["vt_symbol"][:self.top_k].to_list()
        
        current_pos_symbols = [s for s, pos in self.pos_data.items() if pos > 0]
        
        # 3. Check Stop Loss
        for vt_symbol in current_pos_symbols:
            if vt_symbol in bars:
                bar = bars[vt_symbol]
                entry_price = self.entry_prices.get(vt_symbol, 0)
                if entry_price > 0 and bar.close_price < entry_price * (1 - self.stop_loss):
                    # Trigger Stop Loss (Immediate exit)
                    self.set_target(vt_symbol, 0)
                    if vt_symbol in target_symbols:
                        target_symbols.remove(vt_symbol)
        
        # 4. Calculate target position for each stock
        # Use fixed capital to avoid potential cash tracking issues in alpha engine
        total_assets = self.fixed_capital
        
        target_count = len(target_symbols)
        
        if target_count == 0:
            target_per_stock = 0
        else:
            target_per_stock = (total_assets * self.cash_ratio) / target_count

        # 5. Rebalance positions
        
        # Sell: Stocks not in target list
        for vt_symbol in current_pos_symbols:
            if vt_symbol not in target_symbols:
                # Only sell if held long enough
                if self.holding_days[vt_symbol] >= self.min_holding_days:
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

        # 6. Execute trading
        self.execute_trading(bars, self.price_add)

    def on_trade(self, trade: TradeData) -> None:
        """Update entry prices"""
        super().on_trade(trade)
        
        if trade.direction == Direction.LONG:
            # Reset holding days on buy
            self.holding_days[trade.vt_symbol] = 0
            
            # Weighted average entry price
            current_pos = self.pos_data.get(trade.vt_symbol, 0)
            prev_pos = current_pos - trade.volume
            
            if prev_pos <= 0:
                self.entry_prices[trade.vt_symbol] = trade.price
            else:
                prev_cost = prev_pos * self.entry_prices[trade.vt_symbol]
                new_cost = prev_cost + (trade.volume * trade.price)
                self.entry_prices[trade.vt_symbol] = new_cost / current_pos
                
        elif trade.direction == Direction.SHORT:
            if self.pos_data.get(trade.vt_symbol, 0) <= 0:
                self.entry_prices[trade.vt_symbol] = 0.0
                self.holding_days[trade.vt_symbol] = 0
