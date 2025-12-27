
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
    Stock Recommendation Strategy (Daily Decision).
    
    Logic:
    1. Sell: If signal < 0 OR Stop Loss hit -> Sell next day.
    2. Buy: Top N positive signals (not held) -> Buy next day using available cash.
    """
    
    buy_count: int = 1              # Max stocks to buy per day
    max_pos: int = 10               # Max total positions
    stop_loss: float = 0.10         # 10% Stop Loss threshold
    
    # Backtesting parameters
    price_add: float = 0.5          # Slippage/Aggressive order

    def on_init(self) -> None:
        """Initialize strategy"""
        self.holding_days = defaultdict(int)
        self.entry_prices = defaultdict(float)
        self.write_log("RecStrategy Initialized.")

    def on_bars(self, bars: dict[str, BarData]) -> None:
        """On bar data update callback"""
        # Update holding days for positions
        current_holdings = [s for s, pos in self.pos_data.items() if pos > 0]
        
        for vt_symbol in current_holdings:
            self.holding_days[vt_symbol] += 1

        # 1. Get Prediction Signals
        signal_df = self.get_signal()
        
        if signal_df.is_empty():
            return

        # Map signal for easy access
        signal_map = {
            row[0]: row[1] 
            for row in signal_df.select(["vt_symbol", "signal"]).iter_rows()
        }

        # 2. Sell Logic
        for vt_symbol in current_holdings:
            if vt_symbol not in bars:
                continue
                
            bar = bars[vt_symbol]
            entry_price = self.entry_prices.get(vt_symbol, 0)
            signal = signal_map.get(vt_symbol, 0)
            
            should_sell = False
            
            # Condition A: Stop Loss
            if entry_price > 0 and bar.close_price < entry_price * (1 - self.stop_loss):
                should_sell = True
            
            # Condition B: Negative Signal
            if signal < 0:
                should_sell = True
                
            if should_sell:
                self.set_target(vt_symbol, 0)

        # 3. Buy Logic
        # Select candidates: Positive signal, Not held
        candidates = []
        for vt_symbol, signal in signal_map.items():
            if vt_symbol not in current_holdings and signal > 0:
                candidates.append((vt_symbol, signal))
        
        # Sort by signal strength
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Take top N
        target_buys = candidates[:self.buy_count]
        
        # Check Max Position Limit
        current_holding_count = len(current_holdings)
        
        if current_holding_count < self.max_pos:
            slots_available = self.max_pos - current_holding_count
            actual_buys = target_buys[:slots_available]
            
            if actual_buys:
                # Dynamic Position Sizing (Full Position)
                available_cash = self.get_cash_available()
                
                # Distribute cash equally among actual buys
                capital_per_trade = available_cash / len(actual_buys)
                
                self.write_log(f"Buy Decision: Cash={available_cash:.2f}, Targets={len(actual_buys)}, PerTrade={capital_per_trade:.2f}")

                # Minimum trade capital check (prevent dust trading)
                MIN_TRADE_CAPITAL = 5000
                if capital_per_trade < MIN_TRADE_CAPITAL:
                    self.write_log(f"Skipping buys: Capital per trade {capital_per_trade:.2f} < Min {MIN_TRADE_CAPITAL}")
                else:
                    for vt_symbol, _ in actual_buys:
                        if vt_symbol not in bars:
                            continue
                        bar = bars[vt_symbol]
                        if bar.close_price <= 0:
                            continue
                        
                        # Calculate volume
                        vol = floor_vol(capital_per_trade / bar.close_price)
                        if vol > 0:
                            self.set_target(vt_symbol, vol)
                            self.write_log(f"Set Target {vt_symbol}: Vol={vol}, Price={bar.close_price}, Est.Amt={vol*bar.close_price:.2f}")

        # 4. Execute trading
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
