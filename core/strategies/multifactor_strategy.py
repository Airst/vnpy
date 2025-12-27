from core.strategies.base import RecommendationStrategy
from vnpy.trader.object import TickData, BarData, TradeData, OrderData
from vnpy.trader.constant import Direction, Offset
from typing import Dict, List
import pandas as pd
import numpy as np
from core.alpha.engine import AlphaEngine

class MultiFactorStrategy(RecommendationStrategy):
    """
    Multi-factor strategy driven by AlphaLab signals.
    """
    author = "System"
    
    parameters = [
        "signal_name", 
        "max_holdings",
        "capital",
        "rate",
        "sell_threshold"
    ]
    
    def __init__(self, portfolio_engine, strategy_name, vt_symbols, setting):
        super().__init__(portfolio_engine, strategy_name, vt_symbols, setting)
        
        self.signal_name = setting.get("signal_name", "ashare_multi_factor")
        self.max_holdings = setting.get("max_holdings", 5)
        self.capital = setting.get("capital", 1_000_000)
        self.sell_threshold = setting.get("sell_threshold", 0.5)
        self.rates = portfolio_engine.rates
        self.cash = self.capital
        
        print(f"MultiFactorStrategy initialized with signal: {self.signal_name}, max_holdings: {self.max_holdings}, capital: {self.capital}, sell_threshold: {self.sell_threshold}")
        # Signals: {date_str: {vt_symbol: score}}
        self.signal_data = {}
        self.last_scores = {}
        self.last_prices = {}

    def on_init(self):
        self.write_log("MultiFactorStrategy Initialized")
        self.load_signals()
        # self.load_bars(10)

    def update_trade(self, trade: TradeData):
        """
        Callback of new trade data.
        """
        # Calculate commission
        raw_commission = trade.price * trade.volume * self.rates[trade.vt_symbol]
        # Apply minimum commission (e.g., 5.0 for A-shares) to be conservative
        commission = max(raw_commission, 5.0)
        
        if trade.direction == Direction.LONG:
            self.cash -= trade.price * trade.volume
        elif trade.direction == Direction.SHORT:
            # Simulate Stamp Duty for Sells (A-share standard: 0.1%)
            # Even if the engine doesn't charge it, being conservative prevents overspending.
            stamp_duty = trade.price * trade.volume * 0.0005
            commission += stamp_duty
            self.cash += trade.price * trade.volume
        else:
            return
            
        # Always deduct commission
        self.cash -= commission
        
        super().update_trade(trade)

    def load_signals(self):
        """Load pre-calculated signals from AlphaLab"""
        try:
            engine = AlphaEngine()
            df = engine.get_signal_df(self.signal_name)
            
            if df is None or df.is_empty():
                self.write_log(f"No signal data found for {self.signal_name}")
                return

            # Convert to dict for fast lookup
            # Expected cols: datetime, vt_symbol, score
            # Ensure datetime is YYYY-MM-DD string or comparable
            
            # Polars iteration
            for row in df.iter_rows(named=True):
                dt = row["datetime"] # datetime object
                if hasattr(dt, "date"):
                    dt_str = dt.strftime("%Y-%m-%d")
                else:
                    dt_str = str(dt).split(" ")[0]
                    
                symbol = row["vt_symbol"]
                
                # Check for various score column names, prioritizing final_signal
                # 'final_signal' is normalized (-3 to 3) for Ranking/Buying/Selling
                score = row.get("final_signal")
                
                if score is None:
                    # Fallback
                    score = row.get("total_score")
                    if score is None:
                        score = row.get("score")
                
                if score is None:
                    score = -999.0
                
                if dt_str not in self.signal_data:
                    self.signal_data[dt_str] = {}
                    
                self.signal_data[dt_str][symbol] = score
                
            self.write_log(f"Loaded signals for {len(self.signal_data)} days")
            
        except Exception as e:
            self.write_log(f"Error loading signals: {e}")

    def on_start(self):
        self.write_log("MultiFactorStrategy Started")

    def on_stop(self):
        self.write_log("MultiFactorStrategy Stopped")

    def on_bars(self, bars: Dict[str, BarData]):
        """
        Called when a new bar (e.g. daily close) is available for all subscribed symbols.
        """
        self.cancel_all()

        # 1. Get current date
        if not bars:
            return
            
        current_dt = list(bars.values())[0].datetime
        date_str = current_dt.strftime("%Y-%m-%d")
        available_cash = self.cash

        # Update last prices
        for vt_symbol, bar in bars.items():
            self.last_prices[vt_symbol] = bar.close_price
            
        
        # 2. Get Scores
        scores = self.signal_data.get(date_str, {})
        self.last_scores = scores 
        
        if not scores:
            return

        # 3. Rank candidates
        available_symbols = list(bars.keys())
        sorted_symbols = sorted(available_symbols, key=lambda s: scores.get(s, -999), reverse=True)
        
        # 4. Generate Target Portfolio
        target_symbols = []
        for s in sorted_symbols:
            if scores.get(s, 0) > 0: # Only positive scores
                target_symbols.append(s)
            if len(target_symbols) >= self.max_holdings:
                break
        
        # 5. Execute Trading
        
        # Count currently held stocks
        held_symbols = []
        for vt_symbol in self.vt_symbols:
            if self.get_pos(vt_symbol) > 0:
                held_symbols.append(vt_symbol)
        
        held_count = len(held_symbols)

        # 5a. Sell based on explicit sell signal (final_signal < threshold)
        for vt_symbol in held_symbols:
            # Check sell signal using relative score
            score = scores.get(vt_symbol, 0.0)
            print(f"{date_str} Symbol: {vt_symbol}, Score: {score}, Held: {self.get_pos(vt_symbol)}")
            
            # Explicit Sell Condition
            # e.g., if score < -0.5 (underperforming average)
            should_sell = score < self.sell_threshold
            
            if should_sell:
                # Need current price. If not in bars (suspended), we can't sell.
                if vt_symbol not in bars:
                    continue
                    
                price = bars[vt_symbol].close_price
                if price <= 0:
                    continue

                pos = self.get_pos(vt_symbol)
                # Sell at simulated limit price below close (ensure execution)
                self.sell(vt_symbol, price * 0.998, pos)
        
        # 5b. Buy (stocks in target but not held)
        # Identify valid buy candidates (in target, not held)
        buy_candidates = [s for s in target_symbols if s not in held_symbols]
        
        # Determine how many we can buy (limited by available slots)
        # Update held_count based on sells to allow rotation into new stocks
        num_to_buy = min(len(buy_candidates), self.max_holdings - held_count)
        
        if num_to_buy > 0 and available_cash > 0:
            # Determine target value per stock based on available cash and open slots
            # This ensures we don't overspend even if we only find 1 candidate
            # Apply a safety buffer (95%)
            target_value = (available_cash / num_to_buy) * 0.95
            
            for i in range(num_to_buy):
                vt_symbol = buy_candidates[i]
                
                # If symbol not in bars, we can't buy
                if vt_symbol not in bars:
                    continue
                    
                price = bars[vt_symbol].close_price
                if price <= 0:
                    continue
                    
                # Calculate volume: round down to nearest 100
                if target_value > 0:
                    volume = int((target_value / price) / 100) * 100
                    
                    if volume > 0:
                        # Buy at simulated limit price above close (ensure execution)
                        self.buy(vt_symbol, price * 1.0002, volume)

        self.put_event()

    def get_prediction(self, vt_symbol: str):
        # Use the most recent available scores
        score = self.last_scores.get(vt_symbol, 0)
        
        # Also try to check the very latest date in signal_data if last_scores is stale
        if not self.last_scores and self.signal_data:
            latest_date = sorted(self.signal_data.keys())[-1]
            score = self.signal_data[latest_date].get(vt_symbol, 0)

        if score > 1.0:
            return "STRONG_BUY"
        elif score > 0:
            return "BUY"
        elif score < -1.0:
            return "STRONG_SELL"
        elif score < 0:
            return "SELL"
        else:
            return "HOLD"
    