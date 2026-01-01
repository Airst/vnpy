import os

from vnpy_portfoliostrategy import StrategyTemplate

from vnpy.alpha.lab import AlphaLab
from vnpy.trader.object import TickData, BarData, TradeData, OrderData
from vnpy.trader.constant import Direction, Offset
from typing import Dict, List
import pandas as pd
import numpy as np
from pathlib import Path

ALPHA_DB_PATH = "core/alpha_db"

class MultiFactorStrategy(StrategyTemplate):
    """
    Multi-factor strategy driven by AlphaLab signals.
    """
    author = "System"
    
    parameters = [
        "signal_name", 
        "max_holdings",
        "capital",
        "rate",
        "sell_threshold",
        "buy_threshold"
    ]

    project_root = Path(os.getcwd())
    lab_path = project_root / ALPHA_DB_PATH
    lab = AlphaLab(str(lab_path))

    def __init__(self, portfolio_engine, strategy_name, vt_symbols, setting):
        super().__init__(portfolio_engine, strategy_name, vt_symbols, setting)

        self.signal_name = setting.get("signal_name", "ashare_multi_factor")
        self.max_holdings = setting.get("max_holdings", 5)
        self.capital = setting.get("capital", 1_000_000)
        self.sell_threshold = setting.get("sell_threshold", 1)
        self.buy_threshold = setting.get("buy_threshold", 1)
        self.stop_loss_pct = setting.get("stop_loss_pct", 0.10)
        self.trailing_stop_pct = setting.get("trailing_stop_pct", 0.15)
        self.cooldown_days = setting.get("cooldown_days", 3)
        
        self.rates = portfolio_engine.rates
        self.cash = self.capital
        
        print(f"MultiFactorStrategy initialized with lab: {self.lab_path} signal: {self.signal_name}, max_holdings: {self.max_holdings}, capital: {self.capital}, buy_threshold: {self.buy_threshold}, sell_threshold: {self.sell_threshold}, stop_loss: {self.stop_loss_pct}")
        # Signals: {date_str: {vt_symbol: score}}
        self.signal_data = {}
        self.last_scores = {}
        self.last_prices = {}
        
        # Position tracking for Stop Loss
        self.pos_entry_price = {}
        self.pos_high_price = {}
        self.cooldown_map = {} # {vt_symbol: cooldown_counter}

    def on_init(self):
        print("MultiFactorStrategy Initialized")
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
            
            # Update Entry Price (Weighted Average)
            old_pos = self.get_pos(trade.vt_symbol)
            if old_pos == 0:
                 self.pos_entry_price[trade.vt_symbol] = trade.price
                 self.pos_high_price[trade.vt_symbol] = trade.price
            else:
                 # Standard avg price calculation: (old_price * old_vol + new_price * new_vol) / total_vol
                 # Note: self.get_pos returns volume BEFORE this trade update in some engines, 
                 # but StrategyTemplate usually updates position AFTER calling update_trade?
                 # Actually, vnpy_portfoliostrategy updates pos AFTER `update_trade` callback usually.
                 # Let's assume old_pos is current holding before this trade.
                 current_avg = self.pos_entry_price.get(trade.vt_symbol, trade.price)
                 new_avg = (current_avg * old_pos + trade.price * trade.volume) / (old_pos + trade.volume)
                 self.pos_entry_price[trade.vt_symbol] = new_avg
                 # Reset high price if significantly adding? No, keep high price for trailing stop?
                 # Usually trailing stop resets on new entry or keeps high? 
                 # Let's keep high price as max(old_high, new_price)
                 old_high = self.pos_high_price.get(trade.vt_symbol, trade.price)
                 self.pos_high_price[trade.vt_symbol] = max(old_high, trade.price)

        elif trade.direction == Direction.SHORT:
            # Simulate Stamp Duty for Sells (A-share standard: 0.1%)
            # Even if the engine doesn't charge it, being conservative prevents overspending.
            stamp_duty = trade.price * trade.volume * 0.0005
            commission += stamp_duty
            self.cash += trade.price * trade.volume
            
            # If closed completely, remove from tracking
            new_pos = self.get_pos(trade.vt_symbol) - trade.volume
            if new_pos <= 0:
                if trade.vt_symbol in self.pos_entry_price:
                    del self.pos_entry_price[trade.vt_symbol]
                if trade.vt_symbol in self.pos_high_price:
                    del self.pos_high_price[trade.vt_symbol]
        else:
            return
            
        # Always deduct commission
        self.cash -= commission
        
        super().update_trade(trade)

    def load_signals(self):
        """Load pre-calculated signals from AlphaLab"""
        try:
            df = self.lab.load_signal(self.signal_name)
            
            if df is None or df.is_empty():
                print(f"No signal data found for {self.signal_name}")
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
                
            print(f"Loaded signals for {len(self.signal_data)} days")
            
        except Exception as e:
            print(f"Error loading signals: {e}")

    def on_start(self):
        print("MultiFactorStrategy Started")

    def on_stop(self):
        print("MultiFactorStrategy Stopped")

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
            
            # Update High Price for Trailing Stop
            if vt_symbol in self.pos_high_price:
                if bar.close_price > self.pos_high_price[vt_symbol]:
                    self.pos_high_price[vt_symbol] = bar.close_price
            
        
        # 2. Get Scores
        scores = self.signal_data.get(date_str, {})
        self.last_scores = scores 
        
        if not scores:
            return

        # Update Cooldowns
        expired_cooldowns = []
        for s in self.cooldown_map:
            self.cooldown_map[s] -= 1
            if self.cooldown_map[s] <= 0:
                expired_cooldowns.append(s)
        for s in expired_cooldowns:
            del self.cooldown_map[s]


        # 3. Stop Loss Logic (Priority 1)
        held_symbols = []
        stop_loss_triggered = []
        
        for vt_symbol in self.vt_symbols:
            pos = self.get_pos(vt_symbol)
            if pos > 0:
                held_symbols.append(vt_symbol)
                
                if vt_symbol not in bars:
                    continue
                
                price = bars[vt_symbol].close_price
                entry = self.pos_entry_price.get(vt_symbol, price)
                high = self.pos_high_price.get(vt_symbol, price)
                
                # Check Hard Stop
                hard_stop_price = entry * (1 - self.stop_loss_pct)
                # Check Trailing Stop
                trailing_stop_price = high * (1 - self.trailing_stop_pct)
                
                if price < hard_stop_price:
                    print(f"{date_str} {vt_symbol} HARD STOP triggered. Price: {price}, Entry: {entry} (-{(1-price/entry)*100:.1f}%)")
                    self.cooldown_map[vt_symbol] = self.cooldown_days
                    stop_loss_triggered.append(vt_symbol)
                    
                elif price < trailing_stop_price:
                    print(f"{date_str} {vt_symbol} TRAILING STOP triggered. Price: {price}, High: {high} (-{(1-price/high)*100:.1f}%)")
                    self.cooldown_map[vt_symbol] = self.cooldown_days
                    stop_loss_triggered.append(vt_symbol)

        # 4. Rank candidates
        available_symbols = list(bars.keys())
        sorted_symbols = sorted(available_symbols, key=lambda s: scores.get(s, -999), reverse=True)
        
        # 5. Generate Target Portfolio
        target_symbols = []
        for s in sorted_symbols:
            # Filter:
            # 1. Score > threshold
            # 2. Not in cooldown
            if scores.get(s, 0) > self.buy_threshold and s not in self.cooldown_map: 
                target_symbols.append(s)
            if len(target_symbols) >= self.max_holdings:
                break
        
        # 6. Execute Trading (Buy/Sell Rotation)
        held_count = len(held_symbols)
        
        # 6a. Buy (stocks in target but not held)
        # Identify valid buy candidates (in target, not held)
        buy_candidates = [s for s in target_symbols if s not in held_symbols]
        
        # Determine how many we can buy (limited by available slots)
        # Update held_count based on sells to allow rotation into new stocks
        # (Actually we haven't processed 'score-based sells' yet. Let's process Sells FIRST to free up slots/cash?)
        # Standard logic: Buy what fits, Sell what should go.
        # Ideally: Sell candidates -> Free up cash -> Buy new.
        
        # 6b. Sell based on explicit sell signal (final_signal < threshold)
        sell_candidates = list(set([s for s in held_symbols if s not in target_symbols] + stop_loss_triggered))
        
        # We process sells first to free up 'virtual' slots if we assume T+0 cash availability? 
        # A-share is T+1 selling, so cash isn't available same day usually. 
        # But 'held_count' logic matters.
        
        for vt_symbol in sell_candidates:
            # Check sell signal using relative score
            score = scores.get(vt_symbol, 0.0)
            
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
                
                print(f"{date_str}, {vt_symbol} Sell, score: {score}")
                
            else:
                # print(f"{date_str}, {vt_symbol} Held, score: {score}")
                pass

        # Now Buy
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