"""
多因子组合策略 — 基于 AlphaLab 信号的交易执行层

== 当前状态 ==
框架: vnpy_portfoliostrategy (StrategyTemplate)
信号源: AlphaLab parquet 信号文件（由 mlp_signals.py 生成）
风控集成: RiskController（组合级回撤熔断 + 波动率缩仓）
执行特性: 日频调仓，个股止损，追踪止损，冷却期

== 设计决策 ==
- 信号驱动而非规则驱动: 策略本身不做选股判断，完全依赖模型信号排名
- 分层风控: 个股级(止损/追踪止损) + 组合级(RiskController)
- 冷却期: 个股止损后设置冷却天数，避免反复开平
- 持仓数由 RiskController 动态决定（base=5，根据回撤/波动减少）
"""
import os
from datetime import datetime

from vnpy_portfoliostrategy import StrategyTemplate

from vnpy.alpha.lab import AlphaLab
from vnpy.trader.object import TickData, BarData, TradeData, OrderData
from vnpy.trader.constant import Direction, Offset
from typing import Dict, List
import pandas as pd
import numpy as np
from pathlib import Path

from core.risk_controller import RiskController

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

    variables = [
        "cash",
        "pos_entry_price",
        "pos_high_price",
        "cooldown_map",
        "pending_sell"
    ]

    project_root = Path(__file__).parent.parent.parent
    lab_path = project_root / ALPHA_DB_PATH
    lab = AlphaLab(str(lab_path))

    def __init__(self, portfolio_engine, strategy_name, vt_symbols, setting):
        super().__init__(portfolio_engine, strategy_name, vt_symbols, setting)

        self.signal_name = setting.get("signal_name", "ashare_multi_factor")
        self.max_holdings = int(setting.get("max_holdings", 5))
        self.capital = float(setting.get("capital", 1_000_000))
        self.sell_threshold = float(setting.get("sell_threshold", 1.54))
        self.buy_threshold = float(setting.get("buy_threshold", 1))
        self.stop_loss_pct = float(setting.get("stop_loss_pct", 0.05))
        self.trailing_stop_pct = float(setting.get("trailing_stop_pct", 0.15))
        self.cooldown_days = int(setting.get("cooldown_days", 3))
        self.persistence_days = int(setting.get("persistence_days", 3))
        
        self.rate = float(setting.get("rate", 0.0003))
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
        self.pending_sell = {} # {vt_symbol: days_remaining}

        # Portfolio-level risk controller
        self.risk_control_enabled = bool(setting.get("risk_control_enabled", True))
        self.risk_controller = RiskController(
            base_max_holdings=self.max_holdings,
            enabled=self.risk_control_enabled,
        )

    def on_init(self):
        print("MultiFactorStrategy Initialized")
        self.load_signals()
        # self.load_bars(10)

    def update_trade(self, trade: TradeData):
        """
        Callback of new trade data.
        """
        # Calculate commission
        raw_commission = trade.price * trade.volume * self.rate
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
            
            # Clear any pending sell status on new buy
            if trade.vt_symbol in self.pending_sell:
                del self.pending_sell[trade.vt_symbol]

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
                if trade.vt_symbol in self.pending_sell:
                    del self.pending_sell[trade.vt_symbol]
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

        # Update last prices & trailing high
        for vt_symbol, bar in bars.items():
            self.last_prices[vt_symbol] = bar.close_price
            
            if vt_symbol in self.pos_high_price:
                if bar.high_price > self.pos_high_price[vt_symbol]:
                    self.pos_high_price[vt_symbol] = bar.high_price

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
            
        # Update Pending Sells
        expired_pending = []
        for s in self.pending_sell:
            self.pending_sell[s] -= 1
            if self.pending_sell[s] <= 0:
                expired_pending.append(s)
        for s in expired_pending:
            del self.pending_sell[s]

        # 3. Compute portfolio equity and call risk controller
        held_symbols = []
        for vt_symbol in self.vt_symbols:
            if self.get_pos(vt_symbol) > 0:
                held_symbols.append(vt_symbol)

        portfolio_equity = self.cash
        for vt_symbol in held_symbols:
            pos = self.get_pos(vt_symbol)
            price = self.last_prices.get(vt_symbol, self.pos_entry_price.get(vt_symbol, 0))
            portfolio_equity += pos * price

        dynamic_max, force_sell_symbols = self.risk_controller.on_bar(
            portfolio_equity=portfolio_equity,
            current_positions=held_symbols,
            signal_scores=scores,
        )

        # 4. Execute force sells (highest priority)
        for vt_symbol in force_sell_symbols:
            if vt_symbol not in bars:
                continue
            bar = bars[vt_symbol]
            price = bar.close_price
            if price <= 0:
                continue
            pos = self.get_pos(vt_symbol)
            if pos <= 0:
                continue

            daily_range = bar.high_price - bar.low_price
            if daily_range == 0:
                daily_range = price * 0.02
            limit_price = max(price - daily_range, price * 0.95)

            self.sell(vt_symbol, limit_price, pos)
            self.cooldown_map[vt_symbol] = self.cooldown_days
            print(f"{date_str}, {vt_symbol} Sell (RISK_CTRL max={dynamic_max}), limit_price:{limit_price:.2f} (Close:{price}) score: {scores.get(vt_symbol, 0)}")

        # Remove force-sold symbols from held list for subsequent logic
        force_sell_set = set(force_sell_symbols)
        held_symbols = [s for s in held_symbols if s not in force_sell_set]

        # 5. Stop Loss Logic (Priority 2)
        stop_loss_triggered = []
        
        for vt_symbol in held_symbols:
            if vt_symbol not in bars:
                continue
                
            price = bars[vt_symbol].close_price
            entry = self.pos_entry_price.get(vt_symbol, price)
            high = self.pos_high_price.get(vt_symbol, price)
            
            hard_stop_price = entry * (1 - self.stop_loss_pct)
            trailing_stop_price = high * (1 - self.trailing_stop_pct)
            
            if price < hard_stop_price:
                print(f"{date_str} {vt_symbol} HARD STOP triggered. Price: {price}, Entry: {entry} (-{(1-price/entry)*100:.1f}%)")
                self.cooldown_map[vt_symbol] = self.cooldown_days
                stop_loss_triggered.append(vt_symbol)
                
            elif price < trailing_stop_price:
                print(f"{date_str} {vt_symbol} TRAILING STOP triggered. Price: {price}, High: {high} (-{(1-price/high)*100:.1f}%)")
                self.cooldown_map[vt_symbol] = self.cooldown_days
                stop_loss_triggered.append(vt_symbol)

        # 6. Rank candidates (use dynamic_max instead of self.max_holdings)
        available_symbols = list(bars.keys())
        sorted_symbols = sorted(available_symbols, key=lambda s: scores.get(s, -999), reverse=True)

        target_symbols = []
        for s in sorted_symbols:
            if scores.get(s, 0) > self.buy_threshold and s not in self.cooldown_map:
                target_symbols.append(s)
            if len(target_symbols) >= dynamic_max:
                break
        
        # 7. Sell logic
        held_count = len(held_symbols)
        sell_candidates = list(set([s for s in held_symbols if s not in target_symbols] + stop_loss_triggered))
        
        for vt_symbol in held_symbols:
            score = scores.get(vt_symbol, 0.0)
            
            if score < self.sell_threshold:
                self.pending_sell[vt_symbol] = self.persistence_days
            
            if score > (self.buy_threshold + 0.5):
                if vt_symbol in self.pending_sell:
                    del self.pending_sell[vt_symbol]
            
            is_stop_loss = vt_symbol in stop_loss_triggered
            is_persistent_sell = vt_symbol in self.pending_sell
            should_sell = (score < self.sell_threshold) or is_persistent_sell or is_stop_loss
            
            if should_sell:
                if vt_symbol not in bars:
                    continue
                    
                bar = bars[vt_symbol]
                price = bar.close_price
                if price <= 0:
                    continue

                pos = self.get_pos(vt_symbol)
                
                daily_range = bar.high_price - bar.low_price
                if daily_range == 0:
                    daily_range = price * 0.02
                limit_price = max(price - daily_range, price * 0.95)
                
                self.sell(vt_symbol, limit_price, pos)
                
                reason = "STOP_LOSS" if is_stop_loss else ("PERSIST" if is_persistent_sell else "SIGNAL")
                print(f"{date_str}, {vt_symbol} Sell ({reason}), limit_price:{limit_price:.2f} (Close:{price}) score: {score}")

        # 8. Buy logic (use dynamic_max)
        buy_candidates = [s for s in target_symbols if s not in held_symbols]
        num_to_buy = min(len(buy_candidates), dynamic_max - held_count)
        
        if num_to_buy > 0 and available_cash > 0:
            target_value = (available_cash / num_to_buy) * 0.95
            
            for i in range(num_to_buy):
                vt_symbol = buy_candidates[i]
                
                if vt_symbol not in bars:
                    continue
                
                bar = bars[vt_symbol]
                price = bar.close_price
                if price <= 0:
                    continue
                    
                if target_value > 0:
                    volume = int((target_value / price) / 100) * 100
                    
                    if volume > 0:
                        self.buy(vt_symbol, price * 1.02, volume)
                        print(f"{date_str}, {vt_symbol} Buy, price: {price * 1.02}, volume: {volume}, score: {scores.get(vt_symbol, 0)}")

        self.put_event()