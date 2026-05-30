"""
隔夜持股策略 — 严格2日持有期

== 设计决策 ==
- 匹配标签 Close(T+2)/Open(T+1)-1：T日信号 → T+1开盘买入 → T+2收盘卖出
- 固定持有期2天，无止损、无信号退出，纯粹测试因子选股能力
- 每日等权买入 top-N 信号股票，持有到期自动卖出
- 回测中: on_bars(T日) 下单 → cross_order(T+1日) 以开盘成交 → on_bars(T+2日) 卖出
"""
from datetime import datetime
from pathlib import Path
from typing import Dict

from vnpy_portfoliostrategy import StrategyTemplate
from vnpy.alpha.lab import AlphaLab
from vnpy.trader.object import BarData, TradeData
from vnpy.trader.constant import Direction, Offset

ALPHA_DB_PATH = "core/alpha_db"


class OvernightStrategy(StrategyTemplate):
    """
    严格隔夜持股策略：买入后持有2个交易日卖出。
    """
    author = "System"

    parameters = [
        "signal_name",
        "max_holdings",
        "capital",
        "rate",
        "buy_threshold",
    ]

    variables = [
        "cash",
    ]

    project_root = Path(__file__).parent.parent.parent
    lab_path = project_root / ALPHA_DB_PATH
    lab = AlphaLab(str(lab_path))

    def __init__(self, portfolio_engine, strategy_name, vt_symbols, setting):
        super().__init__(portfolio_engine, strategy_name, vt_symbols, setting)

        self.signal_name = setting.get("signal_name", "ashare_multi_factor")
        self.max_holdings = int(setting.get("max_holdings", 5))
        self.capital = float(setting.get("capital", 1_000_000))
        self.buy_threshold = float(setting.get("buy_threshold", 1))
        self.rate = float(setting.get("rate", 0.0003))

        self.cash = self.capital
        self.signal_data = {}
        self.last_prices = {}

        # 持仓追踪: {vt_symbol: holding_days}
        self.holding_days: Dict[str, int] = {}

        print(f"OvernightStrategy initialized: signal={self.signal_name}, "
              f"max_holdings={self.max_holdings}, capital={self.capital}")

    def on_init(self):
        print("OvernightStrategy Initialized")
        self.load_signals()

    def on_start(self):
        print("OvernightStrategy Started")

    def on_stop(self):
        print("OvernightStrategy Stopped")

    def update_trade(self, trade: TradeData):
        """Track cash and commission."""
        raw_commission = trade.price * trade.volume * self.rate
        commission = max(raw_commission, 5.0)

        if trade.direction == Direction.LONG:
            self.cash -= trade.price * trade.volume
        elif trade.direction == Direction.SHORT:
            stamp_duty = trade.price * trade.volume * 0.0005
            commission += stamp_duty
            self.cash += trade.price * trade.volume

        self.cash -= commission
        super().update_trade(trade)

    def load_signals(self):
        """Load pre-calculated signals from AlphaLab."""
        try:
            df = self.lab.load_signal(self.signal_name)

            if df is None or df.is_empty():
                print(f"No signal data found for {self.signal_name}")
                return

            for row in df.iter_rows(named=True):
                dt = row["datetime"]
                if hasattr(dt, "date"):
                    dt_str = dt.strftime("%Y-%m-%d")
                else:
                    dt_str = str(dt).split(" ")[0]

                symbol = row["vt_symbol"]
                score = row.get("final_signal")
                if score is None:
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

    def on_bars(self, bars: Dict[str, BarData]):
        """
        每日逻辑:
        1. 更新持仓天数
        2. 卖出持有>=2天的股票
        3. 买入当日top-N信号股票
        """
        self.cancel_all()

        if not bars:
            return

        current_dt = list(bars.values())[0].datetime
        date_str = current_dt.strftime("%Y-%m-%d")

        # Update prices
        for vt_symbol, bar in bars.items():
            self.last_prices[vt_symbol] = bar.close_price

        # 1. 更新持仓天数
        for vt_symbol in list(self.holding_days.keys()):
            self.holding_days[vt_symbol] += 1

        # 2. 卖出持有 >= 2天的股票（T+1买入成交算第1天，T+2以收盘价MOC卖出）
        for vt_symbol in list(self.holding_days.keys()):
            if self.holding_days[vt_symbol] >= 2:
                pos = self.get_pos(vt_symbol)
                if pos > 0 and vt_symbol in bars:
                    bar = bars[vt_symbol]
                    if bar.close_price > 0:
                        self.strategy_engine.send_moc_order(
                            self, vt_symbol, Direction.SHORT, Offset.CLOSE, pos
                        )
                        print(f"{date_str}, {vt_symbol} Sell MOC (EXPIRE), "
                              f"price:{bar.close_price:.2f}, held {self.holding_days[vt_symbol]} days")
                del self.holding_days[vt_symbol]

        # 3. 获取信号并买入
        scores = self.signal_data.get(date_str, {})
        if not scores:
            return

        # 当前持仓
        held_symbols = [s for s in self.vt_symbols if self.get_pos(s) > 0]
        held_count = len(held_symbols)

        # 排名选股
        available_symbols = list(bars.keys())
        sorted_symbols = sorted(available_symbols,
                                key=lambda s: scores.get(s, -999), reverse=True)

        buy_candidates = []
        for s in sorted_symbols:
            if scores.get(s, 0) > self.buy_threshold and s not in held_symbols:
                buy_candidates.append(s)
            if len(buy_candidates) >= self.max_holdings - held_count:
                break

        # 等权买入
        num_to_buy = len(buy_candidates)
        if num_to_buy > 0 and self.cash > 0:
            target_value = (self.cash / num_to_buy) * 0.95

            for vt_symbol in buy_candidates:
                if vt_symbol not in bars:
                    continue
                bar = bars[vt_symbol]
                price = bar.close_price
                if price <= 0:
                    continue

                volume = int((target_value / price) / 100) * 100
                if volume > 0:
                    self.buy(vt_symbol, price * 1.02, volume)
                    # 买入当天开始计数为0，成交后第一天算1
                    self.holding_days[vt_symbol] = 0
                    print(f"{date_str}, {vt_symbol} Buy, price:{price * 1.02:.2f}, "
                          f"volume:{volume}, score:{scores.get(vt_symbol, 0):.3f}")

        self.put_event()
