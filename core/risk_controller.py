"""
组合级风险控制模块

== 当前状态 ==
风险信号:
  1. 组合回撤（trailing peak drawdown）→ 阶梯式减仓
  2. 短期波动率飙升（20d vs 60d std）→ 额外减仓
恢复机制: 非对称（减仓即时，恢复+1/cooldown周期）
默认配置: base_max_holdings=5, 回撤阶梯[-15%,-20%,-25%,-30%,-35%], vol_threshold=2.0

== 设计决策 ==
- 阶梯式减仓而非一刀切: 避免单次回撤即清仓导致错过反弹
- 非对称恢复: 防止快速反弹后立即满仓又遇二次下跌
- 零持仓死锁恢复: max_holdings=0 时经过 cooldown 后重置 peak_equity
- 波动率信号: 短期/长期 std 比值，捕捉市场恐慌但避免常态波动误触发
- 职责边界: 只管组合级风险（回撤/波动），不管个股止损（那是策略层的事）

== 失败记录 ==
- 市场趋势过滤(60日MA负→减仓): 强制卖出在反弹中错失收益，Q2从-7.31%恶化到-7.52%，Sharpe从1.17降到1.02。风控层无法解决模型因子IC反转的系统性问题
- 无其他失败（风控模块自 V8 以来参数稳定）
"""
from collections import deque
from typing import Dict, List, Tuple
import numpy as np


class RiskController:
    """
    Portfolio-level risk control module.

    Tracks portfolio equity curve, computes drawdown and volatility,
    outputs dynamic max_holdings and force-sell list.

    Risk Signals:
      1. Portfolio drawdown from trailing peak (primary)
      2. Short-term volatility spike vs long-term baseline (secondary)

    Recovery is asymmetric: fast to reduce, slow to recover.
    """

    def __init__(
        self,
        base_max_holdings: int = 5,
        drawdown_levels: List[float] = None,
        vol_threshold: float = 2.0,
        vol_short_window: int = 20,
        vol_long_window: int = 60,
        recovery_cooldown_days: int = 3,
        enabled: bool = True,
    ):
        self.base_max_holdings = base_max_holdings
        self.enabled = enabled

        # Drawdown thresholds (must be sorted descending, i.e. least severe first)
        if drawdown_levels is None:
            drawdown_levels = [-0.15, -0.20, -0.25, -0.30, -0.35]
        self.drawdown_levels = sorted(drawdown_levels, reverse=True)

        self.vol_threshold = vol_threshold
        self.vol_short_window = vol_short_window
        self.vol_long_window = vol_long_window
        self.recovery_cooldown_days = recovery_cooldown_days

        # --- internal state ---
        self.peak_equity: float = 0.0
        self.prev_equity: float = 0.0
        self.daily_returns: deque = deque(maxlen=max(vol_long_window, 120))
        self.current_max_holdings: int = base_max_holdings
        self.bars_since_change: int = 999  # large initial value
        self._bars_at_zero: int = 0  # tracks consecutive bars at max_holdings=0
        self._initialized: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def on_bar(
        self,
        portfolio_equity: float,
        current_positions: List[str],
        signal_scores: Dict[str, float],
    ) -> Tuple[int, List[str]]:
        """
        Called once per trading day.

        Parameters
        ----------
        portfolio_equity : current portfolio value (cash + position market values)
        current_positions : list of currently held vt_symbols
        signal_scores : {vt_symbol: score} for today

        Returns
        -------
        (max_holdings, force_sell_symbols)
            max_holdings: dynamic cap on number of positions
            force_sell_symbols: symbols that must be sold immediately
        """
        # --- bypass if disabled ---
        if not self.enabled:
            return self.base_max_holdings, []

        # --- track equity curve ---
        if not self._initialized:
            self.peak_equity = portfolio_equity
            self.prev_equity = portfolio_equity
            self._initialized = True
        else:
            daily_ret = (portfolio_equity / self.prev_equity) - 1 if self.prev_equity > 0 else 0.0
            self.daily_returns.append(daily_ret)
            self.prev_equity = portfolio_equity

        if portfolio_equity > self.peak_equity:
            self.peak_equity = portfolio_equity

        self.bars_since_change += 1

        # --- signal 1: drawdown ---
        drawdown = (portfolio_equity / self.peak_equity) - 1 if self.peak_equity > 0 else 0.0
        dd_reduction = self._drawdown_reduction(drawdown)

        # --- zero-holdings deadlock breaker ---
        if self.current_max_holdings == 0:
            self._bars_at_zero += 1
            if self._bars_at_zero >= self.recovery_cooldown_days:
                self.peak_equity = portfolio_equity
                self._bars_at_zero = 0
                drawdown = 0.0
                dd_reduction = 0
        else:
            self._bars_at_zero = 0

        # --- signal 2: volatility spike ---
        vol_reduction = self._volatility_reduction()

        # --- combine: total reduction ---
        total_reduction = min(dd_reduction + vol_reduction, self.base_max_holdings)
        target_max = self.base_max_holdings - total_reduction

        # --- apply asymmetric speed ---
        new_max = self._apply_asymmetric_recovery(target_max)
        self.current_max_holdings = new_max

        # --- determine force sells ---
        force_sell = self._compute_force_sells(
            current_positions, signal_scores, new_max
        )

        return new_max, force_sell

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _drawdown_reduction(self, drawdown: float) -> int:
        """Map drawdown percentage to reduction count."""
        reduction = 0
        for level in self.drawdown_levels:
            if drawdown < level:
                reduction += 1
        return reduction

    def _volatility_reduction(self) -> int:
        """Check if recent volatility is spiking relative to longer-term."""
        n_short = self.vol_short_window
        n_long = self.vol_long_window

        if len(self.daily_returns) < n_short:
            return 0

        recent = list(self.daily_returns)
        vol_short = np.std(recent[-n_short:])

        if len(recent) >= n_long:
            vol_long = np.std(recent[-n_long:])
        else:
            vol_long = vol_short

        if vol_long <= 0:
            return 0

        if vol_short > vol_long * self.vol_threshold:
            return 1

        return 0

    def _apply_asymmetric_recovery(self, target_max: int) -> int:
        """
        Fast reduction, slow recovery.

        Reduction: immediate (target can be much lower than current).
        Recovery: +1 per cooldown period, only if target is higher.
        """
        current = self.current_max_holdings

        if target_max < current:
            # Immediate reduction
            self.bars_since_change = 0
            return target_max

        if target_max > current:
            # Slow recovery: only +1 if cooldown has passed
            if self.bars_since_change >= self.recovery_cooldown_days:
                self.bars_since_change = 0
                return current + 1
            else:
                return current

        # No change
        return current

    def _compute_force_sells(
        self,
        current_positions: List[str],
        signal_scores: Dict[str, float],
        max_holdings: int,
    ) -> List[str]:
        """
        If holding more than max_holdings, return weakest-scored symbols to sell.
        """
        n_held = len(current_positions)
        if n_held <= max_holdings:
            return []

        need_to_sell = n_held - max_holdings

        # Sort by signal score ascending (weakest first)
        scored = sorted(
            current_positions,
            key=lambda s: signal_scores.get(s, -999.0),
        )

        return scored[:need_to_sell]
