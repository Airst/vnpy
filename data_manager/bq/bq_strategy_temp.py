from bigmodule import M
import dai
import pandas as pd
import numpy as np

# <aistudiograph>

# @module(comment="读取信号数据")
m_signal = M.extract_data_dai.v16(
    sql="""
SELECT
    date,
    instrument,
    final_signal as score
FROM ashare_mlp_signal_v7
ORDER BY date, instrument
""",
    start_date="2020-01-01",
    start_date_bound_to_trading_date=True,
    end_date="2026-01-01",
    end_date_bound_to_trading_date=True,
    before_start_days=0,
    m_name="m_signal"
)

# @param(id="m_bt", name="initialize")
def m_bt_initialize(context):
    from bigtrader.finance.commission import PerOrder
    context.set_commission(PerOrder(buy_cost=0.0003, sell_cost=0.0013, min_cost=5))
    
    # 策略参数 (来自 MultiFactorStrategy)
    context.max_holdings = 5
    context.buy_threshold = 1.0
    context.sell_threshold = 1.54
    context.stop_loss_pct = 0.05
    context.trailing_stop_pct = 0.15
    context.cooldown_days = 3
    
    # 状态变量
    context.pos_entry_price = {}
    context.pos_high_price = {}
    context.cooldown_map = {} 
    context.stock_count = context.max_holdings
    
    print("MultiFactorStrategy Initialized (BQ)")

# @param(id="m_bt", name="before_trading_start")
def m_bt_before_trading_start(context, data):
    pass

# @param(id="m_bt", name="handle_tick")
def m_bt_handle_tick(context, tick):
    pass

# @param(id="m_bt", name="handle_data")
def m_bt_handle_data(context, data):
    import pandas as pd
    import numpy as np
    
    if data.current_dt is None:
        return
        
    current_date = data.current_dt.strftime("%Y-%m-%d")
    
    # --- 0. 更新冷却期 ---
    expired_cooldowns = []
    for s in context.cooldown_map:
        context.cooldown_map[s] -= 1
        if context.cooldown_map[s] <= 0:
            expired_cooldowns.append(s)
    for s in expired_cooldowns:
        del context.cooldown_map[s]

    # --- 1. 获取当日持仓 ---
    # 获取当前持有且仓位大于0的股票
    equities = {e: p for e, p in context.portfolio.positions.items() if p.amount > 0}
    current_holdings = list(equities.keys())
    
    # --- 2. 获取当日信号与候选 ---
    # context.data 包含回测数据 (m_signal.data)
    try:
        today_signals = context.data[context.data["date"] == current_date]
    except:
        return

    scores = {}
    if not today_signals.empty:
        scores = dict(zip(today_signals["instrument"], today_signals["score"]))
        
    # 候选池：分数 > buy_threshold 且 不在冷却期
    # 按分数降序排列
    candidates = [s for s, sc in scores.items() if sc > context.buy_threshold and s not in context.cooldown_map]
    buy_list = sorted(candidates, key=lambda x: scores[x], reverse=True)
    
    # --- 3. 维护止损与卖出 (Sell Logic) ---
    stop_loss_triggered = set()
    sell_stock = []
    
    for ins in current_holdings:
        # A. 价格数据更新与止损检查
        try:
            price = data.current(ins, "close")
            if price is None or np.isnan(price):
                continue
            
            # 初始化或更新 入场/最高价
            cost = equities[ins].cost_basis
            if ins not in context.pos_entry_price:
                context.pos_entry_price[ins] = cost
                context.pos_high_price[ins] = price
            
            if price > context.pos_high_price[ins]:
                context.pos_high_price[ins] = price
                
            entry = context.pos_entry_price[ins]
            high = context.pos_high_price[ins]
            
            # 检查止损
            hard_stop = entry * (1 - context.stop_loss_pct)
            trailing_stop = high * (1 - context.trailing_stop_pct)
            
            if price < hard_stop or price < trailing_stop:
                stop_loss_triggered.add(ins)
                context.cooldown_map[ins] = context.cooldown_days
                
        except Exception:
            pass

        # B. 卖出判定
        should_sell = False
        score = scores.get(ins, -999) # 如果没有信号，假设很低
        
        # 1. 触发止损 -> 卖
        if ins in stop_loss_triggered:
            should_sell = True
            
        # 2. 分数低于阈值 -> 卖 (注意：不再强制 Top N 卖出，只看绝对分数)
        elif score < context.sell_threshold:
            should_sell = True
            
        if should_sell:
            context.order_target_percent(ins, 0)
            sell_stock.append(ins)
            # 清理状态
            if ins in context.pos_entry_price: del context.pos_entry_price[ins]
            if ins in context.pos_high_price: del context.pos_high_price[ins]
    
    # 更新持仓数量 (减去已发出卖单的股票)
    stock_now = len(current_holdings) - len(sell_stock)
    
    # --- 4. 生成买入订单 (Buy Logic) ---
    stock_count = context.stock_count # 目标最大持仓数 (5)
    
    # 需要买入的数量 = 最大持仓 - 当前预计持仓
    buy_num = stock_count - stock_now
    
    if buy_num > 0 and len(buy_list) > 0:
        # 筛选出实际上需要买入的（不在当前持有列表，也不在刚才卖出列表）
        # 注意：buy_list 已经是 Top N 排序且 > buy_threshold
        real_buy_candidates = [i for i in buy_list if i not in current_holdings and i not in sell_stock]
        
        # 截取前 buy_num 个
        target_buys = real_buy_candidates[:buy_num]
        
        if not target_buys:
            return

        # 资金分配逻辑：
        # 简单均分：每个新开仓使用 1/MaxHoldings 的资金权重?
        # 或者使用剩余资金 / buy_num?
        # 参考 temp.py 逻辑：context.order_value(instrument, cash_for_buy)
        # 这里为了保持一致性，使用 target_percent 更稳健，避免资金透支
        # 目标仓位权重
        target_weight = 1.0 / stock_count
        
        for ins in target_buys:
            # 检查是否有足够现金 (可选，order_target_percent 会自动处理，但加上更安全)
            # if context.portfolio.cash < ... : break
            
            context.order_target_percent(ins, target_weight)
            stock_now += 1

# @param(id="m_bt", name="handle_trade")
def m_bt_handle_trade(context, trade):
    pass

# @param(id="m_bt", name="handle_order")
def m_bt_handle_order(context, order):
    pass

# @param(id="m_bt", name="after_trading")
def m_bt_after_trading(context, data):
    pass

# @module(comment="BigTrader高性能回测")
m_bt = M.bigtrader.v43(
    data=m_signal.data,
    start_date="",
    end_date="",
    initialize=m_bt_initialize,
    before_trading_start=m_bt_before_trading_start,
    handle_tick=m_bt_handle_tick,
    handle_data=m_bt_handle_data,
    handle_trade=m_bt_handle_trade,
    handle_order=m_bt_handle_order,
    after_trading=m_bt_after_trading,
    capital_base=1000000,
    frequency="daily",
    product_type="股票",
    rebalance_period_type="交易日",
    rebalance_period_days="1",
    order_price_field_buy="open",
    order_price_field_sell="close",
    benchmark="沪深300指数",
    m_name="m_bt"
)

# </aistudiograph>