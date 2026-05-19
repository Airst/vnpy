"""
回写 holding 标记到已有回测 JSON 文件。

逻辑：用 FIFO 方式遍历 trades，计算出回测结束时仍持有的股票，
对这些股票的买入记录标记 holding=True。
"""
import json
import os
import glob

BACKTEST_DIR = os.path.join(os.path.dirname(__file__), "..", "alpha_db", "backtest")


def mark_holding(trades):
    """
    FIFO 多空核销：空单按顺序核销同股票的多单。
    回测结束时未被核销的多单标记 holding=True。
    """
    # position_tracker[symbol] = deque of {"idx": trade_index, "volume": remaining}
    position_tracker = {}

    for i, trade in enumerate(trades):
        symbol = trade["symbol"]
        direction = trade["direction"]
        volume = trade["volume"]

        if symbol not in position_tracker:
            position_tracker[symbol] = []

        if direction == "多":
            position_tracker[symbol].append({"idx": i, "volume": volume})
        elif direction == "空":
            remaining = volume
            while remaining > 1e-6 and position_tracker[symbol]:
                buy = position_tracker[symbol][0]
                match = min(remaining, buy["volume"])
                remaining -= match
                buy["volume"] -= match
                if buy["volume"] <= 1e-6:
                    position_tracker[symbol].pop(0)

    # 收集未被核销的多单 index
    holding_indices = set()
    for positions in position_tracker.values():
        for pos in positions:
            if pos["volume"] > 1e-6:
                holding_indices.add(pos["idx"])

    # 标记
    for i, trade in enumerate(trades):
        if i in holding_indices:
            trade["holding"] = True
        else:
            trade.pop("holding", None)

    return trades


def main():
    pattern = os.path.join(BACKTEST_DIR, "*.json")
    files = glob.glob(pattern)
    print(f"Found {len(files)} backtest files")

    for filepath in sorted(files):
        filename = os.path.basename(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        trades = data.get("trades")
        if not trades:
            print(f"  SKIP (no trades): {filename}")
            continue

        data["trades"] = mark_holding(trades)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        holding_count = sum(1 for t in data["trades"] if t.get("holding"))
        print(f"  OK: {filename} ({holding_count} holding trades marked)")


if __name__ == "__main__":
    main()
