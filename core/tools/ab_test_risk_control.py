"""
A/B Test: Risk Control vs No Risk Control
Uses the same signal file to isolate the risk control effect from model training variance.
"""
from datetime import datetime
from core.core_service import CoreService

core = CoreService()
signal_name = "ashare_mlp_signal_v9"
start = datetime.strptime("2022-01-01", "%Y-%m-%d")
end = datetime.strptime("2026-04-03", "%Y-%m-%d")

print("=" * 60)
print("A/B Test: Risk Control Effect (same signals)")
print("=" * 60)

# --- A: Without risk control ---
print("\n>>> [A] Running backtest WITHOUT risk control...")
result_a = core.run_backtest(
    strategy_name="MultiFactorStrategy",
    start=start, end=end,
    setting={
        "max_holdings": 5,
        "signal_name": signal_name,
        "risk_control_enabled": False,
    }
)

# --- B: With risk control ---
print("\n>>> [B] Running backtest WITH risk control...")
result_b = core.run_backtest(
    strategy_name="MultiFactorStrategy",
    start=start, end=end,
    setting={
        "max_holdings": 5,
        "signal_name": signal_name,
        "risk_control_enabled": True,
    }
)

# --- Compare ---
print("\n" + "=" * 60)
print("A/B COMPARISON (same model, same signals)")
print("=" * 60)

stats_a = result_a["statistics"] if isinstance(result_a, dict) else {}
stats_b = result_b["statistics"] if isinstance(result_b, dict) else {}

metrics = [
    ("total_return", "Total Return %"),
    ("annual_return", "Annual Return %"),
    ("sharpe_ratio", "Sharpe Ratio"),
    ("max_ddpercent", "Max DD %"),
    ("return_drawdown_ratio", "Return/DD Ratio"),
]

print(f"{'Metric':<20} {'No RiskCtrl':>15} {'With RiskCtrl':>15} {'Delta':>10}")
print("-" * 62)
for key, label in metrics:
    va = stats_a.get(key, 0)
    vb = stats_b.get(key, 0)
    delta = vb - va
    print(f"{label:<20} {va:>14.2f}% {vb:>14.2f}% {delta:>+9.2f}" if "%" in label or "Return" in label
          else f"{label:<20} {va:>15.2f} {vb:>15.2f} {delta:>+10.2f}")
