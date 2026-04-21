"""
Backtest Next Trading Day Extractor
Extracts next trading day signal details from a backtest JSON file.

Usage:
    python core/tools/extract_next_day.py core/alpha_db/backtest/ashare_mlp_signal_v9_20260420_20260420_20260420_225544.json
    python core/tools/extract_next_day.py --latest v9
"""
import sys
import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

BACKTEST_DIR = PROJECT_ROOT / "core" / "alpha_db" / "backtest"


def extract_from_file(json_path: str):
    """Extract next trading day details from a specific backtest JSON file."""
    path = Path(json_path)
    if not path.exists():
        print(f"[Error] File not found: {path}")
        return

    with open(path, "r") as f:
        data = json.load(f)

    trades = data.get("trades", [])
    stats = data.get("statistics", {})

    print("=" * 60)
    print(f"File: {path.name}")
    print("=" * 60)

    # Print summary stats if available
    if stats.get("total_return", 0) != 0:
        print(f"\nBacktest Period: {stats.get('start_date', '?')} ~ {stats.get('end_date', '?')}")
        print(f"Total Return:    {stats.get('total_return', 0):.2f}%")
        print(f"Annual Return:   {stats.get('annual_return', 0):.2f}%")
        print(f"Sharpe Ratio:    {stats.get('sharpe_ratio', 0):.2f}")
        print(f"Max DD:          {stats.get('max_ddpercent', 0):.2f}%")

    # Extract next day trades
    next_day_trades = [t for t in trades if t.get("date") == "下个交易日"]

    if not next_day_trades:
        print("\nNo next trading day signals found in this file.")
        return

    print(f"\nNext Trading Day Signals ({len(next_day_trades)} stocks):")
    print("-" * 60)
    print(f"{'Symbol':<16} {'Direction':<8} {'Price':>10} {'Volume':>12} {'Notional':>14}")
    print("-" * 60)

    total_notional = 0
    for t in next_day_trades:
        symbol = t.get("symbol", "")
        direction = t.get("direction", "")
        price = t.get("price", 0)
        volume = t.get("volume", 0)
        notional = price * volume
        total_notional += notional
        print(f"{symbol:<16} {direction:<8} {price:>10.2f} {volume:>12.0f} {notional:>14.2f}")

    print("-" * 60)
    print(f"{'Total':<16} {'':<8} {'':>10} {'':>12} {total_notional:>14.2f}")
    print(f"\nCapital allocation: {total_notional:,.2f}")


def find_latest(version: str = "v9"):
    """Find the latest backtest JSON file matching the version pattern.

    Filename format: ashare_mlp_signal_{version}_{start}_{end}_{gen_date}_{gen_time}.json
    Sort by the generation date+time portion (last two underscore-separated segments).
    """
    pattern = f"ashare_mlp_signal_{version}_*.json"
    files = BACKTEST_DIR.glob(pattern)

    def sort_key(p: Path) -> str:
        # e.g. ashare_mlp_signal_v9_20260420_20260420_20260420_225544.json
        # parts: [ashare, mlp, signal, v9, 20260420, 20260420, 20260420, 225544.json]
        stem = p.stem  # remove .json
        parts = stem.split("_")
        # gen_date + gen_time are the last 2 parts
        return "_".join(parts[-2:])

    files = sorted(files, key=sort_key)
    if not files:
        print(f"[Error] No backtest file found matching {pattern} in {BACKTEST_DIR}")
        return None
    return str(files[-1])


def main():
    parser = argparse.ArgumentParser(description="Extract next trading day signals from backtest JSON")
    parser.add_argument("json_path", nargs="?", help="Path to backtest JSON file")
    parser.add_argument("--latest", nargs="?", const="v9", default=None,
                        help="Use the latest backtest file for a version (default: v9)")
    args = parser.parse_args()

    if args.latest:
        json_path = find_latest(args.latest)
        if json_path:
            print(f"[Info] Using latest {args.latest} backtest: {Path(json_path).name}\n")
            extract_from_file(json_path)
    elif args.json_path:
        extract_from_file(args.json_path)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
