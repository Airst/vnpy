"""
LLM Rating Script.

Standalone script for running LLM evaluations on signal Top-K stocks.
Decoupled from model training and backtesting.

Usage:
    python llm_rating.py -v9                    # Run on v9 signals, Top 20
    python llm_rating.py -v9 -k 30              # Run on v9 signals, Top 30
    python llm_rating.py -v9 -f                 # Force re-evaluate, even if valid ratings exist
    python llm_rating.py -v9 -k 30 -f -bs 8     # Force, Top 30, batch size 8
    python llm_rating.py --list-versions        # List available signal versions
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure we're running from project root
project_root = Path(__file__).parent
os.chdir(project_root)

from core.llm.rating_task import LLMRatingTask, DEFAULT_SIGNAL_DIR


def list_versions(signal_dir: str):
    """List available signal versions."""
    sig_dir = Path(signal_dir)
    if not sig_dir.exists():
        print(f"Signal directory not found: {signal_dir}")
        return

    versions = []
    for f in sig_dir.glob("ashare_mlp_signal_v*.parquet"):
        stem = f.stem
        version = stem.replace("ashare_mlp_signal_", "")
        versions.append(version)

    if not versions:
        print("No signal files found.")
        return

    versions.sort()
    print("Available signal versions:")
    for v in versions:
        filepath = sig_dir / f"ashare_mlp_signal_{v}.parquet"
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"  {v:<20} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="LLM Rating: Evaluate Top-K stocks from signal using LLM"
    )

    parser.add_argument(
        "-v", "--version",
        required=False,
        help="Signal version (e.g., v8, v9)",
    )

    # Version shortcut flags
    parser.add_argument("-v8", action="store_true", help="Run on v8 signals")
    parser.add_argument("-v9", action="store_true", help="Run on v9 signals")

    parser.add_argument(
        "-k", "--top_k",
        type=int,
        default=20,
        help="Number of top stocks to evaluate (default: 20)",
    )

    parser.add_argument(
        "-f", "--force",
        action="store_true",
        help="Force re-evaluate all stocks, even if they have valid (non-expired) ratings",
    )

    parser.add_argument(
        "-bs", "--batch_size",
        type=int,
        default=4,
        help="LLM batch size (default: 4)",
    )

    parser.add_argument(
        "-w", "--max_workers",
        type=int,
        default=4,
        help="LLM max concurrent workers (default: 4)",
    )

    parser.add_argument(
        "--signal-dir",
        default=DEFAULT_SIGNAL_DIR,
        help=f"Signal directory (default: {DEFAULT_SIGNAL_DIR})",
    )

    parser.add_argument(
        "--rating-dir",
        default="core/alpha_db/llm_tasks",
        help="Rating output directory (default: core/alpha_db/llm_tasks)",
    )

    parser.add_argument(
        "--list-versions",
        action="store_true",
        help="List available signal versions and exit",
    )

    args = parser.parse_args()

    if args.list_versions:
        list_versions(args.signal_dir)
        return

    # Determine version
    version = args.version
    if not version:
        if args.v8:
            version = "v8"
        elif args.v9:
            version = "v9"

    if not version:
        print("Error: Please specify a version using -v <version> or -v8/-v9 flags.")
        print("Use --list-versions to see available signals.")
        sys.exit(1)

    # Normalize version
    if not version.startswith("v"):
        version = "v" + version

    print(f"=== LLM Rating Script ===")
    print(f"Version: {version}")
    print(f"Top K: {args.top_k}")
    print(f"Force: {args.force}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max workers: {args.max_workers}")

    # Run
    task = LLMRatingTask(
        signal_dir=args.signal_dir,
        rating_dir=args.rating_dir,
    )

    try:
        results = task.run(
            version=version,
            top_k=args.top_k,
            force=args.force,
            batch_size=args.batch_size,
            max_workers=args.max_workers,
        )

        if not results:
            print("\nNo ratings generated.")
            sys.exit(0)

        print(f"\nDone. {len(results)} stocks evaluated.")

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Use --list-versions to see available signal files.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
