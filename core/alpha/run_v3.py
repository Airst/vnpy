import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
import polars as pl
import re
import argparse
from vnpy.alpha import (
    logger
)


# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.alpha.engine import AlphaEngine
from core.alpha.factor_calculator import FactorCalculator
from mlp_signals import MLPSignals
from core.selector.selector import FundamentalSelector

# --- Logger Redirection ---
class LoggerWriter:
    def __init__(self, writer, file):
        self.writer = writer
        self.file = file

    def write(self, message):
        if not message:
            return
        
        if message == "^" or message == "\n" or message.strip() == "":
            self.file.write(message)
            return    
        # If message already starts with a date (e.g. 2025-12-20 or [2025-12-20), don't add timestamp
        if re.search(r'^\s*(\[)?\d{4}-\d{2}-\d{2}', message):
            self.file.write(message)
        else:
            timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")
            self.file.write(timestamp + message)
        self.file.flush()

    def flush(self):
        self.writer.flush()
        self.file.flush()

    def close(self):
        self.file.close()

    def isatty(self):
        if hasattr(self.writer, "isatty"):
            return self.writer.isatty()
        return False

    def fileno(self):
        if hasattr(self.writer, "fileno"):
            return self.writer.fileno()
        raise OSError("LoggerWriter has no fileno")

# Redirect stdout and stderr to web_ui.log
# Check if already redirected to avoid double wrapping on reload
if not hasattr(sys.stdout, 'file') or not isinstance(sys.stdout, LoggerWriter):
    try:
        file = open("run_v3.log", 'w', encoding='utf-8')

        sys.stdout = LoggerWriter(sys.stdout, file)
        sys.stderr = LoggerWriter(sys.stderr, file)

        # Remove default output
        logger.remove()

        # Add terminal output
        fmt: str = "{time:YYYY-MM-DD HH:mm:ss} {message}"
        logger.add(sys.stdout, colorize=True, format=fmt)
    except Exception as e:
        print(f"Failed to setup logger redirection: {e}")
# --------------------------

def run(only_factor: bool = False):
    print("Initializing Alpha Engine for V3...")
    engine = AlphaEngine(
        factor_calculator=FactorCalculator(),
        mlp_signals=MLPSignals(),
        selector=FundamentalSelector(),
        signal_name="ashare_mlp_signal_v3",
        start_date="2020-12-28",
        end_date=datetime.now().strftime("%Y-%m-%d")
    )
    
    signal_df = engine.calculate_factors()

    engine.analyze_factor_performance(signal_df)

    if not only_factor:
        signal_df = engine.calculate_signals(signal_df)

        engine.save_factors(signal_df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="这是一个示例脚本")
    parser.add_argument("-a", "--ans")
    args = parser.parse_args()
    run(True if args.ans else False)

