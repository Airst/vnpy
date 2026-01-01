import sys
import os
from datetime import datetime
from pathlib import Path
import re
import argparse

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from vnpy.alpha import logger
from core.alpha.engine import AlphaEngine
from core.alpha.mlp_signals import MLPSignals
from core.selector.selector import FundamentalSelector

# Import Calculators
from core.alpha.v3_factor_calculator import V3FactorCalculator
from core.alpha.v4_factor_calculator import V4FactorCalculator
from core.alpha.v5_factor_calculator import V5FactorCalculator

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

def setup_logger(version: str):
    log_filename = f"core/alpha/run_{version}.log"
    
    # Redirect stdout and stderr to log file
    if not hasattr(sys.stdout, 'file') or not isinstance(sys.stdout, LoggerWriter):
        try:
            file = open(log_filename, 'w', encoding='utf-8')

            sys.stdout = LoggerWriter(sys.stdout, file)
            sys.stderr = LoggerWriter(sys.stderr, file)

            # Remove default output
            logger.remove()

            # Add terminal output (which now goes to file via LoggerWriter)
            fmt: str = "{time:YYYY-MM-DD HH:mm:ss} {message}"
            logger.add(sys.stdout, colorize=True, format=fmt)
        except Exception as e:
            print(f"Failed to setup logger redirection: {e}")

# --------------------------

def run(version: str, only_factor: bool = False, sync: bool = False):
    # Normalize version string
    version = version.lower()
    if not version.startswith("v"):
        version = "v" + version
    
    setup_logger(version)
    
    print(f"Initializing Alpha Engine for {version.upper()}...")
    
    if version == "v3":
        calculator = V3FactorCalculator()
        signal_name = "ashare_mlp_signal_v3"
        description = "V3"
    elif version == "v4":
        calculator = V4FactorCalculator()
        signal_name = "ashare_mlp_signal_v4"
        description = "V4 (Alpha101)"
    elif version == "v5":
        calculator = V5FactorCalculator()
        signal_name = "ashare_mlp_signal_v5"
        description = "V5 (Alpha158)"
    else:
        print(f"Error: Unknown version '{version}'. Supported versions: v3, v4, v5")
        return

    engine = AlphaEngine(
        factor_calculator=calculator,
        mlp_signals=MLPSignals(),
        selector=FundamentalSelector(),
        signal_name=signal_name,
        start_date="2019-12-28",
        end_date=datetime.now().strftime("%Y-%m-%d")
    )
   
    if sync:
        print("Syncing data from remote source...")
        engine.sync_data()

        return
    
    signal_df = engine.calculate_factors()

    signal_df = engine.analyze_factor_performance(signal_df)

    if not only_factor:
        signal_df = engine.calculate_signals(signal_df)
        engine.save_factors(signal_df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Alpha Run Script")
    
    # Version argument: allow -v v3 or -v 3
    parser.add_argument("-v", "--version", required=False, help="Alpha version (e.g., v3, v4, v5)")
    
    # Support direct flags like -v3, -v4, -v5 for convenience/backward compatibility requests
    parser.add_argument("-v3", action="store_true", help="Run V3")
    parser.add_argument("-v4", action="store_true", help="Run V4")
    parser.add_argument("-v5", action="store_true", help="Run V5")
    
    parser.add_argument("-a", "--ans", action="store_true", help="Only calculate factors (no signal model)")
    parser.add_argument("-s", "--sync", action="store_true", help="Sync data before running")
    
    args = parser.parse_args()
    
    # Determine version
    selected_version = None
    if args.version:
        selected_version = args.version
    elif args.v3:
        selected_version = "v3"
    elif args.v4:
        selected_version = "v4"
    elif args.v5:
        selected_version = "v5"
    
    if not selected_version:
        print("Error: Please specify a version using -v [version] or -v3/-v4/-v5 flags.")
        sys.exit(1)
        
    run(selected_version, args.ans, args.sync)
