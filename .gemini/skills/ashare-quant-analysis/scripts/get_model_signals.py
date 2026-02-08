import sys
import os
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from core.core_service import CoreService

def main():
    parser = argparse.ArgumentParser(description="Get Model Signals")
    parser.add_argument("--vt_symbol", required=True, help="VT Symbol")
    parser.add_argument("--version", required=True, help="Alpha Version (e.g., v8)")
    parser.add_argument("--start_date", default="2023-01-01", help="Start Date")
    parser.add_argument("--end_date", default="2025-01-01", help="End Date")
    
    args = parser.parse_args()
    
    version = args.version.lower()
    if not version.startswith("v"):
        version = "v" + version
        
    signal_name = f"ashare_mlp_signal_{version}"
    
    service = CoreService()
    
    try:
        start_dt = datetime.strptime(args.start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(args.end_date, "%Y-%m-%d")
    except ValueError:
        print("Error: Date format should be YYYY-MM-DD")
        return
    
    print(f"Getting signals for {args.vt_symbol} from {signal_name}...")
    data = service.get_signals_data(signal_name, start_dt, end_dt, [args.vt_symbol])
    
    if "error" in data:
        print(f"Error: {data['error']}")
        # Try finding available signals
        available = service.get_signals()
        print(f"Available signals: {available}")
        return

    # Print series data
    if data.get("series"):
        series = data["series"][0]
        dates = data["dates"]
        scores = series["data"]
        
        print("-" * 30)
        print(f"Signal Scores for {args.vt_symbol}")
        # Print last 10
        count = 0
        for i in range(max(0, len(dates)-10), len(dates)):
            score = scores[i]
            if score is None: score = "N/A"
            print(f"{dates[i]}: {score}")
            count += 1
        
        if count == 0:
            print("No signal data in range.")
            
    else:
        print("No signal data found.")

if __name__ == "__main__":
    main()
