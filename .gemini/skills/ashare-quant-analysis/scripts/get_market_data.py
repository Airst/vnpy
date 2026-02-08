import sys
import os
import argparse
from pathlib import Path
import polars as pl

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from core.alpha.data_loader import DataLoader
from vnpy.alpha.lab import AlphaLab

def main():
    parser = argparse.ArgumentParser(description="Get Market Data")
    parser.add_argument("--vt_symbol", required=True, help="VT Symbol")
    parser.add_argument("--start_date", default="2023-01-01", help="Start Date")
    parser.add_argument("--end_date", default="2025-01-01", help="End Date")
    
    args = parser.parse_args()
    
    lab = AlphaLab(str(PROJECT_ROOT / "core/alpha_db"))
    loader = DataLoader(lab)
    
    print(f"Loading market data for {args.vt_symbol}...")
    try:
        df = loader.load_ashare_data([args.vt_symbol], args.start_date, args.end_date)
        
        if df.is_empty():
            print("No data found.")
            return
            
        print(f"Market Data for {args.vt_symbol}")
        print(df.tail(10))
        
        output_file = f"market_{args.vt_symbol}.csv"
        df.write_csv(output_file)
        print(f"Saved to {output_file}")
    except Exception as e:
        print(f"Error loading market data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
