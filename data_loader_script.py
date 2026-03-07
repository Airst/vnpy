"""
数据加载脚本
用于加载 A 股市场数据并将 DataFrame 的列名及序号保存到文件中
"""
import os
from datetime import datetime
from pathlib import Path
import argparse

from vnpy.alpha import logger
from core.logger_writer import LoggerWriter
from core.alpha.engine import AlphaEngine
from core.alpha.mlp_signals import MLPSignals
from core.selector.selector import FundamentalSelector

# Import Calculator
from core.alpha.v8_factor_calculator import V8FactorCalculator


def setup_logger(version: str):
    """设置日志输出"""
    log_filename = f"log/data_cols.log"
    
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


def save_column_info(df, output_file: str):
    """
    将 DataFrame 的列名称及序号保存到文件中
    
    Args:
        df: Polars DataFrame
        output_file: 输出文件路径
    """
    columns = df.columns
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("DataFrame 列信息\n")
        f.write("=" * 60 + "\n")
        f.write(f"总列数：{len(columns)}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"{'序号':<10} {'列名':<40}\n")
        f.write("-" * 60 + "\n")
        for idx, col_name in enumerate(columns):
            f.write(f"{idx:<10} {col_name:<40}\n")
    
    print(f"\n列信息已保存到：{output_file}")


if __name__ == "__main__":
    import sys
    
    parser = argparse.ArgumentParser(description="数据加载脚本 - 仅加载数据并保存列信息")
    parser.add_argument("-v", "--vt", help="vt_symbol mode (single stock)")
    parser.add_argument("-s", "--start", default="2019-12-28", help="Start date (YYYY-MM-DD)")
    parser.add_argument("-e", "--end", help="End date (YYYY-MM-DD), defaults to last trading day")
    parser.add_argument("-o", "--output", default="core/alpha/data_columns_info.txt", help="Output file for column info")
    
    args = parser.parse_args()
    
    version = "data_load"
    setup_logger(version)
    
    print("初始化数据加载引擎...")
    
    selector = FundamentalSelector([args.vt]) if args.vt else FundamentalSelector()
    earlest_date, latest_date = selector.get_data_range()
    last_trading_date = selector.get_last_trading_day()
    last_trading_date = last_trading_date if last_trading_date else datetime.now()
    
    end_date = args.end if args.end else last_trading_date.strftime("%Y-%m-%d")
    
    engine = AlphaEngine(
        factor_calculator=V8FactorCalculator(),
        mlp_signals=MLPSignals(signal_name="temp", force_retrain=False),
        selector=selector,
        signal_name="temp",
        start_date=args.start,
        end_date=end_date
    )
    
    # 仅执行数据加载
    print(f"\n开始加载数据 ({args.start} 到 {end_date})...")
    data_df = engine.load_data()
    
    print(f"\n数据加载完成!")
    print(f"数据形状：{data_df.shape}")
    print(f"数据列数：{len(data_df.columns)}")
    
    # 保存列信息到文件
    save_column_info(data_df, args.output)
    
    # 打印前几列和后几列作为预览
    print("\n=== 列名预览 ===")
    cols = data_df.columns
    if len(cols) <= 20:
        for i, col in enumerate(cols):
            print(f"  [{i}] {col}")
    else:
        print("前 10 列:")
        for i in range(10):
            print(f"  [{i}] {cols[i]}")
        print("...")
        print("后 10 列:")
        for i in range(-10, 0):
            print(f"  [{len(cols) + i}] {cols[i]}")
