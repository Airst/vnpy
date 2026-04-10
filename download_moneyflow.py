#!/usr/bin/env python
"""
下载资金流向数据脚本
用于获取tushare个股资金流向数据，支持主力资金Alpha因子计算
"""
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_manager.ts_downloader.moneyflow_manager import MoneyFlowManager


def main():
    print("=" * 60)
    print("资金流向数据下载工具 (MoneyFlow Downloader)")
    print("=" * 60)
    print()
    print("数据来源: tushare moneyflow接口")
    print("用途: 计算主力资金Alpha因子 (V9.7)")
    print()
    
    mf = MoneyFlowManager()
    mf.download_all()
    
    print()
    print("=" * 60)
    print("下载完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
