# 系统信息
- 当前操作系统是WSL Ubuntu
- python执行需要用全路径：/home/airst/Workspace/.venv/bin/python

# vnpy项目介绍

## 项目目的
- 构建一套基于vnpy的A股量化分析系统，基于vnpy框架中的MlpModel进行模型训练，为日线交易提供信号输入，以持续获取最大化收益

## 文件目录
- core：本系统的核心代码
- core/aplha：用于因子计算和模型训练
## 当前进展
- 当前因子计算已经从v3迭代至v8（v8_factor_calculator）

## 量化思路
- 量化的内核是寻找市场波动的规律，从波动中获利
- 要挖掘因子让MLP模型能发现市场波动规律，适应市场风格；不要使用复杂的参数来时某一个因子适应某种特定的市场风格，避免过拟合

## Added Memories
- V8FactorCalculator has been updated with Market Style Factors (SMB, HML, UMD, Volatility Spreads) and Style Interactions to support Regime Adaptive Learning.