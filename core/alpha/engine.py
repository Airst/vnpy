"""
AlphaEngine — 因子选股系统编排器

== 当前状态 ==
职责: 数据同步 → 因子计算 → 滚动窗口IC分析 → MLP训练 → 信号生成
版本发现: 由 training.py 动态扫描 core/alpha/v*_factor_calculator.py 并注入
数据源: vnpy 数据库 → AlphaLab parquet 文件
选股宇宙: FundamentalSelector 过滤（EP>0, 换手率>1%, ln_cap>=11.5, 主板）

== 设计决策 ==
- 编排器模式: Engine 不持有版本逻辑，通过依赖注入接收 FactorCalculator 实例
- 3年数据窗口: start_date 默认回溯3年，保证训练窗口(700天)+评估窗口充足
- IC分析: 200天滚动窗口计算因子IC/ICIR，用于因子有效性监控（非训练用）
- 数据路径: core/alpha_db/ 下按 daily/model/signal/backtest 分目录存储
"""
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional
import polars as pl
from vnpy.trader.database import get_database
from vnpy.trader.constant import Interval, Exchange
from vnpy.alpha.lab import AlphaLab

from core.alpha.factor_calculator import FactorCalculator
from core.alpha.mlp_signals import MLPSignals
from core.selector import FundamentalSelector
from core.alpha.data_loader import DataLoader

ALPHA_DB_PATH = "core/alpha_db"

class AlphaEngine:
    def __init__(self, factor_calculator: FactorCalculator, mlp_signals: MLPSignals, selector: FundamentalSelector, signal_name: str, start_date: Optional[str] = None, end_date: Optional[str] = None):
        self.project_root = Path(os.getcwd())
        self.lab_path = self.project_root / ALPHA_DB_PATH
        self.lab = AlphaLab(str(self.lab_path))
        self.selector = selector
        self.factor_calculator = factor_calculator
        self.mlp_signals = mlp_signals
        self.signal_name = signal_name
        self.database = get_database()
        self.data_loader = DataLoader(self.lab)
        # 1. Configuration & Scope
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if not start_date:
            # Load enough history for training (e.g., 3 years)
            start_date = (datetime.now() - timedelta(days=365*3)).strftime("%Y-%m-%d")
        
        self.start_date = start_date
        self.end_date = end_date

    def sync_data(self, start_date: Optional[datetime] = None, end_date: datetime = datetime.now()):
        """
        Sync data from vnpy database to AlphaLab parquet files.
        """
        symbols = self.selector.get_candidate_symbols()
        if not symbols:
            print("[AlphaEngine] No symbols found in selector.")
            return

        # Determine overall range if not provided
        if not start_date:
            s, _ = self.selector.get_data_range()
            start_date = s if s else datetime(2020, 12, 24)
        
        print(f"[AlphaEngine] Syncing data for {len(symbols)} symbols from {start_date} to {end_date}...")

        for vt_symbol in symbols:
            symbol, exchange_str = vt_symbol.split(".")
            exchange = Exchange(exchange_str)
            
            bars = self.database.load_bar_data(
                symbol=symbol,
                exchange=exchange,
                interval=Interval.DAILY,
                start=start_date,
                end=end_date
            )
            
            if bars:
                self.lab.save_bar_data(bars)
        
        print("[AlphaEngine] Data sync complete.")

    def load_data(self) -> pl.DataFrame:
        """
        Load market data for A-share symbols.
        
        Returns:
            pl.DataFrame: Market data DataFrame with columns like datetime, vt_symbol, open, high, low, close, volume
        """
        print(f"[AlphaEngine] Range: {self.start_date} to {self.end_date}")

        # 1. Get Symbols
        symbols = self._get_ashare_symbols()
        if not symbols:
            print("[AlphaEngine] No symbols found.")
            raise ValueError("No A-share symbols found.")
        print(f"[AlphaEngine] Symbols: {len(symbols)}")
        
        # 2. Load Data
        df = self.data_loader.load_ashare_data(symbols, self.start_date, self.end_date)
        if df.is_empty():
            print("[AlphaEngine] No data loaded.")
            raise ValueError("No data loaded.")

        return df

    def calculate_factors(self, data_df: Optional[pl.DataFrame] = None) -> pl.DataFrame:
        """
        Calculate factors based on market data.
        
        Args:
            data_df: Optional market data DataFrame. If None, will load data internally.
            
        Returns:
            pl.DataFrame: Factor DataFrame
        """
        if data_df is None:
            data_df = self.load_data()

        factor_df = self.factor_calculator.calculate_features(data_df)

        return factor_df

    def analyze_factor_performance(self, factors_df: pl.DataFrame, threshold: float = 0.02) -> pl.DataFrame:
        """
        因子绩效分析（滚动窗口模式）
        
        Args:
            factors_df: 包含因子和label的DataFrame
            threshold: (仅用于展示高亮) IC绝对值阈值
            
        Returns:
            pl.DataFrame: 返回包含所有因子的原始DataFrame（不进行剔除，防止Look-ahead Bias）
        """
        print("=== 因子绩效分析 (Rolling Window: 200 Days) ===")
        
        if factors_df.is_empty():
            print("无因子数据可分析")
            return factors_df
            
        # 1. Deep Copy for Analysis (防止污染原始数据)
        df_analysis = factors_df.clone()
        
        # 准备数据：计算未来收益率 (5日)用于IC计算
        # print("计算未来5日收益率作为基准...")
        if "label" in df_analysis.columns:
             # Label is already calculated (normalized future return)
             df_calc = df_analysis.with_columns(pl.col("label").alias("next_ret"))
        elif "close" in df_analysis.columns:
            df_calc = df_analysis.with_columns([
                ((pl.col("close").shift(-5).over("vt_symbol") / pl.col("close")) - 1).alias("next_ret")
            ])
        else:
            print("无法计算未来收益率：缺少 'close' 或 'label' 列")
            return factors_df
        
        # 去除无效数据用于统计
        df_calc = df_calc.filter(pl.col("next_ret").is_not_null())
        
        if df_calc.is_empty():
            print("有效数据不足进行IC分析")
            return factors_df

        # 排除非因子列
        exclude_cols = ["datetime", "vt_symbol", "close", "open", "high", "low", "volume", "next_ret", "label"]
        # 确保只分析原始factors_df中存在的列
        factor_cols = [col for col in factors_df.columns if col not in exclude_cols]
        factor_cols.sort()
        
        print(f"正在分析 {len(factor_cols)} 个因子...")

        # --- Rolling Window Analysis ---
        # 1. Get unique dates
        dates = df_calc["datetime"].unique().sort()
        total_days = len(dates)
        window_size = 200
        
        results = {} # {factor_name: {period: (ic, icir)}}
        
        # Initialize results structure
        for f in factor_cols:
            results[f] = {}

        # Helper to calc stats
        def calc_stats(sub_df, period_name):
            if sub_df.is_empty():
                return
            
            # Calculate Rank IC per day
            ic_exprs = [
                pl.corr(pl.col(f).rank(), pl.col("next_ret").rank()).alias(f) 
                for f in factor_cols
            ]
            
            try:
                daily_ics = sub_df.group_by("datetime").agg(ic_exprs)
                
                # Mean IC & ICIR
                stats = daily_ics.select([
                    pl.col(f).fill_nan(None).mean().alias(f"{f}_mean") for f in factor_cols
                ] + [
                    pl.col(f).fill_nan(None).std().alias(f"{f}_std") for f in factor_cols
                ])
                
                stats_row = stats.row(0)
                cols = stats.columns
                
                for f in factor_cols:
                    mean_ic = stats_row[cols.index(f"{f}_mean")]
                    std_ic = stats_row[cols.index(f"{f}_std")]
                    
                    if mean_ic is None:
                        icir = 0.0
                        mean_ic = 0.0
                    else:
                        icir = mean_ic / (std_ic + 1e-9)
                    
                    # Store as string "IC (ICIR)"
                    results[f][period_name] = (mean_ic, icir)
                    
            except Exception as e:
                print(f"Error in period {period_name}: {e}")

        # 2. Iterate Windows
        num_windows = (total_days + window_size - 1) // window_size
        periods = []
        
        for i in range(num_windows):
            start_idx = i * window_size
            end_idx = min((i + 1) * window_size, total_days)
            
            start_date = dates[start_idx]
            end_date = dates[end_idx - 1] # inclusive
            
            period_name = f"{start_date.strftime('%y%m%d')}-{end_date.strftime('%y%m%d')}"
            periods.append(period_name)
            
            # Filter Data
            # Polars filter by date range
            sub_df = df_calc.filter((pl.col("datetime") >= start_date) & (pl.col("datetime") <= end_date))
            
            calc_stats(sub_df, period_name)
            
        # 3. Overall Stats
        calc_stats(df_calc, "Overall")
        periods.append("Overall")

        # --- Build Result Table ---
        # Columns: Factor, Period1_IC, Period1_ICIR, ..., Overall_IC, Overall_ICIR
        # Simplified: Just show IC (and maybe color code or format string)
        # User asked for a table. Let's make columns: Factor, [Period Name IC/ICIR]...
        
        # Let's separate IC and ICIR tables or combine them?
        # A combined string "0.05 (0.5)" is compact.
        
        table_data = []
        for f in factor_cols:
            row = {"Factor": f}
            overall_ic = results[f].get("Overall", (0,0))[0]
            row["_sort_key"] = abs(overall_ic) # Helper for sorting
            
            for p in periods:
                val = results[f].get(p, None)
                if val:
                    ic, icir = val
                    row[p] = f"{ic:.3f} ({icir:.2f})"
                else:
                    row[p] = "-"
            table_data.append(row)
            
        # Sort by Overall IC magnitude
        table_data.sort(key=lambda x: x["_sort_key"], reverse=True)
        
        # Convert to Polars for display
        final_cols = ["Factor"] + periods
        display_data = []
        for r in table_data:
            dr = {k: r[k] for k in final_cols}
            display_data.append(dr)
            
        if display_data:
            res_df = pl.DataFrame(display_data)
            
            # Adjust Polars display settings to show all columns
            with pl.Config(
                tbl_rows=100, 
                tbl_cols=len(periods)+1, 
                fmt_str_lengths=20,
                tbl_width_chars=200
            ):
                print(res_df)
                
            # Print Summary of Top 5
            print("\n[Top 5 Factors (Overall |IC|)]")
            for i in range(min(5, len(display_data))):
                f = display_data[i]["Factor"]
                v = display_data[i]["Overall"]
                print(f"  {f}: {v}")
        else:
            print("No results generated.")

        # Cleanup intermediate large objects
        del df_analysis
        del df_calc
        del table_data
        del display_data
        
        return factors_df

    def calculate_signals(self, factor_df: pl.DataFrame) -> pl.DataFrame:
        """
        Generate signals using ML models based on factor DataFrame.
        """
        signal_df = self.mlp_signals.generate_signals(factor_df, self.start_date, self.lab)
        return signal_df

    def get_signal_df(self, name: str) -> Optional[pl.DataFrame]:
        return self.lab.load_signal(name)


    def _get_ashare_symbols(self) -> List[str]:
        """获取A股标的"""
        all_symbols = self.selector.get_candidate_symbols()
        
        # A股代码过滤规则
        ashare_symbols = []
        for symbol in all_symbols:
            # 只保留沪深A股（代码以特定前缀开头）
            if any(symbol.startswith(prefix) for prefix in ['000', '002', '300', '600', '601', '603', '688']):
                ashare_symbols.append(symbol)
        
        return ashare_symbols

    def save_signals(self, signal_df):
        if signal_df is not None:
            print(f"[AlphaEngine] Saving signals to '{self.signal_name}'...")
            
            # Try to load existing signals
            existing_df = self.lab.load_signal(self.signal_name)
            
            if existing_df is not None and not existing_df.is_empty():
                print(f"[AlphaEngine] Found existing signals ({len(existing_df)} rows). Merging...")
                
                # Determine the start date of the new signals
                if "datetime" in signal_df.columns and not signal_df.is_empty():
                    last_existing_dt = existing_df["datetime"].max()
                    
                    # Filter new signals: keep those strictly after the last existing date
                    new_signals_to_append = signal_df.filter(pl.col("datetime") > last_existing_dt)

                    if not new_signals_to_append.is_empty():
                        print(f"[AlphaEngine] Appending {len(new_signals_to_append)} new rows (after {last_existing_dt}).")
                        signal_df = pl.concat([existing_df, new_signals_to_append])
                    else:
                        print(f"[AlphaEngine] No new signals to append. Last existing: {last_existing_dt}.")
                        signal_df = existing_df
                    
                    # Sort by datetime and symbol
                    if "vt_symbol" in signal_df.columns:
                        signal_df = signal_df.sort(["datetime", "vt_symbol"])
                    else:
                        signal_df = signal_df.sort("datetime")
                else:
                    print("[AlphaEngine] New signal dataframe is empty or missing datetime. Skipping merge.")

            self.lab.save_signal(self.signal_name, signal_df)
            print("[AlphaEngine] Saved.")