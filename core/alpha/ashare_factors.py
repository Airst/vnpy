import warnings
warnings.filterwarnings('ignore')
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List, Optional
import polars as pl

# GPU支持判断
try:
    import cudf
    import cupy as cp
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

class AShareFactorCalculator:
    """A股专用因子计算器"""
    
    def __init__(self, engine):
        self.engine = engine
        self.factor_weights = {
            'momentum': 0.25,
            'value': 0.20,
            'quality': 0.20,
            'growth': 0.15,
            'liquidity': 0.10,
            'sentiment': 0.10
        }
        
    def calculate_all_factors(self, start_date=None, end_date=None):
        """计算所有A股因子"""
        # 1. 获取股票列表（A股专用）
        symbols = self.get_ashare_symbols()
        
        if not symbols:
            print("未找到A股标的")
            return None
        
        # 2. 设置合理的时间范围
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365*3)).strftime("%Y-%m-%d")
        
        print(f"计算因子数据范围: {start_date} 到 {end_date}")
        print(f"股票数量: {len(symbols)}")
        
        # 3. 加载数据
        df = self.load_ashare_data(symbols, start_date, end_date)
        
        if df.is_empty():
            print("数据加载失败")
            return None
        
        # 4. 计算各维度因子
        factors_df = self.compute_factor_matrix(df)
        
        # 4.1 分析因子表现
        self.analyze_factor_performance(factors_df)
        
        # 5. 因子合成与处理
        final_signal = self.synthesize_signals(factors_df)
        
        # 6. 保存结果
        self.save_factors(factors_df, final_signal)
        
        return final_signal
    
    def get_ashare_symbols(self):
        """获取A股标的（过滤ST、次新股等）"""
        all_symbols = self.engine.selector.get_candidate_symbols()
        
        # A股代码过滤规则
        ashare_symbols = []
        for symbol in all_symbols:
            # 只保留沪深A股（代码以特定前缀开头）
            if any(symbol.startswith(prefix) for prefix in ['000', '002', '300', '600', '601', '603', '688']):
                ashare_symbols.append(symbol)
        
        return ashare_symbols
    
    def load_ashare_data(self, symbols, start_date, end_date):
        """加载A股数据，包含财务数据和行情数据"""
        # 扩展天数考虑A股交易特点
        extended_days = 250  # 考虑A股年线计算
        
        print(f"加载数据，扩展天数: {extended_days}")
        
        # 1. 加载行情数据
        price_df = self.engine.lab.load_bar_df(
            vt_symbols=symbols,
            interval="d",
            start=start_date,
            end=end_date,
            extended_days=extended_days
        )
        
        if price_df is None or price_df.is_empty():
            return pl.DataFrame()
        
        # 2. 加载财务数据（如果有的话）
        financial_df = self.load_financial_data(symbols, end_date)
        
        # 3. 合并数据
        if not financial_df.is_empty():
            # 对财务数据进行前向填充（季度数据填充到日频）
            financial_df = financial_df.sort(["vt_symbol", "report_date"])
            price_df = price_df.join(
                financial_df,
                on=["vt_symbol", "datetime"],
                how="left"
            )
            
            # 前向填充财务数据
            price_df = price_df.sort(["vt_symbol", "datetime"])
            price_df = price_df.with_columns([
                pl.col(col).forward_fill().over("vt_symbol")
                for col in financial_df.columns if col not in ["vt_symbol", "datetime"]
            ])
        
        return price_df
    
    def load_financial_data(self, symbols, end_date):
        """模拟加载财务数据（实际使用时需要接入财务数据库）"""
        # 这里返回一个空DataFrame，实际使用时需要实现
        return pl.DataFrame()
    
    def compute_factor_matrix(self, df):
        """计算因子矩阵"""
        print("开始计算因子矩阵...")
        
        # 转换为GPU DataFrame（如果可用）
        if HAS_GPU:
            gdf = cudf.DataFrame.from_pandas(df.to_pandas())
            return self.compute_factors_gpu(gdf)
        else:
            return self.compute_factors_cpu(df)
    
    def compute_factors_cpu(self, df):
        """CPU版本因子计算"""
        # 确保数据按时间和股票排序
        df = df.sort(["vt_symbol", "datetime"])
        
        # 直接对全量数据进行计算，利用 .over("vt_symbol") 实现分组计算
        # 计算价格相关因子
        df = self.compute_price_factors(df)
        
        # 计算量价因子
        df = self.compute_volume_price_factors(df)
        
        # 计算技术指标因子
        df = self.compute_technical_factors(df)
        
        # 计算风险因子
        df = self.compute_risk_factors(df)
        
        # 计算横截面因子
        cross_sectional_factors = self.compute_cross_sectional_factors(df)
        
        return cross_sectional_factors
    
    def compute_price_factors(self, df):
        """价格类因子"""
        # 基础收益率
        df = df.with_columns([
            pl.col("close").shift(1).over("vt_symbol").alias("prev_close"),
            pl.col("close").shift(5).over("vt_symbol").alias("close_5d"),
            pl.col("close").shift(20).over("vt_symbol").alias("close_20d"),
            pl.col("close").shift(60).over("vt_symbol").alias("close_60d"),
            pl.col("close").shift(120).over("vt_symbol").alias("close_120d"),
        ])
        
        # 动量因子（不同周期）
        df = df.with_columns([
            ((pl.col("close") / pl.col("close_5d")) - 1).alias("mom_5d"),
            ((pl.col("close") / pl.col("close_20d")) - 1).alias("mom_20d"),
            ((pl.col("close") / pl.col("close_60d")) - 1).alias("mom_60d"),
            ((pl.col("close") / pl.col("close_120d")) - 1).alias("mom_120d"),
        ])
        
        # 反转因子（短期）
        df = df.with_columns([
            (-pl.col("mom_5d")).alias("rev_5d"),
            ((pl.col("close") / pl.col("close").shift(1).over("vt_symbol")) - 1).alias("ret_1d"),
        ])
        
        # 价格趋势
        df = df.with_columns([
            (pl.col("close") - pl.col("close").rolling_mean(20).over("vt_symbol")) 
            / pl.col("close").rolling_std(20).over("vt_symbol").alias("price_zscore_20d"),
        ])
        
        
        return df
    
    def compute_volume_price_factors(self, df):
        """量价结合因子（A股特色）"""
        # 资金流向因子
        df = df.with_columns([
            # 量比
            (pl.col("volume") / pl.col("volume").rolling_mean(20).over("vt_symbol")).alias("volume_ratio"),
            
            # 资金流向（简化版）
            (((pl.col("close") - pl.col("open")) / (pl.col("high") - pl.col("low") + 1e-8)) 
             * pl.col("volume")).alias("money_flow_raw"),
        ])
        
        # 标准化资金流向
        df = df.with_columns([
            (pl.col("money_flow_raw") / pl.col("money_flow_raw").rolling_mean(20).over("vt_symbol"))
            .alias("money_flow_20d"),
        ])
        
        # 量价背离检测
        df = df.with_columns([
            pl.when(
                (pl.col("close") > pl.col("close").shift(1).over("vt_symbol")) &
                (pl.col("volume") < pl.col("volume").shift(1).over("vt_symbol"))
            ).then(1).otherwise(0).alias("price_volume_divergence"),
        ])
        
        return df
    
    def compute_technical_factors(self, df):
        """技术指标因子"""
        # 移动平均线因子 - Split to avoid potential Polars optimizer issues
        df = df.with_columns(pl.col("close").rolling_mean(5).over("vt_symbol").alias("ma_5"))
        df = df.with_columns(pl.col("close").rolling_mean(10).over("vt_symbol").alias("ma_10"))
        df = df.with_columns(pl.col("close").rolling_mean(20).over("vt_symbol").alias("ma_20"))
        df = df.with_columns(pl.col("close").rolling_mean(60).over("vt_symbol").alias("ma_60"))
        
        # 均线排列
        df = df.with_columns([
            ((pl.col("ma_5") > pl.col("ma_10")) & 
             (pl.col("ma_10") > pl.col("ma_20"))).cast(pl.Int8).alias("ma_alignment"),
        ])
        
        # 布林带
        rolling_mean = pl.col("close").rolling_mean(20).over("vt_symbol")
        rolling_std = pl.col("close").rolling_std(20).over("vt_symbol")
        
        df = df.with_columns([
            (pl.col("close") - rolling_mean) / (2 * rolling_std).alias("bollinger_position"),
        ])
        
        # RSI（简化版）
        # Calculate returns first as explicit column
        df = df.with_columns(pl.col("close").diff().over("vt_symbol").alias("diff_1d"))
        
        # Calculate gain/loss
        df = df.with_columns([
            pl.col("diff_1d").clip(0, None).alias("gain"),
            (-pl.col("diff_1d")).clip(0, None).alias("loss"),
        ])
        
        # Calculate avg gain/loss
        df = df.with_columns([
            pl.col("gain").rolling_mean(14).over("vt_symbol").alias("avg_gain"),
            pl.col("loss").rolling_mean(14).over("vt_symbol").alias("avg_loss"),
        ])
        
        # Calculate RSI
        df = df.with_columns([
            (100 - (100 / (1 + pl.col("avg_gain") / (pl.col("avg_loss") + 1e-8)))).alias("rsi_14"),
        ])
        
        return df
    
    def compute_risk_factors(self, df):
        """风险类因子"""
        # 计算日收益率
        df = df.with_columns([
            pl.col("close").pct_change().over("vt_symbol").alias("daily_return"),
        ])
        
        # 波动率因子
        df = df.with_columns([
            pl.col("daily_return").rolling_std(20).over("vt_symbol").alias("volatility_20d"),
            pl.col("daily_return").rolling_std(60).over("vt_symbol").alias("volatility_60d"),
            (pl.col("high") / pl.col("low") - 1).alias("daily_range"),
        ])
        
        # Beta计算（需要市场数据，这里简化）
        # 最大回撤
        rolling_max = pl.col("close").rolling_max(20).over("vt_symbol")
        df = df.with_columns([
            ((pl.col("close") - rolling_max) / rolling_max * 100).alias("drawdown_20d"),
        ])
        
        return df
    
    def compute_cross_sectional_factors(self, df):
        """横截面因子处理"""
        print("计算横截面因子...")
        
        results = []
        
        # 按日期分组计算
        dates = df["datetime"].unique().sort()
        
        for date in dates:
            date_df = df.filter(pl.col("datetime") == date)
            
            if date_df.height < 10:  # 至少10只股票
                continue
            
            # 对每个因子进行横截面处理
            factor_columns = [
                'mom_5d', 'mom_20d', 'mom_60d', 'mom_120d',
                'rev_5d', 'price_zscore_20d',
                'volume_ratio', 'money_flow_20d',
                'ma_alignment', 'bollinger_position', 'rsi_14',
                'volatility_20d', 'drawdown_20d'
            ]
            
            # 实际存在的列
            existing_cols = [col for col in factor_columns if col in date_df.columns]
            
            processed_factors = {}
            
            for factor in existing_cols:
                values = date_df[factor]
                
                # 移除极端值（3个标准差之外）
                mean_val = values.mean()
                std_val = values.std()
                
                if mean_val is not None and std_val is not None and std_val > 0:
                    # Winsorize处理
                    lower = mean_val - 3 * std_val
                    upper = mean_val + 3 * std_val
                    
                    winsorized = values.clip(lower, upper)
                    
                    # Z-score标准化
                    zscore = (winsorized - winsorized.mean()) / (winsorized.std() + 1e-8)
                    
                    # 行业中性化（如果有行业数据）
                    # 这里需要接入行业分类数据
                    
                    processed_factors[factor] = zscore
                else:
                    processed_factors[factor] = pl.Series([0.0] * len(values))
            
            # 创建处理后的DataFrame
            processed_df = pl.DataFrame({
                "datetime": [date] * len(date_df),
                "vt_symbol": date_df["vt_symbol"],
                "close": date_df["close"],
                **processed_factors
            })
            
            results.append(processed_df)
        
        if results:
            return pl.concat(results)
        else:
            return pl.DataFrame()
    
    def compute_factors_gpu(self, gdf):
        """GPU版本因子计算（使用cuDF）"""
        try:
            # 按股票排序
            gdf = gdf.sort_values(["vt_symbol", "datetime"])
            gdf = gdf.reset_index(drop=True)
            
            # 分组计算
            grouped = gdf.groupby("vt_symbol")
            
            # 动量因子
            for window in [1, 5, 20, 60, 120]:
                gdf[f'close_lag_{window}'] = grouped['close'].shift(window)
                gdf[f'mom_{window}d'] = (gdf['close'] / gdf[f'close_lag_{window}']) - 1
            
            # 反转因子
            gdf['rev_5d'] = -gdf['mom_5d']
            
            # 波动率
            gdf['ret_1d'] = grouped['close'].pct_change()
            # rolling操作返回MultiIndex，需要reset_index去除group层级以对齐原DataFrame
            gdf['volatility_20d'] = gdf.groupby("vt_symbol")['ret_1d'].rolling(20).std().reset_index(level=0, drop=True)
            
            # 量比
            gdf['volume_ma20'] = grouped['volume'].rolling(20).mean().reset_index(level=0, drop=True)
            gdf['volume_ratio'] = gdf['volume'] / gdf['volume_ma20']
            
            # 技术指标
            gdf['ma_5'] = grouped['close'].rolling(5).mean().reset_index(level=0, drop=True)
            gdf['ma_20'] = grouped['close'].rolling(20).mean().reset_index(level=0, drop=True)
            gdf['ma_60'] = grouped['close'].rolling(60).mean().reset_index(level=0, drop=True)
            
            # 均线排列
            gdf['ma_alignment'] = ((gdf['ma_5'] > gdf['ma_20']) & 
                                   (gdf['ma_20'] > gdf['ma_60'])).astype('int8')
            
            # 布林带位置
            rolling_std = grouped['close'].rolling(20).std().reset_index(level=0, drop=True)
            gdf['bollinger_position'] = (gdf['close'] - gdf['ma_20']) / (2 * rolling_std)
            
            # 转换为Polars
            result_df = pl.from_pandas(gdf.to_pandas())
            
            # 计算横截面
            return self.compute_cross_sectional_factors(result_df)
            
        except Exception as e:
            print(f"GPU计算失败: {e}")
            import traceback
            traceback.print_exc()
            # 回退到CPU计算
            return self.compute_factors_cpu(pl.from_pandas(gdf.to_pandas()))
    
    def synthesize_signals(self, factors_df):
        """
        Synthesize signals using AlphaLab MLP Model (Machine Learning).
        """
        if factors_df.is_empty():
            return None
        
        print("Synthesizing signals using MLP Model...")
        
        # 1. Imports (Lazy import to avoid circular dependencies or import errors if not set up)
        try:
            from vnpy.alpha.model.models.mlp_model import MlpModel
            from vnpy.alpha.dataset import AlphaDataset, Segment
        except ImportError as e:
            print(f"Import failed: {e}. Fallback to simple synthesis.")
            return self.synthesize_signals_simple(factors_df)

        # 2. Prepare Data and Label
        # Calculate Label: 5-day future return (Target for ML)
        # Note: We use shift(-5) to get future price. 
        # For training, we need valid labels. For latest inference, label will be null (which is fine for prediction).
        df = factors_df.with_columns([
            ((pl.col("close").shift(-5).over("vt_symbol") / pl.col("close")) - 1).alias("label")
        ])
        
        # Define Features (exclude metadata and label)
        # Exclude intermediate columns if any
        exclude_cols = ["datetime", "vt_symbol", "close", "open", "high", "low", "volume", "label", "total_score", "final_signal"]
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        
        print(f"Features ({len(feature_cols)}): {feature_cols}")
        
        # 3. Define Periods for Dataset
        # Sort by date
        dates = df["datetime"].unique().sort()
        if len(dates) < 20:
            print("Data too short for ML.")
            return self.synthesize_signals_simple(factors_df)
            
        start_date = dates[0]
        end_date = dates[-1]
        
        # Simple split: 70% Train, 20% Valid, 10% Test (which we will use for full inference mostly)
        # But actually, for "Production" inference, we might want to train on ALL available history (up to recent).
        # For this research script, let's use a fixed split to demonstrate workflow.
        
        total_days = len(dates)
        train_end_idx = int(total_days * 0.7)
        valid_end_idx = int(total_days * 0.9)
        
        train_period = (dates[0].strftime("%Y-%m-%d"), dates[train_end_idx].strftime("%Y-%m-%d"))
        valid_period = (dates[train_end_idx+1].strftime("%Y-%m-%d"), dates[valid_end_idx].strftime("%Y-%m-%d"))
        test_period = (dates[valid_end_idx+1].strftime("%Y-%m-%d"), dates[-1].strftime("%Y-%m-%d"))
        
        print(f"Train: {train_period}, Valid: {valid_period}, Test: {test_period}")
        
        # 4. Create AlphaDataset
        # We need to ensure 'df' passed to AlphaDataset has feature columns + label + datetime + vt_symbol
        # The AlphaDataset expects data to be ready or calculates it via expressions. 
        # Here we already calculated features in 'df'.
        
        # IMPORTANT: AlphaDataset usually expects raw data and calculates features via expressions.
        # But here we pass a DF that HAS features. 
        # We need to register these columns as features in the dataset.
        
        dataset = AlphaDataset(
            df=df,
            train_period=train_period,
            valid_period=valid_period,
            test_period=test_period,
            process_type="append"
        )
        
        # Add feature columns to dataset
        for col in feature_cols:
            dataset.add_feature(name=col, result=pl.DataFrame()) # Hack: 'result' arg is not used like this usually?
            # Wait, dataset.add_feature with 'result' expects a DF with "datetime", "vt_symbol", "data".
            # This is inefficient to split 'df' into many DFs.
            # But AlphaDataset.prepare_data uses these.
            # actually, if we pass 'df' that ALREADY contains features, we might not need prepare_data 
            # IF we manually set learn_df/infer_df?
            # Let's look at AlphaDataset code. 
            # prepare_data calculates features and joins them.
            # If our 'df' passed to __init__ IS the data we want, we can skip prepare_data?
            # But prepare_data sets result_df, raw_df, learn_df.
            
            # WORKAROUND: We will simulate the dataset internal state.
            pass
            
        # Manually setup dataset internals since we already computed features
        # Filter columns to only what we need
        keep_cols = ["datetime", "vt_symbol"] + feature_cols + ["label"]
        dataset.result_df = df.select([c for c in keep_cols if c in df.columns])
        dataset.raw_df = dataset.result_df
        dataset.learn_df = dataset.result_df
        dataset.infer_df = dataset.result_df
        
        # Add Processors
        # 1. Drop NaN labels for training (Learning phase)
        # 2. Normalize features (CS Rank Norm is good for neural nets on financial data)
        # 3. Fill NaN features (median or 0)
        
        def processor_setup(df):
            # Custom processor pipeline
            # 1. Fill NaN features
            df = df.with_columns([
                pl.col(c).fill_null(0) for c in feature_cols
            ])
            # 2. Rank Norm (Cross Sectional)
            # Need to implement or use utility. using existing function requires adapter
            # Simple Z-score per day
            for col in feature_cols:
                mean = pl.col(col).mean().over("datetime")
                std = pl.col(col).std().over("datetime")
                df = df.with_columns(
                    ((pl.col(col) - mean) / (std + 1e-8)).clip(-3, 3).alias(col)
                )
            return df
            
        dataset.add_processor("learn", processor_setup)
        dataset.add_processor("infer", processor_setup)
        
        # Process Data
        dataset.process_data()
        
        # 5. Initialize and Train MLP Model
        print("Training MLP Model...")
        device = "cuda" if HAS_GPU else "cpu"
        print(f"Device: {device}")
        
        model = MlpModel(
            input_size=len(feature_cols),
            hidden_sizes=(256, 128,),
            n_epochs=500, # Reduced for speed in demo
            batch_size=4096,
            lr=0.001,
            device=device,
            seed=42,
            optimizer="adam",
        )
        
        try:
            model.fit(dataset)
        except Exception as e:
            print(f"Training failed: {e}")
            import traceback
            traceback.print_exc()
            return self.synthesize_signals_simple(factors_df)
            
        # 6. Predict (Synthesize)
        print("Generating predictions...")
        
        # We want to predict for the WHOLE dataset (or at least the end part).
        # AlphaLab predict takes a Segment. 
        # We can hijack Segment.TEST to cover what we want, OR use _predict_batch directly if we can access data.
        # Let's use Segment.TEST which we defined as the last part.
        # But for 'calculate_all_factors', we usually want signals for the WHOLE period? 
        # Or at least the requested period.
        
        # Let's assume we want to predict on the Valid+Test period (out of sample-ish) 
        # plus maybe Train if we want to see fit.
        # Ideally, we construct a "PREDICT" segment covering start_date to end_date.
        # Hack: modify data_periods temporarily
        
        # Actually, let's just use the internal model predict on the whole processed df
        # The dataset.infer_df contains all data (processed).
        
        # Get all features
        full_data = dataset.infer_df.select(feature_cols).to_numpy()
        import torch
        
        # Use internal predict
        # model._predict_batch expects tensor
        # But model is inside MlpModel.
        # MlpModel has .predict(dataset, segment).
        
        # Let's just update TEST period to cover everything for prediction convenience
        dataset.data_periods[Segment.TEST] = (start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        
        predictions = model.predict(dataset, Segment.TEST)
        
        # 7. Construct Result DataFrame
        # Get the corresponding metadata
        pred_df = dataset.fetch_infer(Segment.TEST).select(["datetime", "vt_symbol"])
        
        # Add predictions (flattened)
        # Predictions are numpy array.
        # Check length alignment
        if len(predictions) != len(pred_df):
            print(f"Prediction length mismatch! Pred: {len(predictions)}, DF: {len(pred_df)}")
            return self.synthesize_signals_simple(factors_df)
            
        final_df = pred_df.with_columns(
            pl.Series(predictions).alias("raw_pred")
        )
        
        # Normalize predictions to -3 to 3 range (Signal format)
        final_df = final_df.with_columns([
            pl.col("raw_pred").rank(method="average").over("datetime").alias("rank"),
            pl.col("raw_pred").count().over("datetime").alias("count")
        ])
        
        final_df = final_df.with_columns([
            (((pl.col("rank") / pl.col("count")) - 0.5) * 3.46)
            .clip(-3, 3)
            .alias("final_signal")
        ])
        
        # Join back with original factors if needed, or just return signal
        # We should return 'total_score' (maybe raw pred) and 'final_signal'
        # 'total_score' is the raw prediction (return), 'final_signal' is the rank-normalized score
        final_df = final_df.with_columns([
            pl.col("raw_pred").alias("total_score")
        ])
        
        return final_df.select(["datetime", "vt_symbol", "total_score", "final_signal"])

    def synthesize_signals_simple(self, factors_df):
        """合成最终信号 (Original Simple Logic)"""
        if factors_df.is_empty():
            return None
        
        print("合成最终信号 (Simple)...")
        
        # 因子分组
        factor_groups = {
            'momentum': ['mom_20d', 'mom_60d', 'mom_120d'],
            'reversal': ['rev_5d', 'price_zscore_20d'],
            'volume': ['volume_ratio', 'money_flow_20d'],
            'technical': ['ma_alignment', 'bollinger_position', 'rsi_14'],
            'risk': ['volatility_20d', 'drawdown_20d']
        }
        
        # 复制一份数据以避免修改原始数据
        df = factors_df.clone()
        
        # 初始化总分
        df = df.with_columns(pl.lit(0.0).alias("total_score"))
        
        # 计算各组得分并累加
        for group, factors in factor_groups.items():
            # 筛选存在的因子
            existing = [f for f in factors if f in df.columns]
            if not existing:
                continue
            
            # 计算组内平均分
            # 使用 sum 累加表达式
            group_sum_expr = pl.col(existing[0]).fill_null(0)
            for f in existing[1:]:
                group_sum_expr = group_sum_expr + pl.col(f).fill_null(0)
                
            group_score_expr = group_sum_expr / len(existing)
            
            # 添加组得分列
            group_score_col = f"{group}_score"
            df = df.with_columns(group_score_expr.alias(group_score_col))
            
            # 加权累加到总分
            weight = self.factor_weights.get(group, 0.15)
            df = df.with_columns((pl.col("total_score") + pl.col(group_score_col) * weight).alias("total_score"))
            
        # 按日期对得分进行横截面排名
        df = df.with_columns([
            pl.col("total_score").rank(method="average").over("datetime").alias("rank"),
            pl.col("total_score").count().over("datetime").alias("count")
        ])
        
        # 转换为标准正态分布
        df = df.with_columns([
            (((pl.col("rank") / pl.col("count")) - 0.5) * 3.46)
            .clip(-3, 3)
            .alias("final_signal")
        ])
        
        # 只保留需要的列
        keep_cols = ["datetime", "vt_symbol", "total_score", "final_signal"] + \
                   [f"{g}_score" for g in factor_groups.keys() if f"{g}_score" in df.columns]
        
        # 确保列存在
        final_cols = [c for c in keep_cols if c in df.columns]
        
        return df.select(final_cols)
    
    def save_factors(self, factors_df, signal_df):
        """保存因子数据"""
        if signal_df is not None:
            # 保存最终信号
            print("保存最终信号...")
            self.engine.lab.save_signal("ashare_multi_factor", signal_df)
            print(f"信号保存成功，数据量: {len(signal_df)}")
            
            # 保存详细因子数据（可选）
            # factors_df.write_parquet("./factor_data.parquet")
            # print("详细因子数据已保存")
        else:
            print("信号数据为空，保存失败")
    
    def analyze_factor_performance(self, factors_df):
        """因子绩效分析（简化版）"""
        print("\n=== 因子绩效分析 ===")
        
        if factors_df.is_empty():
            print("无因子数据可分析")
            return
        
        # 准备数据：计算未来收益率 (5日)
        print("计算未来5日收益率作为基准...")
        df = factors_df.with_columns([
            ((pl.col("close").shift(-5).over("vt_symbol") / pl.col("close")) - 1).alias("next_ret")
        ])
        
        # 去除无效数据
        df = df.filter(pl.col("next_ret").is_not_null())
        
        if df.is_empty():
            print("有效数据不足")
            return

        # 分析各因子的IC（信息系数）
        # 排除非因子列
        exclude_cols = ["datetime", "vt_symbol", "close", "open", "high", "low", "volume", "next_ret", "label"]
        factor_cols = [col for col in df.columns if col not in exclude_cols]
        
        print(f"分析 {len(factor_cols)} 个因子的IC (Rank IC)...")
        
        ic_results = []
        
        for factor in factor_cols:
            try:
                # 计算 Rank IC (Spearman Correlation)
                # Polars corr 默认为 Pearson，先 rank 再 corr 近似 Spearman
                ic = df.select(
                    pl.corr(
                        pl.col(factor).rank(), 
                        pl.col("next_ret").rank()
                    )
                ).item()
                
                ic_results.append({"factor": factor, "ic": ic})
                print(f"  - {factor}: IC = {ic:.4f}")
            except Exception as e:
                print(f"  - {factor}: 计算失败 ({e})")
        
        # 简单总结
        if ic_results:
            avg_ic = sum(r["ic"] for r in ic_results if r["ic"] is not None) / len(ic_results)
            print(f"\n平均 IC: {avg_ic:.4f}")
        
        print("分析完成")

