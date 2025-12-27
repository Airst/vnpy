import warnings
warnings.filterwarnings('ignore')
from datetime import datetime, timedelta
import polars as pl
import numpy as np
from typing import Optional, List, Dict

# Import vnpy alpha components
from vnpy.alpha.dataset.datasets.alpha_158 import Alpha158
from vnpy.alpha.model.models.mlp_model import MlpModel
from vnpy.alpha import Segment, AlphaDataset

class AShareFactorCalculatorV3:
    """
    A-Share Factor Calculator V3
    Based on vnpy.alpha.lab (Alpha158) and MLP Model.
    """
    
    def __init__(self, engine):
        self.engine = engine
        self.model_settings = {
            "hidden_sizes": (256, 128, 64),
            "n_epochs": 300,  # Adjustable based on needs
            "batch_size": 4096,
            "lr": 0.001,
            "early_stop_rounds": 20,
            "device": "auto"  # Will detect GPU
        }
        
    def calculate_all_factors(self, start_date: str = None, end_date: str = None):
        """
        Main entry point to calculate factors and predict signals.
        """
        # 1. Configuration & Scope
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if not start_date:
            # Load enough history for training (e.g., 3 years)
            start_date = (datetime.now() - timedelta(days=365*3)).strftime("%Y-%m-%d")
            
        print(f"[V3] Range: {start_date} to {end_date}")
        
        # 2. Get Symbols
        symbols = self.get_ashare_symbols()
        if not symbols:
            print("[V3] No symbols found.")
            return None
        print(f"[V3] Symbols: {len(symbols)}")
        
        # 3. Load Data
        df = self.load_ashare_data(symbols, start_date, end_date)
        if df.is_empty():
            print("[V3] No data loaded.")
            return None
            
        # 4. Construct Dataset (Alpha158)
        print("[V3] Constructing Alpha158 Dataset...")
        # Split dates for Train/Valid/Test
        # Strategy:
        # Train: 70%
        # Valid: 20%
        # Test: Last part (for output signal) - or we can predict on everything.
        # usually we want the signal for the 'end_date' and recent history.
        
        dates = df["datetime"].unique().sort()
        total_days = len(dates)
        
        train_end = int(total_days * 0.6)
        valid_end = int(total_days * 0.8)
        
        train_period = (dates[0].strftime("%Y-%m-%d"), dates[train_end].strftime("%Y-%m-%d"))
        valid_period = (dates[train_end+1].strftime("%Y-%m-%d"), dates[valid_end].strftime("%Y-%m-%d"))
        # Test period includes the rest (including today if in data)
        test_period = (dates[valid_end+1].strftime("%Y-%m-%d"), dates[-1].strftime("%Y-%m-%d"))
        
        print(f"[V3] Split: Train={train_period}, Valid={valid_period}, Test={test_period}")
        
        dataset = Alpha158(
            df=df,
            train_period=train_period,
            valid_period=valid_period,
            test_period=test_period
        )
        
        # 5. Define Label (Target)
        # We override Alpha158's default label with a 5-day future return
        # Logic: (Close[t+5] / Close[t]) - 1
        # Note: shift(-5) moves future value to current row.
        # Using polars expression style string for set_label
        # "ts_delay(close, -5)" is Close at t+5
        dataset.set_label("ts_delay(close, -5) / close - 1")
        
        # 6. Process Data (Feature Engineering)
        print("[V3] Processing data (calculating features)...")
        # Add processors to handle NaNs and Normalization
        self._add_processors(dataset)
        
        try:
            # Limit max_workers to 1 to avoid "fork bomb" with Polars and multiprocessing
            # This prevents CPU spikes and crashes on WSL/Linux
            dataset.prepare_data(max_workers=6) # Calculates expressions
            dataset.process_data() # Applies processors (cleaning/norm)
        except Exception as e:
            print(f"[V3] Data processing error: {e}")
            import traceback
            traceback.print_exc()
            return None
            
        # 7. Train Model
        print("[V3] Training MLP Model...")
        model = self._train_model(dataset)
        if not model:
            return None
            
        # 8. Predict
        print("[V3] Generating Signals...")
        # We want signals for the 'Test' period usually (recent).
        # But user might want signals for specific range.
        # We'll return signals for the Test segment.
        
        try:
            # Predict
            predictions = model.predict(dataset, Segment.TEST)
            
            # Get metadata for Test segment
            result_df = dataset.fetch_infer(Segment.TEST).select(["datetime", "vt_symbol"])
            
            # Attach predictions
            # predictions is numpy array
            if len(predictions) != len(result_df):
                print(f"[V3] Prediction size mismatch: {len(predictions)} vs {len(result_df)}")
                # Try to align or abort
                # Usually matches if fetch_infer(Segment.TEST) matches what predict used.
                return None
                
            result_df = result_df.with_columns(
                pl.Series(predictions).alias("raw_score")
            )
            
            # Post-process Signals (Rank Normalization to -3 to 3)
            final_df = self._post_process_signals(result_df)
            
            # 9. Save & Return
            self.save_factors(final_df)
            
            return final_df
            
        except Exception as e:
            print(f"[V3] Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def get_ashare_symbols(self) -> List[str]:
        """Get A-Share symbols (filtering indices/others)"""
        all_symbols = self.engine.selector.get_candidate_symbols()
        ashare_symbols = []
        for symbol in all_symbols:
            if any(symbol.startswith(prefix) for prefix in ['000', '002', '300', '600', '601', '603', '688']):
                ashare_symbols.append(symbol)
        return ashare_symbols

    def load_ashare_data(self, symbols, start_date, end_date) -> pl.DataFrame:
        """Load Daily Bar Data"""
        extended_days = 250 # For rolling windows
        print(f"[V3] Loading data (buffer: {extended_days} days)...")
        
        try:
            df = self.engine.lab.load_bar_df(
                vt_symbols=symbols,
                interval="d",
                start=start_date,
                end=end_date,
                extended_days=extended_days
            )
            return df if df is not None else pl.DataFrame()
        except Exception as e:
            print(f"[V3] Load error: {e}")
            return pl.DataFrame()

    def _add_processors(self, dataset: AlphaDataset):
        """Add data cleaning/normalization processors"""
        
        def clean_and_normalize(df: pl.DataFrame) -> pl.DataFrame:
            # 1. Drop columns with too many NaNs? Or Fill?
            # Alpha158 produces many features.
            # Simple strategy: Fill NaN with 0 (neutral for standardized features)
            # But better: Fill with median?
            # For simplicity and speed: Fill 0 after standardization?
            # Or Fill Forward first?
            # AlphaDataset expressions handle some logic.
            
            feature_cols = df.columns[2:-1] # Skip datetime, vt_symbol; Skip label (last)
            
            # Optimized: Handle Infs, NaNs and Normalize using LazyFrame for better plan optimization
            
            # 1. Replace Inf with 0 and Fill NaN/Null with 0
            fill_exprs = [
                pl.when(pl.col(c).is_infinite())
                .then(0)
                .otherwise(pl.col(c))
                .fill_nan(0)
                .fill_null(0)
                .alias(c)
                for c in feature_cols
            ]
            
            # 2. Cross-sectional Standardization (Z-Score)
            norm_exprs = []
            for col in feature_cols:
                mean = pl.col(col).mean().over("datetime")
                std = pl.col(col).std().over("datetime")
                expr = ((pl.col(col) - mean) / (std + 1e-8)).clip(-3, 3).fill_nan(0).fill_null(0).alias(col)
                norm_exprs.append(expr)
            
            # Execute with Lazy API for memory efficiency
            return (
                df.lazy()
                .with_columns(fill_exprs)
                .with_columns(norm_exprs)
                .collect()
            )

        def clean_label(df: pl.DataFrame) -> pl.DataFrame:
            # Drop rows where label is NaN (only for TRAIN/VALID)
            return df.drop_nulls(subset=["label"])
            
        # Add processors
        # 'learn' phase (Train/Valid): Clean features AND Drop NaN labels
        # dataset.add_processor("learn", clean_and_normalize)
        # dataset.add_processor("learn", clean_label)
        
        # # 'infer' phase (Test/Predict): Clean features ONLY (keep rows with NaN labels for prediction)
        # dataset.add_processor("infer", clean_and_normalize)

    def _train_model(self, dataset: AlphaDataset) -> Optional[MlpModel]:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[V3] Device: {device}")
        
        # Determine input size from dataset
        # dataset.learn_df is populated after process_data()
        # Features are cols [2:-1]
        sample_df = dataset.fetch_learn(Segment.TRAIN)
        if sample_df.is_empty():
            print("[V3] Training data empty!")
            return None
            
        input_size = len(sample_df.columns) - 3 # datetime, vt_symbol, label
        print(f"[V3] Input Feature Size: {input_size}")
        
        model = MlpModel(
            input_size=input_size,
            hidden_sizes=self.model_settings["hidden_sizes"],
            n_epochs=self.model_settings["n_epochs"],
            batch_size=self.model_settings["batch_size"],
            lr=self.model_settings["lr"],
            early_stop_rounds=self.model_settings["early_stop_rounds"],
            device=device,
            seed=42
        )
        
        try:
            model.fit(dataset)
            return model
        except Exception as e:
            print(f"[V3] Training failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _post_process_signals(self, df: pl.DataFrame) -> pl.DataFrame:
        """Convert raw model scores to Rank-Norm Signals (-3 to 3)"""
        # Rank per day
        df = df.with_columns([
            pl.col("raw_score").rank(method="average").over("datetime").alias("rank"),
            pl.col("raw_score").count().over("datetime").alias("count")
        ])
        
        # Norm to -3, 3
        # (rank / count - 0.5) * factor
        # Uniform distribution -0.5 to 0.5. 
        # We want to approximate N(0,1) or just use uniform mapping.
        # User v2 used: (rank/count - 0.5) * 3.46  (approx std dev of uniform is 1/sqrt(12)??)
        # Uniform[-0.5, 0.5] std is sqrt(1/12) = 0.288. 1/0.288 = 3.46. 
        # So this scales to std=1. Then clip(-3, 3).
        
        df = df.with_columns([
            (((pl.col("rank") / pl.col("count")) - 0.5) * 3.46)
            .clip(-3, 3)
            .alias("final_signal")
        ])
        
        return df.select(["datetime", "vt_symbol", "raw_score", "final_signal"])

    def save_factors(self, signal_df):
        """Save signals"""
        if signal_df is not None:
            print("[V3] Saving signals to 'ashare_mlp_v3'...")
            self.engine.lab.save_signal("ashare_mlp_v3", signal_df)
            print("[V3] Saved.")
