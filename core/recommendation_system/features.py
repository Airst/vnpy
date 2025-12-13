
import polars as pl
from vnpy.alpha.dataset import AlphaDataset

class RecommendationDataset(AlphaDataset):
    """
    A simple dataset for stock recommendation (Trend Following).
    """
    def __init__(
        self,
        df: pl.DataFrame,
        train_period: tuple[str, str],
        valid_period: tuple[str, str],
        test_period: tuple[str, str]
    ) -> None:
        super().__init__(
            df=df,
            train_period=train_period,
            valid_period=valid_period,
            test_period=test_period
        )

        # 1. Momentum Factors
        # ROC (Rate of Change)
        for w in [5, 10, 20, 60]:
            self.add_feature(f"roc_{w}", f"close / ts_delay(close, {w}) - 1")
        
        # MA Trend: Close / MA
        for w in [5, 10, 20, 60]:
            self.add_feature(f"ma_trend_{w}", f"close / ts_mean(close, {w})")

        # 2. Volatility
        # ATR / Close
        self.add_feature("atr_ratio_14", "ta_atr(high, low, close, 14) / close")
        
        # Standard Deviation (Rolling)
        self.add_feature("std_20", "ts_std(close, 20) / close")
        
        # Bollinger Width (Approximate): 4 * Std / MA
        self.add_feature("boll_width_20", "(ts_std(close, 20) * 4) / ts_mean(close, 20)")

        # 3. Oscillators
        self.add_feature("rsi_14", "ta_rsi(close, 14)")
        self.add_feature("rsi_6", "ta_rsi(close, 6)")

        # 4. Volume Factors
        # Volume ROC
        self.add_feature("vol_roc_5", "volume / ts_delay(volume, 5) - 1")
        # Volume / MA Volume
        self.add_feature("vol_ma_ratio_20", "volume / ts_mean(volume, 20)")

        # 5. Label: Next 5 days return
        self.set_label("ts_delay(close, -5) / close - 1")
