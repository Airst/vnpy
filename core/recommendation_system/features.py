
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
        # ROC (Rate of Change) 20 days
        self.add_feature("roc_20", "ts_delay(close, 20) / close - 1")
        
        # MA Trend: Close / MA60
        self.add_feature("ma_trend", "close / ts_mean(close, 60)")

        # 2. Volatility
        # ATR / Close
        self.add_feature("atr_ratio", "ta_atr(high, low, close, 14) / close")

        # 3. RSI
        self.add_feature("rsi_14", "ta_rsi(close, 14)")

        # 4. Label: Next 5 days return
        self.set_label("ts_delay(close, -5) / close - 1")
