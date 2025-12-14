from datetime import timedelta
import polars as pl
from vnpy.trader.ui import QtWidgets, QtCore
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class BacktestResultDialog(QtWidgets.QDialog):
    """
    Dialog to display backtest results including Chart, Daily Data, and Trade List.
    """
    def __init__(self, daily_df: pl.DataFrame, trades: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Backtest Results")
        self.resize(1200, 800)
        
        # Create Layout
        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)
        
        # Create Tabs
        tab_widget = QtWidgets.QTabWidget()
        layout.addWidget(tab_widget)
        
        # Tab 1: Chart
        self.figure = Figure(figsize=(10, 10))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        self.canvas.updateGeometry()
        tab_widget.addTab(self.canvas, "Chart")
        self.plot_chart(daily_df)
        
        # Tab 2: Daily Results
        self.daily_table = QtWidgets.QTableWidget()
        tab_widget.addTab(self.daily_table, "Daily Results")
        self.load_daily_df(daily_df)
        
        # Tab 3: Trades
        self.trade_table = QtWidgets.QTableWidget()
        tab_widget.addTab(self.trade_table, "Trades")
        self.load_trades(trades)

    def plot_chart(self, df: pl.DataFrame):
        # Fill NaNs to avoid plotting errors
        df = df.fill_nan(0).fill_null(0)
        
        dates = df["date"].to_numpy()
        balance = df["balance"].to_numpy()
        drawdown = df["drawdown"].to_numpy()
        net_pnl = df["net_pnl"].to_numpy()
        
        # Balance
        ax1 = self.figure.add_subplot(411)
        ax1.plot(dates, balance)
        ax1.set_title("Balance")
        ax1.grid(True)
        # Hide x labels for ax1 (shared)
        ax1.tick_params(labelbottom=False)
        
        # Drawdown
        ax2 = self.figure.add_subplot(412, sharex=ax1)
        ax2.fill_between(dates, drawdown, 0, color="red", alpha=0.5)
        ax2.plot(dates, drawdown, color="red")
        ax2.set_title("Drawdown")
        ax2.grid(True)
        ax2.tick_params(labelbottom=False)
        
        # Daily PnL
        ax3 = self.figure.add_subplot(413, sharex=ax1)
        ax3.bar(dates, net_pnl)
        ax3.set_title("Daily PnL")
        ax3.grid(True)
        # Rotate dates on the last shared axis
        ax3.tick_params(axis='x', rotation=30)
        
        # PnL Distribution
        ax4 = self.figure.add_subplot(414)
        ax4.hist(net_pnl, bins=50)
        ax4.set_title("Daily PnL Distribution")
        ax4.grid(True)
        
        self.figure.tight_layout()

    def load_daily_df(self, df: pl.DataFrame):
        if df.is_empty():
            return
            
        headers = df.columns
        self.daily_table.setColumnCount(len(headers))
        self.daily_table.setHorizontalHeaderLabels(headers)
        self.daily_table.setRowCount(len(df))
        
        for row_idx, row in enumerate(df.iter_rows()):
            for col_idx, value in enumerate(row):
                item = QtWidgets.QTableWidgetItem(str(value))
                self.daily_table.setItem(row_idx, col_idx, item)
        
        self.daily_table.resizeColumnsToContents()

    def load_trades(self, trades: list):
        if not trades:
            return
            
        headers = ["datetime", "symbol", "exchange", "direction", "offset", "price", "volume", "orderid", "tradeid", "gateway"]
        self.trade_table.setColumnCount(len(headers))
        self.trade_table.setHorizontalHeaderLabels(headers)
        self.trade_table.setRowCount(len(trades))
        
        for row_idx, trade in enumerate(trades):
            # Map trade object to list
            row_data = [
                trade.datetime,
                trade.symbol,
                trade.exchange.value if hasattr(trade.exchange, "value") else str(trade.exchange),
                trade.direction.value if hasattr(trade.direction, "value") else str(trade.direction),
                trade.offset.value if hasattr(trade.offset, "value") else str(trade.offset),
                trade.price,
                trade.volume,
                trade.orderid,
                trade.tradeid,
                trade.gateway_name
            ]
            
            for col_idx, value in enumerate(row_data):
                item = QtWidgets.QTableWidgetItem(str(value))
                self.trade_table.setItem(row_idx, col_idx, item)
                
        self.trade_table.resizeColumnsToContents()


class RecommendationDialog(QtWidgets.QDialog):
    def __init__(self, df: pl.DataFrame, title="Recommendations"):
        super().__init__()
        self.setWindowTitle(title)
        self.resize(1000, 600)
        
        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)
        
        # Info Label
        if not df.is_empty():
            data_date = df["datetime"][0]
            date_str = data_date.strftime("%Y-%m-%d")
            
            # Simple heuristic for next trading day (T+1)
            weekday = data_date.weekday()
            if weekday == 4: # Friday
                days_add = 3
            elif weekday == 5: # Saturday
                days_add = 2
            else:
                days_add = 1
                
            action_date = data_date + timedelta(days=days_add)
            action_date_str = action_date.strftime("%Y-%m-%d")
        else:
            date_str = "N/A"
            action_date_str = "N/A"
        
        # Find Top Pick
        top_pick = "None"
        df_sorted = df.sort("signal", descending=True)
        if not df_sorted.is_empty():
            top_signal = df_sorted["signal"][0]
            if top_signal > 0:
                top_pick = f"{df_sorted['vt_symbol'][0]} (Score: {top_signal:.4f})"
        
        label_text = (f"Data Date: {date_str} (Based on Close)\n"
                      f"Action Date: {action_date_str} (Target Trade Day)\n"
                      f"Top Recommendation: {top_pick}")
                      
        label = QtWidgets.QLabel(label_text)
        label.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px; color: blue;")
        layout.addWidget(label)
        
        # Table
        self.table = QtWidgets.QTableWidget()
        layout.addWidget(self.table)
        
        self.load_data(df)
        
    def load_data(self, df: pl.DataFrame):
        if df.is_empty():
            return
            
        headers = ["Symbol", "Score", "Close Price", "Rank"]
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setRowCount(len(df))
        
        # Sort by score desc
        df_sorted = df.sort("signal", descending=True)
        
        for row_idx, row in enumerate(df_sorted.iter_rows(named=True)):
            symbol = row["vt_symbol"]
            score = row["signal"]
            close = row["close"]
            rank = row_idx + 1
            
            # Highlight positive scores
            score_item = QtWidgets.QTableWidgetItem(f"{score:.4f}")
            if score > 0:
                score_item.setForeground(QtCore.Qt.darkGreen)
            else:
                score_item.setForeground(QtCore.Qt.red)
                
            self.table.setItem(row_idx, 0, QtWidgets.QTableWidgetItem(str(symbol)))
            self.table.setItem(row_idx, 1, score_item)
            self.table.setItem(row_idx, 2, QtWidgets.QTableWidgetItem(f"{close:.2f}"))
            self.table.setItem(row_idx, 3, QtWidgets.QTableWidgetItem(str(rank)))
            
        self.table.resizeColumnsToContents()
