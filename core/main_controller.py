from pathlib import Path
from typing import List
from contextlib import asynccontextmanager
from datetime import datetime, time
import asyncio

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from core.core_service import CoreService
from core.trade_service import TradeService
from core.trade.daily_task import DailyTrader
from core.logger_writer import LoggerWriter
from vnpy.trader.logger import logger as ts_logger
from vnpy.alpha.logger import logger

# Ensure project root is correct: core/main_controller.py -> core -> root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

core_service = CoreService()
trade_service = TradeService()

# Scheduler
async def scheduler():
    print("[Scheduler] Started. Waiting for 09:10...")
    while True:
        now = datetime.now()
        target_time = time(9, 10)
        
        # Check if it's 09:10
        if now.hour == target_time.hour and now.minute == target_time.minute:
             print(f"[{now}] Triggering Daily Task...")
             try:
                 # Ensure trade service is connected (or try to connect)
                 if not trade_service._connected:
                     print("[Scheduler] TradeService not connected. Attempting connect...")
                     trade_service.connect()
                 
                 trader = DailyTrader(trade_service.main_engine, trade_service.get_strategy_engine())
                 # Run synchronously for now as vnpy is not async safe usually
                 # Ideally, run in executor if it blocks too long, but for now direct call
                 trader.run()
             except Exception as e:
                 print(f"[Scheduler] Error running daily task: {e}")
                 import traceback
                 traceback.print_exc()
             
             # Sleep for 61 seconds to avoid double trigger
             await asyncio.sleep(61)
        else:
            # Sleep 30s
            await asyncio.sleep(30)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start scheduler
    task = asyncio.create_task(scheduler())
    yield
    # Cleanup
    task.cancel()
    trade_service.close()
    print("Shut down TradeService...")

app = FastAPI(lifespan=lifespan)

trade_service.connect()

# API Models
class BacktestRequest(BaseModel):
    strategy_name: str
    start_date: str
    end_date: str
    max_positions: int = 10
    setting: dict = {}

class SignalDataRequest(BaseModel):
    signal_name: str
    start_date: str
    end_date: str
    vt_symbols: List[str] = []

# API Routes
@app.get("/strategies")
def get_strategies():
    return {"strategies": core_service.get_strategies()}

@app.get("/factors")
def get_factors():
    return {"factors": core_service.get_signals()}

@app.get("/api/data_range")
def get_data_range():
    start, end = core_service.get_data_range()
    if start and end:
        return {
            "start": start.strftime("%Y%m%d"),
            "end": end.strftime("%Y%m%d")
        }
    return {"start": "", "end": ""}

@app.post("/api/backtest")
def run_backtest(req: BacktestRequest):
    try:
        start = datetime.strptime(req.start_date, "%Y%m%d")
        end = datetime.strptime(req.end_date, "%Y%m%d")
        
        result = core_service.run_backtest(
            strategy_name=req.strategy_name,
            start=start,
            end=end,
            setting=req.setting
        )
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/backtest/history")
def get_backtest_history():
    return {"history": core_service.get_backtest_history()}

@app.get("/api/backtest/result/{filename}")
def get_backtest_result(filename: str):
    try:
        return core_service.get_backtest_result(filename)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Backtest result not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/signal_data")
def get_signal_data(req: SignalDataRequest):
    try:
        start = datetime.strptime(req.start_date, "%Y%m%d")
        end = datetime.strptime(req.end_date, "%Y%m%d")
        result = core_service.get_signals_data(
            signal_name=req.signal_name,
            start_date=start,
            end_date=end,
            vt_symbols=req.vt_symbols
        )
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/symbols/search")
def search_symbols(keyword: str):
    return {"symbols": core_service.search_symbols(keyword)}

# --- Trade API ---
@app.post("/api/trade/connect")
def connect_trade():
    return trade_service.connect()

@app.post("/api/trade/reset")
def reset_trade():
    trade_service.reset_connection()
    return trade_service.connect()

@app.get("/api/trade/accounts")
def get_accounts():
    return {"accounts": trade_service.get_accounts()}

@app.get("/api/trade/positions")
def get_positions():
    return {"positions": trade_service.get_positions()}

@app.get("/api/trade/orders")
def get_orders():
    return {"orders": trade_service.get_orders()}

@app.get("/api/trade/trades")
def get_trades():
    return {"trades": trade_service.get_trades()}

@app.post("/api/trade/orders/cancel_all")
def cancel_all_orders():
    return trade_service.cancel_all_orders()


# Static Files Logic (Moved from controller)
# PROJECT_ROOT is already defined above
static_assets_path = PROJECT_ROOT / "core/web_ui/dist/assets"
index_html_path = PROJECT_ROOT / "core/web_ui/dist/index.html"

if static_assets_path.exists():
    app.mount("/assets", StaticFiles(directory=str(static_assets_path)), name="assets")

@app.get("/{full_path:path}")
async def serve_react_app(full_path: str):
    if full_path.startswith("api/"):
        raise HTTPException(status_code=404, detail="Not Found")
    return FileResponse(str(index_html_path))