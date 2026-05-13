from pathlib import Path
from typing import List, Dict, Optional
from contextlib import asynccontextmanager
from datetime import datetime, time
import asyncio
import schedule
import json
import threading
import polars as pl

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from core.core_service import CoreService
from core.trade_service import TradeService
from core.logger_writer import LoggerWriter
from vnpy.trader.logger import logger as ts_logger
from vnpy.alpha.logger import logger

# Ensure project root is correct: core/main_controller.py -> core -> root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

core_service = CoreService()
trade_service = None
#trade_service = TradeService()

# --- Background LLM Task State ---
_llm_task_status: Dict[str, Dict] = {}  # vt_symbol -> {status, message, date}
_llm_task_lock = threading.Lock()

# Scheduler
def run_daily_task():
    print(f"[{datetime.now()}] Triggering Daily Task...")
    try:
        # Ensure trade service is connected (or try to connect)
        if not trade_service._connected:
            print("[Scheduler] TradeService not connected. Attempting connect...")
            trade_service.connect()

        # Run synchronously for now as vnpy is not async safe usually
        # Ideally, run in executor if it blocks too long, but for now direct call
        trade_service.run_daily_trade()
    except Exception as e:
        print(f"[Scheduler] Error running daily task: {e}")
        import traceback
        traceback.print_exc()

async def scheduler():
    print("[Scheduler] Started. Waiting for 09:10...")
    schedule.every().day.at("09:21").do(run_daily_task)
    
    while True:
        schedule.run_pending()
        await asyncio.sleep(1)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start scheduler
    #task = asyncio.create_task(scheduler())
    yield
    # Cleanup
    #task.cancel()
    trade_service.close()
    print("Shut down TradeService...")

app = FastAPI(lifespan=lifespan)

if trade_service is not None:
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

@app.get("/api/signals")
def get_signals():
    """Get list of available signals from alpha_db/signal directory."""
    return {"signals": core_service.get_signals()}

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

@app.get("/api/kline")
def get_kline(vt_symbol: str, start_date: str, end_date: str):
    """Get OHLCV candlestick data for a symbol within a date range."""
    try:
        start = datetime.strptime(start_date, "%Y%m%d")
        end = datetime.strptime(end_date, "%Y%m%d")
        bars = core_service.lab.load_bar_data(vt_symbol, "d", start, end)
        data = [
            {
                "time": bar.datetime.strftime("%Y-%m-%d"),
                "open": bar.open_price,
                "high": bar.high_price,
                "low": bar.low_price,
                "close": bar.close_price,
                "volume": bar.volume,
            }
            for bar in bars
        ]
        return {"data": data}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/symbols/search")
def search_symbols(keyword: str):
    return {"symbols": core_service.search_symbols(keyword)}

# --- LLM Rating API ---
@app.get("/api/llm_ratings")
def get_llm_ratings(
    vt_symbol: str = "",
    page: int = 1,
    page_size: int = 20,
    rating_filter: str = "",
    signal_name: str = ""
):
    """Get LLM ratings.
    
    If vt_symbol is provided: return single stock's rating history.
    Otherwise: return paginated list of all ratings with summary.
    """
    # Single stock mode
    if vt_symbol:
        return _get_single_stock_rating(vt_symbol, signal_name)
    
    # List mode with pagination
    return _get_all_ratings_list(page, page_size, rating_filter, signal_name)


def _get_single_stock_rating(vt_symbol: str, signal_name: str = ""):
    """Get rating history for a single stock."""
    try:
        # Security: prevent path traversal
        if ".." in vt_symbol or "/" in vt_symbol or "\\" in vt_symbol:
            raise HTTPException(status_code=400, detail="Invalid symbol")
        
        ratings_dir = PROJECT_ROOT / "core" / "alpha_db" / "llm_tasks"
        filepath = ratings_dir / f"{vt_symbol}.json"
        
        if not filepath.exists():
            raise HTTPException(status_code=404, detail=f"No rating found for: {vt_symbol}")
        
        with open(filepath, "r", encoding="utf-8") as f:
            history = json.load(f)
        
        if not isinstance(history, list) or not history:
            raise HTTPException(status_code=404, detail=f"No rating found for: {vt_symbol}")
        
        # Attach latest signal score if signal_name is provided
        if signal_name:
            try:
                signal_df = core_service.lab.load_signal(signal_name)
                if signal_df is not None and not signal_df.is_empty():
                    # Get latest score for this stock
                    latest_dt = signal_df['datetime'].max()
                    latest_df = signal_df.filter(
                        (pl.col('datetime') == latest_dt) &
                        (pl.col('vt_symbol') == vt_symbol)
                    )
                    latest_score = None
                    if not latest_df.is_empty():
                        row = latest_df.row(0, named=True)
                        latest_score = row.get('final_signal') or row.get('total_score') or 0
                    
                    # Attach latest score to all history entries
                    for entry in history:
                        entry['score'] = latest_score
            except Exception:
                pass  # Signal loading is optional
        
        latest = history[-1]
        date_str = latest.get("date", "")
        
        return {
            "vt_symbol": vt_symbol,
            "date": date_str,
            "latest": latest,
            "history": history,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _get_all_ratings_list(page: int, page_size: int, rating_filter: str, signal_name: str):
    """Get all stock ratings with server-side pagination."""
    try:
        ratings_dir = PROJECT_ROOT / "core" / "alpha_db" / "llm_tasks"
        
        # Step 1: Collect ALL existing ratings (keyed by vt_symbol)
        rated_map = {}  # vt_symbol -> latest rating dict
        if ratings_dir.exists():
            for f in ratings_dir.glob("*.json"):
                if f.name.startswith("ratings_"):
                    continue
                try:
                    with open(f, "r", encoding="utf-8") as fp:
                        history = json.load(fp)
                    if isinstance(history, list) and history:
                        entry = history[-1]
                        rated_map[entry.get("vt_symbol", f.stem)] = entry
                except Exception:
                    pass
        
        # Step 2: Load signal scores and get top 30
        signal_date = None
        score_map = {}
        top_signal_symbols = []
        if signal_name:
            try:
                signal_df = core_service.lab.load_signal(signal_name)
                if signal_df is not None and not signal_df.is_empty():
                    latest_dt = signal_df['datetime'].max()
                    signal_date = latest_dt.strftime("%Y-%m-%d") if latest_dt else None
                    latest_df = signal_df.filter(pl.col('datetime') == latest_dt)
                    for row in latest_df.iter_rows(named=True):
                        score = row.get('final_signal') or row.get('total_score') or 0
                        score_map[row['vt_symbol']] = score
                    # Top 30 by score
                    sorted_symbols = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
                    top_signal_symbols = [s for s, _ in sorted_symbols[:30]]
            except Exception:
                pass
        
        # Step 3: Build merged stock list = top 30 signal + all historically rated (deduplicated)
        seen = set()
        all_ratings = []
        
        # Add top 30 signal stocks first
        for sym in top_signal_symbols:
            seen.add(sym)
            if sym in rated_map:
                entry = rated_map[sym].copy()
            else:
                # Placeholder for unrated signal stock
                entry = {
                    "vt_symbol": sym,
                    "date": None,
                    "rating": None,
                    "action": None,
                    "risk_level": None,
                    "reason": None,
                    "confidence": None,
                    "analysis_dimensions": {},
                    "key_factors": [],
                    "error": None,
                }
            entry['score'] = score_map.get(sym)
            all_ratings.append(entry)
        
        # Add remaining historically rated stocks
        for sym, entry in rated_map.items():
            if sym not in seen:
                seen.add(sym)
                entry_copy = entry.copy()
                entry_copy['score'] = score_map.get(sym)
                all_ratings.append(entry_copy)
        
        # Step 4: Compute summary stats from rated stocks only
        rated_entries = [r for r in all_ratings if r.get("rating") is not None]
        good = sum(1 for r in rated_entries if r.get("rating") == "good")
        bad = sum(1 for r in rated_entries if r.get("rating") == "bad")
        neutral = sum(1 for r in rated_entries if r.get("rating") == "neutral")
        error = sum(1 for r in rated_entries if r.get("error"))
        confidences = [r.get("confidence", 0) for r in rated_entries if r.get("confidence") is not None]
        avg_conf = f"{sum(confidences) / len(confidences):.2f}" if confidences else "N/A"
        
        total_unfiltered = len(all_ratings)
        
        # Step 5: Apply rating filter for display
        if rating_filter:
            all_ratings = [r for r in all_ratings if r.get("rating", "") == rating_filter]
        
        # Step 6: Sort by score descending (None scores go to bottom)
        all_ratings.sort(key=lambda x: x.get('score') if x.get('score') is not None else float('-inf'), reverse=True)
        
        total = len(all_ratings)
        start = (page - 1) * page_size
        end = start + page_size
        page_ratings = all_ratings[start:end]
        
        return {
            "ratings": page_ratings,
            "total": total,
            "total_unfiltered": total_unfiltered,
            "page": page,
            "page_size": page_size,
            "signal_date": signal_date,
            "summary": {
                "good": good,
                "bad": bad,
                "neutral": neutral,
                "error": error,
                "avg_confidence": avg_conf,
            },
        }
    except Exception as e:
        logger.error(f"Error listing all ratings: {e}")
        return {
            "ratings": [], "total": 0, "total_unfiltered": 0, "page": page, "page_size": page_size,
            "summary": {"good": 0, "bad": 0, "neutral": 0, "error": 0, "avg_confidence": "N/A"},
            "signal_date": None,
        }

# Deprecated: This endpoint is no longer used
# @app.get("/api/llm_ratings/{filename}")
# def get_llm_rating_file(filename: str):
#     """Get all ratings for a specific date (aggregated from per-stock files)."""
#     try:
#         import json
#         import polars as pl
#         from datetime import datetime
#         
#         # Security: prevent path traversal
#         if ".." in filename or "/" in filename or "\\" in filename:
#             raise HTTPException(status_code=400, detail="Invalid filename")
#         
#         ratings_dir = PROJECT_ROOT / "core" / "alpha_db" / "llm_tasks"
#         if not ratings_dir.exists():
#             raise HTTPException(status_code=404, detail=f"Rating directory not found")
#         
#         # Collect all stocks for this date
#         rating_list = []
#         for f in ratings_dir.glob("*.json"):
#             if f.name.startswith("ratings_"):  # Skip legacy files
#                 continue
#             try:
#                 with open(f, "r", encoding="utf-8") as fp:
#                     history = json.load(fp)
#                 if isinstance(history, list) and history:
#                     # Find entry matching the date (use latest if date matches, or find closest)
#                     for entry in reversed(history):
#                         if entry.get("date") == filename:
#                             rating_list.append(entry)
#                             break
#             except Exception:
#                 pass
#         
#         if not rating_list:
#             raise HTTPException(status_code=404, detail=f"No ratings found for date: {filename}")
#         
#         # Try to attach signal scores
#         try:
#             signal_name = "ashare_mlp_signal_v9"  # Default
#             signal_df = core_service.lab.load_signal(signal_name)
#             if signal_df is not None and not signal_df.is_empty():
#                 try:
#                     target_dt = datetime.strptime(filename, "%Y-%m-%d")
#                     day_signals = signal_df.filter(pl.col("datetime") == target_dt)
#                     score_map = {}
#                     for row in day_signals.iter_rows(named=True):
#                         score = row.get("final_signal") or row.get("total_score") or 0
#                         score_map[row["vt_symbol"]] = score
#                     
#                     for r in rating_list:
#                         r["score"] = score_map.get(r["vt_symbol"], None)
#                 except Exception:
#                     pass
#         except Exception:
#             pass  # Signal loading is optional
#         
#         return {
#             "date": filename,
#             "file": filename,
#             "ratings": rating_list,
#         }
#     except HTTPException:
#         raise
#     except Exception as e:
#         import traceback
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail=str(e))

# --- LLM Re-evaluation Background Task ---
_batch_task_status: Dict = {"running": False, "total": 0, "completed": 0, "failed": 0, "results": []}


def _run_llm_reevaluate(vt_symbol: str, score: float, check_date: str):
    """Background thread function to re-evaluate a single stock."""
    from core.llm import StockRatingScreener, save_ratings

    with _llm_task_lock:
        _llm_task_status[vt_symbol] = {
            "status": "running",
            "message": "LLM 评估中...",
            "date": check_date,
        }

    try:
        screener = StockRatingScreener()
        rating = screener.rate_one(vt_symbol, score, check_date)

        # Save the rating (appends to per-stock file)
        save_ratings([rating], str(PROJECT_ROOT / "core" / "alpha_db" / "llm_tasks"))

        with _llm_task_lock:
            _llm_task_status[vt_symbol] = {
                "status": "completed",
                "message": f"评估完成: {rating.rating}",
                "date": check_date,
                "rating": rating.rating,
                "confidence": rating.confidence,
                "reason": rating.reason,
            }
    except Exception as e:
        with _llm_task_lock:
            _llm_task_status[vt_symbol] = {
                "status": "failed",
                "message": f"评估失败: {str(e)}",
                "date": check_date,
            }
        logger.error(f"LLM re-evaluation failed for {vt_symbol}: {e}")


def _run_batch_reevaluate(failed_symbols: List[Dict[str, float]], check_date: str):
    """Background thread function to re-evaluate a batch of failed stocks."""
    from core.llm import StockRatingScreener, save_ratings
    from concurrent.futures import ThreadPoolExecutor, as_completed

    with _llm_task_lock:
        _batch_task_status["running"] = True
        _batch_task_status["total"] = len(failed_symbols)
        _batch_task_status["completed"] = 0
        _batch_task_status["failed"] = 0
        _batch_task_status["results"] = []

    print(f"[BatchReevaluate] Starting batch re-evaluation for {len(failed_symbols)} stocks...")

    completed = 0
    failed = 0
    results = []

    # Process in batches of 4
    batch_size = 4
    for batch_start in range(0, len(failed_symbols), batch_size):
        batch = failed_symbols[batch_start:batch_start + batch_size]
        print(f"[BatchReevaluate] Processing batch {batch_start // batch_size + 1} ({len(batch)} stocks)...")

        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            future_to_symbol = {}
            for item in batch:
                vt_symbol = item["vt_symbol"]
                score = item.get("score", 0.0)

                with _llm_task_lock:
                    _llm_task_status[vt_symbol] = {
                        "status": "running",
                        "message": "LLM 评估中...",
                        "date": check_date,
                    }

                def eval_one(sym=vt_symbol, sc=score):
                    screener = StockRatingScreener()
                    rating = screener.rate_one(sym, sc, check_date)
                    save_ratings([rating], str(PROJECT_ROOT / "core" / "alpha_db" / "llm_tasks"))
                    return rating

                future = executor.submit(eval_one)
                future_to_symbol[future] = vt_symbol

            for future in as_completed(future_to_symbol):
                vt_symbol = future_to_symbol[future]
                try:
                    rating = future.result()
                    with _llm_task_lock:
                        _llm_task_status[vt_symbol] = {
                            "status": "completed",
                            "message": f"评估完成: {rating.rating}",
                            "date": check_date,
                            "rating": rating.rating,
                            "confidence": rating.confidence,
                        }
                    completed += 1
                    results.append({"vt_symbol": vt_symbol, "status": "completed", "rating": rating.rating})
                    print(f"[BatchReevaluate]   {vt_symbol} → {rating.rating}")
                except Exception as e:
                    with _llm_task_lock:
                        _llm_task_status[vt_symbol] = {
                            "status": "failed",
                            "message": f"评估失败: {str(e)}",
                            "date": check_date,
                        }
                    failed += 1
                    results.append({"vt_symbol": vt_symbol, "status": "failed", "error": str(e)})
                    print(f"[BatchReevaluate]   {vt_symbol} failed: {e}")

        # Update progress
        with _llm_task_lock:
            _batch_task_status["completed"] = completed
            _batch_task_status["failed"] = failed
            _batch_task_status["results"] = results

    with _llm_task_lock:
        _batch_task_status["running"] = False

    print(f"[BatchReevaluate] Batch complete: {completed} succeeded, {failed} failed")


@app.post("/api/llm_ratings/reevaluate")
async def trigger_llm_reevaluate(vt_symbol: str, score: float = 0.0, check_date: Optional[str] = None):
    """Trigger background LLM re-evaluation for a single stock.
    
    Returns immediately without waiting for the task to complete.
    """
    if check_date is None:
        check_date = datetime.now().strftime("%Y-%m-%d")

    # Check if already running
    with _llm_task_lock:
        existing = _llm_task_status.get(vt_symbol)
        if existing and existing.get("status") == "running":
            raise HTTPException(status_code=409, detail=f"评估任务正在执行中: {vt_symbol}")

    # Launch background task using ThreadPoolExecutor
    thread = threading.Thread(
        target=_run_llm_reevaluate,
        args=(vt_symbol, score, check_date),
        daemon=True,
    )
    thread.start()

    return {
        "message": "评估任务已提交",
        "vt_symbol": vt_symbol,
        "check_date": check_date,
    }


@app.post("/api/llm_ratings/reevaluate_failed")
async def trigger_batch_reevaluate(signal_name: str = "", check_date: Optional[str] = None):
    """Trigger background batch re-evaluation for failed + unrated stocks.
    
    Finds:
    1. Stocks with error in latest rating entry (failed)
    2. Stocks from signal top list that have no rating file at all (unrated)
    
    If signal_name not provided, uses the latest selected signal from request query param.
    Falls back to no unrated scanning if signal loading fails.
    """
    if check_date is None:
        check_date = datetime.now().strftime("%Y-%m-%d")

    # Check if batch task already running
    with _llm_task_lock:
        if _batch_task_status.get("running"):
            raise HTTPException(status_code=409, detail="批量评估任务正在执行中")

    ratings_dir = PROJECT_ROOT / "core" / "alpha_db" / "llm_tasks"

    # Step 1: Find failed stocks
    rated_symbols = set()
    failed_symbols = []

    if ratings_dir.exists():
        for f in ratings_dir.glob("*.json"):
            if f.name.startswith("ratings_"):
                continue
            try:
                with open(f, "r", encoding="utf-8") as fp:
                    history = json.load(fp)
                if isinstance(history, list) and history:
                    latest = history[-1]
                    sym = latest.get("vt_symbol", f.stem)
                    rated_symbols.add(sym)
                    if latest.get("error"):
                        failed_symbols.append({
                            "vt_symbol": sym,
                            "score": latest.get("score", 0.0),
                        })
            except Exception:
                pass

    # Step 2: Find unrated stocks from signal top list
    unrated_symbols = []
    if signal_name:
        try:
            signal_df = core_service.lab.load_signal(signal_name)
            if signal_df is not None and not signal_df.is_empty():
                latest_dt = signal_df['datetime'].max()
                latest_df = signal_df.filter(pl.col('datetime') == latest_dt)
                for row in latest_df.iter_rows(named=True):
                    score = row.get('final_signal') or row.get('total_score') or 0
                    sym = row['vt_symbol']
                    if sym not in rated_symbols:
                        unrated_symbols.append({
                            "vt_symbol": sym,
                            "score": score,
                        })
        except Exception:
            pass

    # Step 3: Combine failed + unrated
    all_symbols = failed_symbols + unrated_symbols

    if not all_symbols:
        return {"message": "没有需要评估的股票", "count": 0, "failed": len(failed_symbols), "unrated": len(unrated_symbols)}

    # Launch background batch task
    thread = threading.Thread(
        target=_run_batch_reevaluate,
        args=(all_symbols, check_date),
        daemon=True,
    )
    thread.start()

    return {
        "message": f"批量评估任务已提交，共 {len(all_symbols)} 只股票（失败 {len(failed_symbols)} + 未评估 {len(unrated_symbols)}）",
        "count": len(all_symbols),
        "failed": len(failed_symbols),
        "unrated": len(unrated_symbols),
        "symbols": [s["vt_symbol"] for s in all_symbols],
        "check_date": check_date,
    }


@app.post("/api/llm_ratings/reevaluate_all")
async def trigger_reevaluate_all(check_date: Optional[str] = None):
    """Trigger background re-evaluation for ALL existing rated stocks.

    Scans all per-stock files and submits every stock for re-evaluation,
    regardless of current rating or error status.
    """
    if check_date is None:
        check_date = datetime.now().strftime("%Y-%m-%d")

    with _llm_task_lock:
        if _batch_task_status.get("running"):
            raise HTTPException(status_code=409, detail="批量评估任务正在执行中")

    ratings_dir = PROJECT_ROOT / "core" / "alpha_db" / "llm_tasks"
    all_symbols = []

    for f in ratings_dir.glob("*.json"):
        if f.name.startswith("ratings_"):
            continue
        try:
            with open(f, "r", encoding="utf-8") as fp:
                history = json.load(fp)
            if isinstance(history, list) and history:
                latest = history[-1]
                all_symbols.append({
                    "vt_symbol": latest.get("vt_symbol", f.stem),
                    "score": latest.get("score", 0.0) or 0.0,
                })
        except Exception:
            pass

    if not all_symbols:
        return {"message": "没有已评估的股票", "count": 0}

    thread = threading.Thread(
        target=_run_batch_reevaluate,
        args=(all_symbols, check_date),
        daemon=True,
    )
    thread.start()

    return {
        "message": f"全部重新评估任务已提交，共 {len(all_symbols)} 只股票",
        "count": len(all_symbols),
        "check_date": check_date,
    }


@app.get("/api/llm_ratings/batch_status")
def get_batch_task_status():
    """Get status of the batch re-evaluation task."""
    with _llm_task_lock:
        return dict(_batch_task_status)


@app.get("/api/llm_ratings/task_status/{vt_symbol}")
def get_llm_task_status(vt_symbol: str):
    """Get status of a background LLM evaluation task."""
    with _llm_task_lock:
        status = _llm_task_status.get(vt_symbol)
    
    if not status:
        return {"status": "unknown", "message": "无任务记录"}
    
    return status




@app.delete("/api/llm_ratings/stock/{vt_symbol}")
def delete_stock_rating(vt_symbol: str, date: Optional[str] = None):
    """Delete a specific rating entry from a stock's history.
    
    If date is provided, delete that specific entry.
    If date is not provided, delete the latest entry.
    If the history becomes empty, delete the file.
    """
    try:
        # Security: prevent path traversal
        if ".." in vt_symbol or "/" in vt_symbol or "\\" in vt_symbol:
            raise HTTPException(status_code=400, detail="Invalid symbol")
        
        ratings_dir = PROJECT_ROOT / "core" / "alpha_db" / "llm_tasks"
        filepath = ratings_dir / f"{vt_symbol}.json"
        
        if not filepath.exists():
            raise HTTPException(status_code=404, detail=f"Rating file not found: {vt_symbol}")
        
        with open(filepath, "r", encoding="utf-8") as f:
            history = json.load(f)
        
        if not isinstance(history, list) or not history:
            raise HTTPException(status_code=404, detail=f"No ratings found for: {vt_symbol}")
        
        if date:
            # Delete specific entry by date
            original_len = len(history)
            history = [h for h in history if h.get("date") != date]
            if len(history) == original_len:
                raise HTTPException(status_code=404, detail=f"No rating found for date: {date}")
        else:
            # Delete latest entry
            history.pop()
        
        if not history:
            # Delete the file if no history left
            filepath.unlink()
            return {"message": f"Deleted all ratings for {vt_symbol}", "vt_symbol": vt_symbol}
        
        # Save updated history
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        
        return {"message": f"Deleted rating for {vt_symbol}", "vt_symbol": vt_symbol, "remaining": len(history)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Trade API ---
@app.post("/api/trade/connect")
def connect_trade():
    return trade_service.connect()

@app.post("/api/trade/reset")
def reset_trade():
    return trade_service.reset_connection()

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