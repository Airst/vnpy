import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from datetime import datetime
import sys
import os
import threading
import re
import subprocess
import asyncio
from pathlib import Path

# --- Logger Redirection ---
class LoggerWriter:
    def __init__(self, writer, file):
        self.writer = writer
        self.file = file

    def write(self, message):
        if not message:
            return
        
        if message == "^" or message == "\n" or message.strip() == "":
            self.file.write(message)
            return    
        # If message already starts with a date (e.g. 2025-12-20 or [2025-12-20), don't add timestamp
        if re.search(r'^\s*(\[)?\d{4}-\d{2}-\d{2}', message):
            self.file.write(message)
        else:
            timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")
            self.file.write(timestamp + message)
        self.file.flush()

    def flush(self):
        self.writer.flush()
        self.file.flush()

    def close(self):
        self.file.close()

    def isatty(self):
        if hasattr(self.writer, "isatty"):
            return self.writer.isatty()
        return False

    def fileno(self):
        if hasattr(self.writer, "fileno"):
            return self.writer.fileno()
        raise OSError("LoggerWriter has no fileno")

# Redirect stdout and stderr to web_ui.log
# Check if already redirected to avoid double wrapping on reload
if not hasattr(sys.stdout, 'file') or not isinstance(sys.stdout, LoggerWriter):
    try:
        file = open("web_ui.log", "w")
        sys.stdout = LoggerWriter(sys.stdout, file)
        sys.stderr = LoggerWriter(sys.stderr, file)
    except Exception as e:
        print(f"Failed to setup logger redirection: {e}")
# --------------------------

# Ensure project root is in path
# core/main.py -> core -> root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from core.core_service import CoreService

app = FastAPI()
core_service = CoreService()

# API Routes
class BacktestRequest(BaseModel):
    strategy_name: str
    start_date: str
    end_date: str
    max_positions: int = 10  # Limit max concurrent positions
    setting: dict = {}

class PredictionRequest(BaseModel):
    strategy_name: str
    setting: dict = {}

async def ingest_alpha_data():
    """
    Generator that runs the data download script and yields output line by line.
    """
    # Use the same python executable as the current process
    python_executable = sys.executable
    script_path = os.path.join(PROJECT_ROOT, "data_download", "download_data.py")
    
    yield f"Starting data ingestion process using {python_executable}...\n"
    yield f"Script: {script_path}\n"
    
    try:
        process = await asyncio.create_subprocess_exec(
            python_executable, script_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(PROJECT_ROOT)
        )

        while True:
            line = await process.stdout.readline()
            if not line:
                break
            yield line.decode('utf-8')

        await process.wait()
        yield f"Process finished with exit code {process.returncode}\n"
        
    except Exception as e:
        yield f"Error during execution: {str(e)}\n"

async def run_alpha_research_stream():
    """
    Generator that runs the alpha research script and yields output line by line.
    """
    python_executable = sys.executable
    script_path = os.path.join(PROJECT_ROOT, "core", "alpha", "run_research.py")
    
    yield f"Starting alpha calculation process...\n"
    yield f"Script: {script_path}\n"
    
    try:
        # Set PYTHONPATH to include project root so imports work
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT)
        
        process = await asyncio.create_subprocess_exec(
            python_executable, script_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
            env=env
        )

        while True:
            line = await process.stdout.readline()
            if not line:
                break
            yield line.decode('utf-8')

        await process.wait()
        yield f"Process finished with exit code {process.returncode}\n"
        
    except Exception as e:
        yield f"Error during execution: {str(e)}\n"

@app.post("/api/alpha/ingest")
async def api_ingest_alpha():
    return StreamingResponse(ingest_alpha_data(), media_type="text/plain")

@app.post("/api/alpha/calculate")
async def api_calculate_alpha():
    return StreamingResponse(run_alpha_research_stream(), media_type="text/plain")

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

@app.post("/api/predict")
def run_prediction(req: PredictionRequest):
    try:
        predictions = core_service.run_prediction(
            strategy_name=req.strategy_name,
            setting=req.setting
        )
        return {"results": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Static Files (React Frontend)
# Mount assets first to avoid conflict with root catch-all
if os.path.exists("core/web_ui/dist/assets"):
    app.mount("/assets", StaticFiles(directory="core/web_ui/dist/assets"), name="assets")

@app.get("/{full_path:path}")
async def serve_react_app(full_path: str):
    # Serve index.html for any path that isn't an API call or static asset
    # This supports client-side routing if we add it later
    if full_path.startswith("api/"):
        raise HTTPException(status_code=404, detail="Not Found")
    return FileResponse("core/web_ui/dist/index.html")

if __name__ == "__main__":
    uvicorn.run("core.main:app", host="0.0.0.0", port=8000, reload=True)
