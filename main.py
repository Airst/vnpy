import uvicorn
import sys
import os
import re
from datetime import datetime
from pathlib import Path
from core.logger_writer import LoggerWriter
from vnpy.alpha.logger import logger

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent
LOG_ROOT = PROJECT_ROOT / "log"

# Redirect stdout and stderr
if not hasattr(sys.stdout, 'file') or not isinstance(sys.stdout, LoggerWriter):

    try:
        log_filename = os.environ.get("VNPY_WEB_UI_LOG_FILE")
        
        print(f"Current log file from env: {log_filename}")
        
        if not log_filename:
            log_files = sorted(Path(LOG_ROOT).glob("web_ui_*.log"), key=lambda p: p.stat().st_mtime)
            
            while len(log_files) >= 3:
                oldest_log = log_files.pop(0)
                try:
                    oldest_log.unlink()
                    print(f"Deleted old log file: {oldest_log}")
                except Exception as e:
                    print(f"Failed to delete old log file {oldest_log}: {e}")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"{LOG_ROOT}/web_ui_{timestamp}.log"
            os.environ["VNPY_WEB_UI_LOG_FILE"] = log_filename

        file = open(log_filename, "a", encoding="utf-8")
        sys.stdout = LoggerWriter(sys.stdout, file)
        sys.stderr = LoggerWriter(sys.stderr, file)
        
        logger.remove()

        fmt: str = "{time:YYYY-MM-DD HH:mm:ss} {message}"
        logger.add(sys.stdout, colorize=True, format=fmt)
        print(f"Logging to {log_filename}")
    except Exception as e:
        print(f"Failed to setup logger redirection: {e}")

if __name__ == "__main__":
    print(f"Starting Uvicorn server for FastAPI app..., {PROJECT_ROOT}")
    # --------------------------
    uvicorn.run(
        "core.main_controller:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        reload_excludes=[
            str(PROJECT_ROOT / "core" / "alpha_db/"),
            str(PROJECT_ROOT / "core" / "web_ui" / "node_modules/"),
            str(PROJECT_ROOT / "core" / "web_ui" / "dist/"),
            str(PROJECT_ROOT / "log/"),
            str(PROJECT_ROOT / ".git"),
            str(PROJECT_ROOT / "build/"),
            "**/__pycache__",
            "**/*.pyc",
            "**/*.parquet",
        ],
    )
    
    print("Forcing process exit...")
    os._exit(0)