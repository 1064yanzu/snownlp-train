import logging
import os
import platform
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Any, Optional

_LOGGERS: dict[str, logging.Logger] = {}


def _safe_makedirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _today_stamp() -> str:
    return datetime.now().strftime("%Y%m%d")


def get_logger(
    name: str = "app",
    log_dir: str = "logs",
    level: int = logging.INFO,
) -> logging.Logger:
    key = f"{name}|{os.path.abspath(log_dir)}"
    if key in _LOGGERS:
        return _LOGGERS[key]

    _safe_makedirs(log_dir)

    logger = logging.getLogger(f"snownlp_train.{name}")
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        fmt = logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        logfile = os.path.join(log_dir, f"snownlp_train_{_today_stamp()}_{name}.log")
        file_handler = RotatingFileHandler(
            logfile,
            maxBytes=5 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setFormatter(fmt)
        logger.addHandler(stream_handler)

    _LOGGERS[key] = logger
    return logger


def get_log_file_path(logger: logging.Logger) -> Optional[str]:
    for h in logger.handlers:
        if isinstance(h, RotatingFileHandler):
            return getattr(h, "baseFilename", None)
    return None


def dir_writable(path: str) -> bool:
    try:
        _safe_makedirs(path)
        test_path = os.path.join(path, f".write_test_{os.getpid()}_{int(datetime.now().timestamp())}")
        with open(test_path, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(test_path)
        return True
    except Exception:
        return False


def disk_free_bytes(path: str) -> Optional[int]:
    try:
        import shutil

        usage = shutil.disk_usage(path)
        return int(usage.free)
    except Exception:
        return None


def runtime_summary() -> dict[str, Any]:
    return {
        "python": sys.version.replace("\n", " "),
        "executable": sys.executable,
        "platform": platform.platform(),
        "cwd": os.getcwd(),
    }
