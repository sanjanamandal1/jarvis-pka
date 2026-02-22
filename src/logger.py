"""
Centralized logging for JARVIS PKA.
Logs to both console and a rotating file (.pka_data/jarvis.log).
"""
from __future__ import annotations
import logging
import os
from pathlib import Path
from logging.handlers import RotatingFileHandler

LOG_DIR = Path(".pka_data")
LOG_FILE = LOG_DIR / "jarvis.log"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


def get_logger(name: str = "jarvis") -> logging.Logger:
    global _configured
    logger = logging.getLogger(name)

    if not _configured:
        logger.setLevel(logging.DEBUG)
        LOG_DIR.mkdir(parents=True, exist_ok=True)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))

        # File handler (rotates at 2MB, keeps 3 backups)
        fh = RotatingFileHandler(LOG_FILE, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))

        logger.addHandler(ch)
        logger.addHandler(fh)
        _configured = True

    return logger


# Convenience shortcuts
def info(msg: str):  get_logger().info(msg)
def debug(msg: str): get_logger().debug(msg)
def warn(msg: str):  get_logger().warning(msg)
def error(msg: str): get_logger().error(msg)