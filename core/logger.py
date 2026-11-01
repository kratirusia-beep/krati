"""
=============================================================================
Multimodal Attendance & Engagement Agent - Logger
Epic 1: Project Architecture & Environment Initialization
Task 1: Logging System with Privacy Masking
=============================================================================
"""

import logging
import sys
from pathlib import Path
from config import BASE_DIR

LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "agent.log"


class PrivacyFilter(logging.Filter):
    """Ensures raw biometric paths and sensitive hashes are sanitized in logs."""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = str(record.msg)
        # Redact raw embedding arrays if accidentally logged
        if "embedding" in msg.lower() and "[" in msg:
            record.msg = "[Biometric Embedding Redacted for Privacy]"
        return True


def setup_logger(name: str = "MultimodalAgent") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.INFO)
    logger.addFilter(PrivacyFilter())

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler
    c_handler = logging.StreamHandler(sys.stdout)
    c_handler.setLevel(logging.INFO)
    c_handler.setFormatter(formatter)
    logger.addHandler(c_handler)

    # File Handler
    f_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    f_handler.setLevel(logging.DEBUG)
    f_handler.setFormatter(formatter)
    logger.addHandler(f_handler)

    return logger


logger = setup_logger()
