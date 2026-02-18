"""
Shared logging setup for ML pipeline scripts.

Usage:
    from logging_utils import setup_logging
    setup_logging("INFO")  # call once at start of main()
"""

import logging
import sys


def setup_logging(log_level="INFO"):
    """Configure root logger with a stdout handler.

    Args:
        log_level: One of DEBUG, INFO, WARNING, ERROR, CRITICAL.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    root = logging.getLogger()
    # Avoid duplicate handlers if called more than once
    if not root.handlers:
        root.setLevel(level)
        root.addHandler(handler)
    else:
        root.setLevel(level)
