"""MLOps course project package."""

import sys
from pathlib import Path

from loguru import logger


def setup_logging(script_name: str = "project") -> None:
    """Set up loguru configuration for the package.
    
    Args:
        script_name: Name of the script for log file naming.
    """
    log_dir = Path(__file__).parent.parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)

    logger.remove()
    logger.add(
        sys.stderr,
        format="<level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
    )
    logger.add(
        log_dir / f"{script_name}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation="50 MB",
        retention="7 days",
    )


__all__ = ["setup_logging"]
