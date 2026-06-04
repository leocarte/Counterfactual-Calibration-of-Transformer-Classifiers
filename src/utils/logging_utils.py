"""Structured logging setup."""
import logging
from rich.logging import RichHandler


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Create a rich-formatted logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = RichHandler(rich_tracebacks=True, markup=True)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level))
    return logger
