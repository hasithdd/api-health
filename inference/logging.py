import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

def setup_logger():
    logger = logging.getLogger("inference")
    logger.setLevel(logging.INFO)

    handler = RotatingFileHandler(
        LOG_DIR / "api.log",
        maxBytes=5_000_000,
        backupCount=5,
    )

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s | %(message)s"
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger
