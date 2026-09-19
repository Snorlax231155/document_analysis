import time
import logging
from contextlib import contextmanager
from typing import Generator
from app.core.config import settings

# Configure standard root logger
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger("document_intelligence")


@contextmanager
def log_execution_time(action_name: str) -> Generator[dict, None, None]:
    """Context manager to measure and log execution duration in milliseconds."""
    start_time = time.perf_counter()
    metrics = {"elapsed_ms": 0.0}
    try:
        yield metrics
    finally:
        elapsed = (time.perf_counter() - start_time) * 1000.0
        metrics["elapsed_ms"] = round(elapsed, 2)
        logger.info(f"[{action_name}] completed in {metrics['elapsed_ms']} ms")
