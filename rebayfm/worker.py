"""Worker-process entry points for the processing pipeline.

Application.run() holds the GIL almost continuously, starving any Python
thread in the same process. Processing therefore runs in a separate
process with its own GIL; only picklable EventTask/FMResult objects
cross the boundary.
"""

from __future__ import annotations

import logging

from rebayfm.config import Config
from rebayfm.core.models import EventTask, FMResult

_processor = None


def init_worker(config: Config) -> None:
    """ProcessPoolExecutor initializer: build the pipeline in the child."""
    logging.basicConfig(
        level=getattr(logging, config.logging.level.upper(), logging.DEBUG),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    from rebayfm.core.processor import FMProcessor

    global _processor
    _processor = FMProcessor(config)


def run_batch(tasks: list[EventTask]) -> dict[str, FMResult]:
    return _processor.process_batch(tasks)
