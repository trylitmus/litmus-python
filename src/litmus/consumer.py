"""Background consumer thread that drains the event queue in batches.

A daemon thread that wakes on a flush interval, collects up to
flush_at items from the queue, and POSTs them as a single batch.
Retries with exponential backoff on transient failures.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from queue import Empty
from threading import Thread
from typing import TYPE_CHECKING

from litmus.request import APIError, batch_post

if TYPE_CHECKING:
    from queue import Queue

log = logging.getLogger("litmus")

# Hard ceiling per message to avoid blowing up the batch
MAX_MSG_SIZE = 900 * 1024  # 900 KiB
BATCH_SIZE_LIMIT = 5 * 1024 * 1024  # 5 MiB total batch


class Consumer(Thread):
    """Drains the client's queue and ships batches to the ingest API."""

    def __init__(
        self,
        queue: Queue[dict],
        api_key: str,
        host: str | None = None,
        on_error: Callable[[Exception, list[dict]], None] | None = None,
        flush_at: int = 100,
        flush_interval: float = 0.5,
        use_gzip: bool = False,
        retries: int = 10,
        timeout: int = 15,
    ):
        super().__init__()
        self.daemon = True
        self.queue = queue
        self.api_key = api_key
        self.host = host
        self.on_error = on_error
        self.flush_at = flush_at
        self.flush_interval = flush_interval
        self.use_gzip = use_gzip
        self.retries = retries
        self.timeout = timeout
        self.running = True

    def run(self) -> None:
        log.debug("consumer thread started")
        while self.running:
            self.upload()
        log.debug("consumer thread exited")

    def pause(self) -> None:
        self.running = False

    def upload(self) -> bool:
        """Pull the next batch off the queue and send it. Returns success."""
        batch = self._next_batch()
        if not batch:
            return False

        try:
            self._send_with_retries(batch)
            return True
        except Exception as exc:
            log.error("error uploading: %s", exc)
            if self.on_error:
                try:
                    self.on_error(exc, batch)
                except Exception as handler_err:
                    log.error("on_error callback failed: %s", handler_err)
            return False
        finally:
            for _ in batch:
                self.queue.task_done()

    # -- internal ------------------------------------------------------------

    def _next_batch(self) -> list[dict]:
        """Collect items from the queue until we hit flush_at count,
        flush_interval timeout, or the batch size limit."""
        items: list[dict] = []
        total_size = 0
        start = time.monotonic()

        while len(items) < self.flush_at:
            elapsed = time.monotonic() - start
            if elapsed >= self.flush_interval:
                break
            try:
                item = self.queue.get(
                    block=True,
                    timeout=self.flush_interval - elapsed,
                )
                item_size = len(json.dumps(item).encode())
                if item_size > MAX_MSG_SIZE:
                    log.error("event exceeds 900 KiB limit, dropping")
                    self.queue.task_done()
                    continue
                items.append(item)
                total_size += item_size
                if total_size >= BATCH_SIZE_LIMIT:
                    log.debug("hit batch size limit (%d bytes)", total_size)
                    break
            except Empty:
                break

        return items

    def _send_with_retries(self, batch: list[dict]) -> None:
        """Attempt to POST the batch, retrying transient errors."""
        last_exc: Exception | None = None

        for attempt in range(self.retries + 1):
            try:
                batch_post(
                    self.api_key,
                    host=self.host,
                    use_gzip=self.use_gzip,
                    timeout=self.timeout,
                    batch=batch,
                )
                return
            except Exception as exc:
                last_exc = exc
                if not self._is_retryable(exc):
                    raise
                if attempt < self.retries:
                    retry_after = getattr(exc, "retry_after", None)
                    if retry_after and retry_after > 0:
                        time.sleep(retry_after)
                    else:
                        time.sleep(min(2**attempt, 30))

        if last_exc:
            raise last_exc

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        if isinstance(exc, APIError):
            if exc.status == "N/A":
                return False
            status = int(exc.status)
            # 4xx is not retryable except 408 (timeout) and 429 (rate limit)
            if 400 <= status < 500 and status not in (408, 429):
                return False
        return True
