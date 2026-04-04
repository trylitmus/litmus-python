"""HTTP transport for batch event ingestion.

Handles gzip compression, retries with exponential backoff, and
Retry-After header parsing.
"""

from __future__ import annotations

import gzip as gzip_mod
import json
import logging
from datetime import UTC, datetime
from io import BytesIO

import httpx

from litmus.version import VERSION

log = logging.getLogger("litmus")

DEFAULT_HOST = "https://events.trylitmus.app"
USER_AGENT = f"litmus-python/{VERSION}"


class APIError(Exception):
    def __init__(
        self,
        status: int | str,
        message: str,
        retry_after: float | None = None,
    ):
        self.status = status
        self.message = message
        self.retry_after = retry_after

    def __str__(self) -> str:
        return f"[Litmus] {self.message} ({self.status})"


_client: httpx.Client | None = None


def _get_client() -> httpx.Client:
    global _client
    if _client is None:
        _client = httpx.Client(
            transport=httpx.HTTPTransport(retries=2),
            follow_redirects=True,
        )
    return _client


def batch_post(
    api_key: str,
    host: str | None = None,
    use_gzip: bool = False,
    timeout: int = 15,
    batch: list[dict] | None = None,
) -> None:
    """POST a batch of events to /v1/events.

    This is the only network call the SDK makes. Everything else
    feeds into a queue that eventually calls this.
    """
    url = (host or DEFAULT_HOST).rstrip("/") + "/v1/events"

    body = {
        "events": batch or [],
        "sent_at": datetime.now(tz=UTC).isoformat(),
    }
    data = json.dumps(body, default=_json_serializer)

    headers = {
        "Content-Type": "application/json",
        "User-Agent": USER_AGENT,
        "Authorization": f"Bearer {api_key}",
    }

    if use_gzip:
        headers["Content-Encoding"] = "gzip"
        buf = BytesIO()
        with gzip_mod.GzipFile(fileobj=buf, mode="w") as gz:
            gz.write(data.encode("utf-8"))
        content = buf.getvalue()
    else:
        content = data.encode("utf-8")

    res = _get_client().post(url, content=content, headers=headers, timeout=timeout)

    if res.status_code in (200, 202):
        log.debug("batch of %d events uploaded", len(batch or []))
        return

    # Parse Retry-After for the consumer's backoff logic
    retry_after = _parse_retry_after(res)

    try:
        payload = res.json()
        detail = payload.get("error", res.text)
    except (ValueError, KeyError):
        detail = res.text

    raise APIError(res.status_code, detail, retry_after=retry_after)


def _parse_retry_after(res: httpx.Response) -> float | None:
    header = res.headers.get("Retry-After")
    if not header:
        return None
    try:
        return float(header)
    except (ValueError, TypeError):
        pass
    try:
        from email.utils import parsedate_to_datetime

        target = parsedate_to_datetime(header)
        return max(0.0, (target - datetime.now(UTC)).total_seconds())
    except (ValueError, TypeError):
        return None


def _json_serializer(obj: object) -> str:
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
