"""Litmus Python SDK client.

Drop-in event tracking with background batching, modeled after
PostHog's producer/consumer architecture. Events go into a
thread-safe queue, a daemon thread drains them in batches, and
batch_post() ships them to /v1/events.

    from litmus import LitmusClient

    client = LitmusClient(api_key="ltm_pk_live_...")
    gen = client.generation("session-123", prompt_id="content_gen")
    gen.event("$accept")
    gen.event("$edit", edit_distance=0.3)
    client.shutdown()
"""

from __future__ import annotations

import atexit
import logging
import queue
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from litmus.consumer import Consumer
from litmus.environment import collect_startup_metadata
from litmus.request import DEFAULT_HOST, batch_post
from litmus.version import VERSION

log = logging.getLogger("litmus")


# Literal union gives autocomplete in editors; Union[..., str] still accepts custom events.
SystemEvent = Literal[
    "$generation",
    "$regenerate",
    "$copy",
    "$edit",
    "$abandon",
    "$accept",
    "$view",
    "$partial_copy",
    "$refine",
    "$followup",
    "$rephrase",
    "$undo",
    "$share",
    "$flag",
    "$rate",
    "$escalate",
    "$switch_model",
    "$retry_context",
    "$post_accept_edit",
    "$sessionend",
    "$blur",
    "$return",
    "$scroll_regression",
    "$navigate",
    "$interrupt",
    "$startup",
]
EventType = SystemEvent | str


class Generation:
    """Handle for a single AI generation. Lets you record behavioral
    signals without re-threading IDs on every call.

        gen = client.generation("session-123")
        gen.event("$accept")
        gen.event("$edit", edit_distance=0.3)
        gen.event("my_custom_signal", whatever=True)
    """

    __slots__ = ("id", "_session_id", "_defaults", "_client")

    def __init__(
        self,
        client: LitmusClient,
        session_id: str,
        generation_id: str,
        defaults: dict,
    ):
        self._client = client
        self._session_id = session_id
        self.id = generation_id
        self._defaults = defaults

    def event(
        self,
        event_type: EventType,
        *,
        model: str | None = None,
        provider: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        total_tokens: int | None = None,
        duration_ms: int | None = None,
        ttft_ms: int | None = None,
        cost: float | None = None,
        **metadata: object,
    ) -> None:
        """Record a behavioral signal against this generation.

        Wire-level fields (model, provider, tokens, duration_ms, ttft_ms,
        cost) go as top-level event properties — the OpenAPI contract
        defines them there, and downstream analytics (BQI, dashboards)
        query them as real Postgres columns, not JSONB keys.

        Everything else passed via **kwargs lands in metadata.

        gen.event("$accept")
        gen.event("$generation", model="gpt-4o", input_tokens=150)
        gen.event("$edit", edit_distance=0.3)  # lands in metadata
        gen.event("$share", channel="slack")   # lands in metadata
        """
        merged = {**self._defaults.get("metadata", {}), **metadata}
        self._client.track(
            event_type=event_type,
            session_id=self._session_id,
            user_id=self._defaults.get("user_id"),
            prompt_id=self._defaults.get("prompt_id"),
            prompt_version=self._defaults.get("prompt_version"),
            generation_id=self.id,
            metadata=merged if merged else None,
            model=model,
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            duration_ms=duration_ms,
            ttft_ms=ttft_ms,
            cost=cost,
        )

    def edit(self, before: str, after: str, **metadata: object) -> None:
        """Record that the user modified the output before using it.

        Send the raw before/after text. The backend computes edit distance,
        diff classification, and derived metrics.
        """
        self.event("$edit", before=before, after=after, **metadata)

    def accept(self, **metadata: object) -> None:
        """User used the output as-is."""
        self.event("$accept", **metadata)

    def copy(self, **metadata: object) -> None:
        """User copied the output."""
        self.event("$copy", **metadata)

    def regenerate(self, **metadata: object) -> None:
        """User requested a new output. Fire BEFORE creating the next generation."""
        self.event("$regenerate", **metadata)

    def share(self, channel: str | None = None, **metadata: object) -> None:
        """User shared the output."""
        if channel:
            metadata["channel"] = channel
        self.event("$share", **metadata)


class Feature:
    """Scoped handle for an AI feature. Carries defaults so you don't
    repeat prompt_id/model/user_id on every generation.

        summarizer = client.feature("summarizer", model="gpt-4o")
        gen = summarizer.generation("session-123")
        gen.accept()
    """

    __slots__ = ("_client", "_defaults", "name")

    def __init__(self, client: LitmusClient, name: str, defaults: dict):
        self._client = client
        self.name = name
        self._defaults = {**defaults, "prompt_id": defaults.get("prompt_id", name)}

    def generation(
        self,
        session_id: str,
        user_id: str | None = None,
        prompt_version: str | None = None,
        model: str | None = None,
        provider: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        total_tokens: int | None = None,
        duration_ms: int | None = None,
        ttft_ms: int | None = None,
        cost: float | None = None,
        metadata: dict | None = None,
        generation_id: str | None = None,
    ) -> Generation:
        """Create a generation using this feature's defaults.

        Per-call kwargs (model, provider, tokens, latency, cost) win over
        the feature-scoped default. Wire-level fields are forwarded at the
        top level — they land in real Postgres columns, not in metadata.
        """
        return self._client.generation(
            session_id=session_id,
            user_id=user_id or self._defaults.get("user_id"),
            prompt_id=self._defaults.get("prompt_id"),
            prompt_version=prompt_version or self._defaults.get("prompt_version"),
            model=model or self._defaults.get("model"),
            provider=provider or self._defaults.get("provider"),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            duration_ms=duration_ms,
            ttft_ms=ttft_ms,
            cost=cost,
            metadata={
                "feature": self.name,
                **self._defaults.get("metadata", {}),
                **(metadata or {}),
            },
            generation_id=generation_id,
        )

    def track(
        self,
        event_type: EventType,
        session_id: str,
        user_id: str | None = None,
        prompt_id: str | None = None,
        prompt_version: str | None = None,
        generation_id: str | None = None,
        model: str | None = None,
        provider: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        total_tokens: int | None = None,
        duration_ms: int | None = None,
        ttft_ms: int | None = None,
        cost: float | None = None,
        metadata: dict | None = None,
    ) -> str | None:
        """Track an event scoped to this feature.

        All wire-level fields forward at the top level so the ingest server
        can write them to typed columns. feature name always tags metadata.
        """
        return self._client.track(
            event_type=event_type,
            session_id=session_id,
            user_id=user_id or self._defaults.get("user_id"),
            prompt_id=prompt_id or self._defaults.get("prompt_id"),
            prompt_version=prompt_version or self._defaults.get("prompt_version"),
            generation_id=generation_id,
            model=model or self._defaults.get("model"),
            provider=provider or self._defaults.get("provider"),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            duration_ms=duration_ms,
            ttft_ms=ttft_ms,
            cost=cost,
            metadata={
                **self._defaults.get("metadata", {}),
                "feature": self.name,
                **(metadata or {}),
            },
        )


class LitmusClient:
    """Litmus event client with background batching.

    Args:
        api_key: Your Litmus API key (ltm_pk_live_... or ltm_pk_test_...).
        host: Ingest endpoint URL. Defaults to https://ingest.trylitmus.app.
        max_queue_size: Max events buffered in memory before dropping. Default: 10000.
        on_error: Callback(exception, batch) invoked on send failure.
        flush_at: Batch size threshold that triggers an upload. Default: 100.
        flush_interval: Seconds to wait before flushing a partial batch. Default: 0.5.
        gzip: Compress payloads with gzip. Default: False.
        max_retries: Max retry attempts per batch. Default: 3.
        sync_mode: Send events inline (no background thread). Useful for
                   serverless or testing. Default: False.
        timeout: HTTP timeout in seconds. Default: 15.
        threads: Number of consumer threads. Default: 1.
        send: Actually send events (set False for dry-run). Default: True.
        debug: Enable DEBUG-level logging. Default: False.
        disabled: Silently drop all events. Default: False.
    """

    log = logging.getLogger("litmus")

    def __init__(
        self,
        api_key: str,
        host: str | None = None,
        max_queue_size: int = 10_000,
        on_error: Callable[[Exception, list[dict]], None] | None = None,
        flush_at: int = 10,
        flush_interval: float = 0.5,
        gzip: bool = False,
        max_retries: int = 3,
        sync_mode: bool = False,
        timeout: int = 15,
        threads: int = 1,
        send: bool = True,
        debug: bool = False,
        disabled: bool = False,
    ):
        self._queue: queue.Queue[dict] = queue.Queue(max_queue_size)
        self.api_key = api_key
        self.host = (host or DEFAULT_HOST).rstrip("/")
        self.on_error = on_error
        self.send = send
        self.sync_mode = sync_mode
        self.gzip = gzip
        self.timeout = timeout
        self.disabled = disabled
        self.consumers: list[Consumer] | None = None

        if debug:
            logging.basicConfig()
            self.log.setLevel(logging.DEBUG)

        if sync_mode:
            self.consumers = None
        else:
            if send:
                atexit.register(self.join)

            self.consumers = []
            for _ in range(threads):
                consumer = Consumer(
                    queue=self._queue,
                    api_key=self.api_key,
                    host=self.host,
                    on_error=on_error,
                    flush_at=flush_at,
                    flush_interval=flush_interval,
                    use_gzip=gzip,
                    retries=max_retries,
                    timeout=timeout,
                )
                self.consumers.append(consumer)
                if send:
                    consumer.start()

        # Fire $startup so the ingest server knows the SDK initialized.
        # Fastest possible signal for the setup wizard (no user interaction
        # required) and carries environment metadata for debugging.
        # Skip in sync_mode to avoid blocking the constructor with an HTTP
        # call (sync_mode is for serverless where cold start latency matters).
        if not sync_mode and not disabled:
            self.track(
                event_type="$startup",
                session_id="",
                metadata=collect_startup_metadata(),
            )

    # -- public API ----------------------------------------------------------

    def track(
        self,
        event_type: EventType,
        session_id: str,
        user_id: str | None = None,
        prompt_id: str | None = None,
        prompt_version: str | None = None,
        generation_id: str | None = None,
        metadata: dict | None = None,
        timestamp: datetime | None = None,
        model: str | None = None,
        provider: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        total_tokens: int | None = None,
        duration_ms: int | None = None,
        ttft_ms: int | None = None,
        cost: float | None = None,
    ) -> str | None:
        """Enqueue a single event. Returns the event UUID or None if dropped."""
        if self.disabled:
            return None

        ts = timestamp or datetime.now(tz=UTC)
        event_id = str(uuid4())

        msg: dict = {
            "id": event_id,
            "type": event_type,
            "session_id": session_id,
            "timestamp": ts.isoformat(),
        }
        if user_id:
            msg["user_id"] = user_id
        if prompt_id:
            msg["prompt_id"] = prompt_id
        if prompt_version:
            msg["prompt_version"] = prompt_version
        if generation_id:
            msg["generation_id"] = generation_id

        if model:
            msg["model"] = model
        if provider:
            msg["provider"] = provider
        if input_tokens is not None:
            msg["input_tokens"] = input_tokens
        if output_tokens is not None:
            msg["output_tokens"] = output_tokens
        if total_tokens is not None:
            msg["total_tokens"] = total_tokens
        if duration_ms is not None:
            msg["duration_ms"] = duration_ms
        if ttft_ms is not None:
            msg["ttft_ms"] = ttft_ms
        if cost is not None:
            msg["cost"] = cost

        props = {"$lib": "litmus-python", "$lib_version": VERSION}
        if metadata:
            props.update(metadata)
        msg["metadata"] = props

        self.log.debug("queueing: %s", msg)

        if not self.send:
            return event_id

        if self.sync_mode:
            batch_post(
                self.api_key,
                host=self.host,
                use_gzip=self.gzip,
                timeout=self.timeout,
                batch=[msg],
            )
            return event_id

        try:
            self._queue.put(msg, block=False)
            return event_id
        except queue.Full:
            self.log.warning("litmus queue is full, event dropped")
            return None

    def generation(
        self,
        session_id: str,
        user_id: str | None = None,
        prompt_id: str | None = None,
        prompt_version: str | None = None,
        model: str | None = None,
        provider: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        total_tokens: int | None = None,
        duration_ms: int | None = None,
        ttft_ms: int | None = None,
        cost: float | None = None,
        metadata: dict | None = None,
        generation_id: str | None = None,
    ) -> Generation:
        """Create a generation and return a handle for recording signals.

        Pass ``generation_id`` when the caller has already minted an id
        (e.g. to correlate with an OTel/OpenRouter trace broadcast). When
        omitted, a fresh UUID4 is used. The id is emitted on the
        ``$generation`` event, so downstream joins work either way.
        """
        if generation_id is None:
            generation_id = str(uuid4())
        defaults = {
            "user_id": user_id,
            "prompt_id": prompt_id,
            "prompt_version": prompt_version,
            "model": model,
            "metadata": metadata or {},
        }

        self.track(
            event_type="$generation",
            session_id=session_id,
            user_id=user_id,
            generation_id=generation_id,
            prompt_id=prompt_id,
            prompt_version=prompt_version,
            metadata=metadata,
            model=model,
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            duration_ms=duration_ms,
            ttft_ms=ttft_ms,
            cost=cost,
        )

        return Generation(self, session_id, generation_id, defaults)

    def attach(
        self,
        generation_id: str,
        session_id: str,
        user_id: str | None = None,
        prompt_id: str | None = None,
        prompt_version: str | None = None,
        metadata: dict | None = None,
    ) -> Generation:
        """Attach to an existing generation without re-emitting $generation.

        Use this when the generation was already created (by this service,
        another service, or a frontend SDK) and you have the generation_id.
        prompt_id/prompt_version are optional here because the $generation
        event already carries that context, everything joins on generation_id.

            gen = client.attach(request.generation_id, session_id)
            gen.accept()  # server-side behavioral signal
        """
        defaults = {
            "user_id": user_id,
            "prompt_id": prompt_id,
            "prompt_version": prompt_version,
            "metadata": metadata or {},
        }
        return Generation(self, session_id, generation_id, defaults)

    def feature(
        self,
        name: str,
        model: str | None = None,
        provider: str | None = None,
        user_id: str | None = None,
        prompt_version: str | None = None,
        metadata: dict | None = None,
    ) -> Feature:
        """Create a scoped feature handle that carries defaults."""
        defaults = {
            "prompt_id": name,
            "model": model,
            "provider": provider,
            "user_id": user_id,
            "prompt_version": prompt_version,
            "metadata": metadata or {},
        }
        return Feature(self, name, defaults)

    def flush(self) -> None:
        """Block until the queue is fully drained."""
        size = self._queue.qsize()
        self._queue.join()
        self.log.debug("flushed ~%d items", size)

    def join(self) -> None:
        """Stop consumer threads (call after flush)."""
        if self.consumers:
            for consumer in self.consumers:
                consumer.pause()
                try:
                    consumer.join()
                except RuntimeError:
                    pass

    def shutdown(self) -> None:
        """Flush all pending events and stop consumer threads.
        Call this before process exit in serverless environments."""
        self.flush()
        self.join()
