"""Tests for the Litmus Python SDK.

Uses httpx's MockTransport to intercept HTTP calls (no mocking internals).
Most tests use sync_mode for determinism. Threading integration tests
swap the transport on the shared client.
"""

from __future__ import annotations

import json
from collections.abc import Generator
from datetime import datetime

import httpx
import pytest

from litmus import LitmusClient

INGEST_URL = "https://ingest.trylitmus.com/v1/events"


class CallLog:
    """Collects requests sent through a MockTransport."""

    def __init__(self, status: int = 202, body: dict | None = None):
        self.calls: list[httpx.Request] = []
        self._responses: list[tuple[int, dict]] = []
        self._default_status = status
        self._default_body = body or {"accepted": 1}

    def add_response(self, status: int, body: dict) -> None:
        self._responses.append((status, body))

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        if self._responses:
            status, body = self._responses.pop(0)
        else:
            status = self._default_status
            body = self._default_body
        return httpx.Response(status, json=body)

    def events(self) -> list[dict]:
        result = []
        for call in self.calls:
            body = json.loads(call.content)
            result.extend(body["events"])
        return result


@pytest.fixture
def mock_transport() -> Generator[CallLog, None, None]:
    """Swap the module-level httpx.Client with a MockTransport, restore after."""
    import litmus.request as req

    old_client = req._client
    log = CallLog()
    req._client = httpx.Client(transport=httpx.MockTransport(log.handler))
    yield log
    if req._client is not None:
        req._client.close()
    req._client = old_client


def _make_mock_client(log: CallLog) -> httpx.Client:
    """Build an httpx.Client backed by a CallLog's MockTransport."""
    return httpx.Client(transport=httpx.MockTransport(log.handler))


class TestTrackSync:
    """sync_mode sends inline, no threads. Deterministic."""

    def test_single_event(self, mock_transport: CallLog) -> None:
        client = LitmusClient(api_key="ltm_pk_test_abc", sync_mode=True)
        event_id = client.track(event_type="$accept", session_id="s1")

        assert event_id is not None
        assert len(mock_transport.calls) == 1
        body = json.loads(mock_transport.calls[0].content)
        assert len(body["events"]) == 1
        assert body["events"][0]["type"] == "$accept"
        assert body["events"][0]["session_id"] == "s1"
        assert body["events"][0]["id"] == event_id

    def test_all_fields(self, mock_transport: CallLog) -> None:
        client = LitmusClient(api_key="ltm_pk_test_abc", sync_mode=True)
        client.track(
            event_type="$edit",
            session_id="s1",
            user_id="u1",
            prompt_id="content_gen",
            prompt_version="v2",
            generation_id="gen-123",
            metadata={"edit_distance": 0.3},
        )

        event = json.loads(mock_transport.calls[0].content)["events"][0]
        assert event["user_id"] == "u1"
        assert event["prompt_id"] == "content_gen"
        assert event["prompt_version"] == "v2"
        assert event["generation_id"] == "gen-123"
        assert event["metadata"]["edit_distance"] == 0.3
        assert event["metadata"]["$lib"] == "litmus-python"

    def test_bearer_token(self, mock_transport: CallLog) -> None:
        client = LitmusClient(api_key="ltm_pk_test_secret", sync_mode=True)
        client.track(event_type="$view", session_id="s1")

        auth = mock_transport.calls[0].headers["Authorization"]
        assert auth == "Bearer ltm_pk_test_secret"

    def test_optional_fields_omitted(self, mock_transport: CallLog) -> None:
        client = LitmusClient(api_key="ltm_pk_test_abc", sync_mode=True)
        client.track(event_type="$view", session_id="s1")

        event = json.loads(mock_transport.calls[0].content)["events"][0]
        assert "user_id" not in event
        assert "prompt_id" not in event
        assert "generation_id" not in event

    def test_timestamp_is_iso8601(self, mock_transport: CallLog) -> None:
        client = LitmusClient(api_key="ltm_pk_test_abc", sync_mode=True)
        client.track(event_type="$view", session_id="s1")

        event = json.loads(mock_transport.calls[0].content)["events"][0]
        datetime.fromisoformat(event["timestamp"])


class TestDisabledAndDryRun:
    def test_disabled_drops_silently(self) -> None:
        client = LitmusClient(api_key="ltm_pk_test_abc", disabled=True, send=False)
        result = client.track(event_type="$accept", session_id="s1")
        assert result is None

    def test_send_false_returns_id(self) -> None:
        client = LitmusClient(api_key="ltm_pk_test_abc", send=False)
        result = client.track(event_type="$accept", session_id="s1")
        assert result is not None


class TestGeneration:
    def test_generation_emits_creation_and_signals(self, mock_transport: CallLog) -> None:
        client = LitmusClient(api_key="ltm_pk_test_abc", sync_mode=True)
        gen = client.generation("s1", prompt_id="summarizer", user_id="u1")
        gen.accept()
        gen.edit(edit_distance=0.5)
        gen.copy()

        events = mock_transport.events()
        types = [e["type"] for e in events]
        assert "$generation" in types
        assert "$accept" in types
        assert "$edit" in types
        assert "$copy" in types

        gen_ids = {e["generation_id"] for e in events}
        assert len(gen_ids) == 1
        assert gen.id in gen_ids

    def test_generation_rate(self, mock_transport: CallLog) -> None:
        client = LitmusClient(api_key="ltm_pk_test_abc", sync_mode=True)
        gen = client.generation("s1")
        gen.rate(5, scale="5-star", metadata={"comment": "great"})

        events = mock_transport.events()
        rate_event = next(e for e in events if e["type"] == "$rate")
        assert rate_event["metadata"]["value"] == 5
        assert rate_event["metadata"]["scale"] == "5-star"
        assert rate_event["metadata"]["comment"] == "great"

    def test_generation_share(self, mock_transport: CallLog) -> None:
        client = LitmusClient(api_key="ltm_pk_test_abc", sync_mode=True)
        gen = client.generation("s1")
        gen.share(channel="slack", edited_before_share=True)

        events = mock_transport.events()
        share_event = next(e for e in events if e["type"] == "$share")
        assert share_event["metadata"]["channel"] == "slack"
        assert share_event["metadata"]["edited_before_share"] is True

    def test_generation_flag(self, mock_transport: CallLog) -> None:
        client = LitmusClient(api_key="ltm_pk_test_abc", sync_mode=True)
        gen = client.generation("s1")
        gen.flag(reason="hallucination")

        events = mock_transport.events()
        flag_event = next(e for e in events if e["type"] == "$flag")
        assert flag_event["metadata"]["reason"] == "hallucination"

    def test_generation_post_accept_edit(self, mock_transport: CallLog) -> None:
        client = LitmusClient(api_key="ltm_pk_test_abc", sync_mode=True)
        gen = client.generation("s1")
        gen.post_accept_edit(edit_distance=0.2, time_since_accept_ms=5000)

        events = mock_transport.events()
        pae = next(e for e in events if e["type"] == "$post_accept_edit")
        assert pae["metadata"]["edit_distance"] == 0.2
        assert pae["metadata"]["time_since_accept_ms"] == 5000

    def test_all_signal_methods(self, mock_transport: CallLog) -> None:
        """Every signal method on Generation should produce an event."""
        client = LitmusClient(api_key="ltm_pk_test_abc", sync_mode=True)
        gen = client.generation("s1")
        gen.accept()
        gen.edit()
        gen.regenerate()
        gen.copy()
        gen.abandon()
        gen.view()
        gen.refine()
        gen.followup()
        gen.rephrase()
        gen.undo()
        gen.share()
        gen.flag()
        gen.rate(1)
        gen.escalate()
        gen.post_accept_edit()

        events = mock_transport.events()
        types = {e["type"] for e in events}
        expected = {
            "$generation",
            "$accept",
            "$edit",
            "$regenerate",
            "$copy",
            "$abandon",
            "$view",
            "$refine",
            "$followup",
            "$rephrase",
            "$undo",
            "$share",
            "$flag",
            "$rate",
            "$escalate",
            "$post_accept_edit",
        }
        assert expected.issubset(types)


class TestFeature:
    def test_feature_injects_defaults(self, mock_transport: CallLog) -> None:
        client = LitmusClient(api_key="ltm_pk_test_abc", sync_mode=True)
        feat = client.feature("content_gen", model="gpt-4o", user_id="u1")
        gen = feat.generation("s1")
        gen.accept()

        events = mock_transport.events()
        gen_event = next(e for e in events if e["type"] == "$generation")
        assert gen_event["user_id"] == "u1"
        assert gen_event["metadata"]["feature"] == "content_gen"
        assert gen_event["metadata"]["model"] == "gpt-4o"

    def test_feature_track(self, mock_transport: CallLog) -> None:
        client = LitmusClient(api_key="ltm_pk_test_abc", sync_mode=True)
        feat = client.feature("content_gen", user_id="u1")
        feat.track(event_type="$view", session_id="s1")

        events = mock_transport.events()
        assert len(events) == 1
        assert events[0]["metadata"]["feature"] == "content_gen"
        assert events[0]["user_id"] == "u1"


class TestAttach:
    """attach() returns a Generation handle without emitting $generation."""

    def test_attach_does_not_emit_generation_event(self, mock_transport: CallLog) -> None:
        client = LitmusClient(api_key="ltm_pk_test_abc", sync_mode=True)
        gen = client.attach("existing-gen-id", "s1", user_id="u1")
        gen.accept()

        events = mock_transport.events()
        types = [e["type"] for e in events]
        assert "$generation" not in types
        assert "$accept" in types

    def test_attach_uses_provided_generation_id(self, mock_transport: CallLog) -> None:
        client = LitmusClient(api_key="ltm_pk_test_abc", sync_mode=True)
        gen = client.attach("backend-gen-uuid", "s1")
        assert gen.id == "backend-gen-uuid"

        gen.edit(edit_distance=0.4)
        gen.copy()

        events = mock_transport.events()
        for event in events:
            assert event["generation_id"] == "backend-gen-uuid"

    def test_attach_carries_defaults(self, mock_transport: CallLog) -> None:
        client = LitmusClient(api_key="ltm_pk_test_abc", sync_mode=True)
        gen = client.attach(
            "gen-123",
            "s1",
            user_id="u1",
            prompt_id="summarizer",
            metadata={"source": "backend"},
        )
        gen.accept()

        event = mock_transport.events()[0]
        assert event["user_id"] == "u1"
        assert event["prompt_id"] == "summarizer"
        assert event["metadata"]["source"] == "backend"

    def test_cross_sdk_flow(self, mock_transport: CallLog) -> None:
        """Simulate the full backend-creates, frontend-attaches flow."""
        client = LitmusClient(api_key="ltm_pk_test_abc", sync_mode=True)

        # Backend creates generation (emits $generation)
        backend_gen = client.generation("s1", prompt_id="content_gen", user_id="u1")
        generation_id = backend_gen.id

        # Frontend attaches to same generation_id (no $generation emitted)
        frontend_gen = client.attach(generation_id, "s1", user_id="u1")
        frontend_gen.accept()
        frontend_gen.edit(edit_distance=0.3)

        events = mock_transport.events()
        gen_events = [e for e in events if e["type"] == "$generation"]
        assert len(gen_events) == 1  # only one $generation

        # All events share the same generation_id
        gen_ids = {e["generation_id"] for e in events}
        assert gen_ids == {generation_id}


class TestRetries:
    def test_sync_raises_on_4xx(self, mock_transport: CallLog) -> None:
        """Client errors raise immediately in sync_mode."""
        import litmus.request as req

        log = CallLog(status=400, body={"error": "bad request"})
        req._client = httpx.Client(transport=httpx.MockTransport(log.handler))

        client = LitmusClient(api_key="ltm_pk_test_abc", sync_mode=True)
        from litmus import APIError

        with pytest.raises(APIError) as exc_info:
            client.track(event_type="$accept", session_id="s1")
        assert exc_info.value.status == 400


class TestThreadedIntegration:
    """Verify the actual threading path works end-to-end."""

    def test_threaded_delivery(self, mock_transport: CallLog) -> None:
        """shutdown() -> flush() -> queue.join() is the sync primitive."""
        client = LitmusClient(
            api_key="ltm_pk_test_abc",
            flush_at=1,
            flush_interval=0.1,
        )
        event_id = client.track(event_type="$accept", session_id="s1")
        client.shutdown()

        assert len(mock_transport.calls) >= 1
        body = json.loads(mock_transport.calls[0].content)
        assert body["events"][0]["id"] == event_id

    def test_threaded_batching(self, mock_transport: CallLog) -> None:
        client = LitmusClient(
            api_key="ltm_pk_test_abc",
            flush_at=5,
            flush_interval=10,
        )
        for i in range(5):
            client.track(event_type="$view", session_id=f"s{i}")
        # flush_at=5 triggers the batch, queue.join() waits for delivery
        client.shutdown()

        assert len(mock_transport.calls) >= 1
        body = json.loads(mock_transport.calls[0].content)
        assert len(body["events"]) == 5

    def test_threaded_retry_on_500(self) -> None:
        import litmus.request as req

        log = CallLog()
        log.add_response(500, {"error": "boom"})
        # Second call gets the default 202
        old_client = req._client
        req._client = httpx.Client(transport=httpx.MockTransport(log.handler))

        try:
            client = LitmusClient(
                api_key="ltm_pk_test_abc",
                flush_at=1,
                flush_interval=0.1,
                max_retries=2,
            )
            client.track(event_type="$accept", session_id="s1")
            # queue.join() blocks until the consumer marks the item done,
            # which includes the retry backoff + second attempt
            client.shutdown()

            assert len(log.calls) == 2
        finally:
            req._client.close()
            req._client = old_client


class TestQueueFull:
    def test_returns_none_when_full(self) -> None:
        # Tiny queue, consumers paused
        client = LitmusClient(api_key="ltm_pk_test_abc", max_queue_size=1, send=False)
        # Flip send on and stuff the queue manually
        client.send = True
        client._queue.put({"fake": True})
        result = client.track(event_type="$view", session_id="s1")
        assert result is None
