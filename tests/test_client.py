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
from litmus.version import VERSION

INGEST_URL = "https://ingest.trylitmus.app/v1/events"


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
        assert event["metadata"]["$lib_version"] == VERSION

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
        gen.event("$accept")
        gen.event("$edit", edit_distance=0.5)
        gen.event("$copy")

        events = mock_transport.events()
        types = [e["type"] for e in events]
        assert "$generation" in types
        assert "$accept" in types
        assert "$edit" in types
        assert "$copy" in types

        gen_ids = {e["generation_id"] for e in events}
        assert len(gen_ids) == 1
        assert gen.id in gen_ids

    def test_event_with_metadata(self, mock_transport: CallLog) -> None:
        client = LitmusClient(api_key="ltm_pk_test_abc", sync_mode=True)
        gen = client.generation("s1")
        gen.event("$rate", value=5, scale="5-star", comment="great")

        events = mock_transport.events()
        rate_event = next(e for e in events if e["type"] == "$rate")
        assert rate_event["metadata"]["value"] == 5
        assert rate_event["metadata"]["scale"] == "5-star"
        assert rate_event["metadata"]["comment"] == "great"

    def test_event_share(self, mock_transport: CallLog) -> None:
        client = LitmusClient(api_key="ltm_pk_test_abc", sync_mode=True)
        gen = client.generation("s1")
        gen.event("$share", channel="slack", edited_before_share=True)

        events = mock_transport.events()
        share_event = next(e for e in events if e["type"] == "$share")
        assert share_event["metadata"]["channel"] == "slack"
        assert share_event["metadata"]["edited_before_share"] is True

    def test_event_flag(self, mock_transport: CallLog) -> None:
        client = LitmusClient(api_key="ltm_pk_test_abc", sync_mode=True)
        gen = client.generation("s1")
        gen.event("$flag", reason="hallucination")

        events = mock_transport.events()
        flag_event = next(e for e in events if e["type"] == "$flag")
        assert flag_event["metadata"]["reason"] == "hallucination"

    def test_event_post_accept_edit(self, mock_transport: CallLog) -> None:
        client = LitmusClient(api_key="ltm_pk_test_abc", sync_mode=True)
        gen = client.generation("s1")
        gen.event("$post_accept_edit", edit_distance=0.2, time_since_accept_ms=5000)

        events = mock_transport.events()
        pae = next(e for e in events if e["type"] == "$post_accept_edit")
        assert pae["metadata"]["edit_distance"] == 0.2
        assert pae["metadata"]["time_since_accept_ms"] == 5000

    def test_custom_event(self, mock_transport: CallLog) -> None:
        """Custom events work just like system events."""
        client = LitmusClient(api_key="ltm_pk_test_abc", sync_mode=True)
        gen = client.generation("s1")
        gen.event("my_custom_signal", score=0.9, source="manual")

        events = mock_transport.events()
        custom = next(e for e in events if e["type"] == "my_custom_signal")
        assert custom["metadata"]["score"] == 0.9
        assert custom["metadata"]["source"] == "manual"

    def test_all_system_events(self, mock_transport: CallLog) -> None:
        """Every system event type should work through gen.event()."""
        client = LitmusClient(api_key="ltm_pk_test_abc", sync_mode=True)
        gen = client.generation("s1")
        system_events = [
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
        ]
        for evt in system_events:
            gen.event(evt)

        events = mock_transport.events()
        types = {e["type"] for e in events}
        # +1 for $generation from client.generation()
        expected = {"$generation", *system_events}
        assert expected.issubset(types)


class TestFeature:
    def test_feature_injects_defaults(self, mock_transport: CallLog) -> None:
        client = LitmusClient(api_key="ltm_pk_test_abc", sync_mode=True)
        feat = client.feature("content_gen", model="gpt-4o", user_id="u1")
        gen = feat.generation("s1")
        gen.event("$accept")

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


class TestGenerationIdOverride:
    """generation() accepts an externally-minted generation_id so callers
    can correlate the $generation event with ids they already emit elsewhere
    (e.g. OTel span metadata broadcast by OpenRouter)."""

    def test_uses_provided_generation_id(self, mock_transport: CallLog) -> None:
        client = LitmusClient(api_key="ltm_pk_test_abc", sync_mode=True)
        gen = client.generation("s1", prompt_id="chat", generation_id="external-uuid")
        assert gen.id == "external-uuid"

        gen.event("$accept")
        events = mock_transport.events()
        assert all(e["generation_id"] == "external-uuid" for e in events)

    def test_default_generates_uuid_when_absent(self, mock_transport: CallLog) -> None:
        client = LitmusClient(api_key="ltm_pk_test_abc", sync_mode=True)
        gen = client.generation("s1")
        # UUID4 string: 36 chars, 4 hyphens
        assert len(gen.id) == 36
        assert gen.id.count("-") == 4


class TestGenerationTopLevelFields:
    """generation() forwards model, provider, tokens, cost, latency as
    TOP-LEVEL wire fields on the $generation event. The ingest server
    stores these in dedicated columns (model TEXT, provider TEXT, etc.)
    so dashboards can query them without JSONB lookups.

    Regression guard for the v0.4.0 bug where these were accepted as
    kwargs but silently dropped before send."""

    def test_model_is_top_level(self, mock_transport: CallLog) -> None:
        client = LitmusClient(api_key="ltm_pk_test_abc", sync_mode=True)
        client.generation("s1", prompt_id="chat", model="claude-sonnet-4-20250514")

        event = next(e for e in mock_transport.events() if e["type"] == "$generation")
        assert event["model"] == "claude-sonnet-4-20250514"
        # Must NOT be in metadata — that was the old path we're moving off.
        assert event["metadata"].get("model") is None

    def test_provider_and_usage_are_top_level(self, mock_transport: CallLog) -> None:
        client = LitmusClient(api_key="ltm_pk_test_abc", sync_mode=True)
        client.generation(
            "s1",
            prompt_id="chat",
            model="gpt-4o",
            provider="openai",
            input_tokens=120,
            output_tokens=340,
            total_tokens=460,
            duration_ms=1850,
            ttft_ms=240,
            cost=0.0042,
        )

        event = next(e for e in mock_transport.events() if e["type"] == "$generation")
        assert event["model"] == "gpt-4o"
        assert event["provider"] == "openai"
        assert event["input_tokens"] == 120
        assert event["output_tokens"] == 340
        assert event["total_tokens"] == 460
        assert event["duration_ms"] == 1850
        assert event["ttft_ms"] == 240
        assert event["cost"] == 0.0042

    def test_optional_fields_omitted_when_none(self, mock_transport: CallLog) -> None:
        """None-valued fields are dropped from the wire payload entirely —
        the ingest server treats missing and null the same, but we keep
        payloads lean."""
        client = LitmusClient(api_key="ltm_pk_test_abc", sync_mode=True)
        client.generation("s1", prompt_id="chat")

        event = next(e for e in mock_transport.events() if e["type"] == "$generation")
        for key in (
            "model",
            "provider",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "duration_ms",
            "ttft_ms",
            "cost",
        ):
            assert key not in event, f"{key} should be omitted when None"

    def test_feature_forwards_model_top_level(self, mock_transport: CallLog) -> None:
        """Feature-scoped defaults also end up as top-level wire fields so
        dashboards using feature() get the same model column as callers
        using generation() directly."""
        client = LitmusClient(api_key="ltm_pk_test_abc", sync_mode=True)
        feat = client.feature("summarizer", model="claude-haiku-4")
        feat.generation("s1")

        event = next(e for e in mock_transport.events() if e["type"] == "$generation")
        assert event["model"] == "claude-haiku-4"


class TestAttach:
    """attach() returns a Generation handle without emitting $generation."""

    def test_attach_does_not_emit_generation_event(self, mock_transport: CallLog) -> None:
        client = LitmusClient(api_key="ltm_pk_test_abc", sync_mode=True)
        gen = client.attach("existing-gen-id", "s1")
        gen.event("$accept")

        events = mock_transport.events()
        types = [e["type"] for e in events]
        assert "$generation" not in types
        assert "$accept" in types

    def test_attach_uses_provided_generation_id(self, mock_transport: CallLog) -> None:
        client = LitmusClient(api_key="ltm_pk_test_abc", sync_mode=True)
        gen = client.attach("backend-gen-uuid", "s1")
        assert gen.id == "backend-gen-uuid"

        gen.event("$edit", edit_distance=0.4)
        gen.event("$copy")

        events = mock_transport.events()
        for event in events:
            assert event["generation_id"] == "backend-gen-uuid"

    def test_attach_minimal_no_opts(self, mock_transport: CallLog) -> None:
        """attach() needs only generation_id and session_id. No user_id needed."""
        client = LitmusClient(api_key="ltm_pk_test_abc", sync_mode=True)
        gen = client.attach("gen-abc", "s1")
        gen.event("$copy")

        event = mock_transport.events()[0]
        assert event["generation_id"] == "gen-abc"
        assert event["session_id"] == "s1"
        assert "user_id" not in event


class TestCorrelation:
    """Verify that generation() and attach() produce events that correlate
    on generation_id. This is the core contract for cross-SDK usage:
    backend emits $generation with prompt context, frontend emits behavioral
    signals, everything joins on generation_id."""

    def test_all_events_share_generation_id(self, mock_transport: CallLog) -> None:
        """The fundamental correlation: every event from both generation()
        and attach() references the same generation_id."""
        client = LitmusClient(api_key="ltm_pk_test_abc", sync_mode=True)

        # Backend creates generation (emits $generation)
        backend_gen = client.generation(
            "s1",
            prompt_id="content_gen",
            prompt_version="v2.3",
            metadata={"model": "gpt-4o", "latency_ms": 420},
        )
        generation_id = backend_gen.id

        # Frontend attaches, records behavioral signals (no $generation)
        frontend_gen = client.attach(generation_id, "s1")
        frontend_gen.event("$accept")
        frontend_gen.event("$edit", edit_distance=0.3)
        frontend_gen.event("$copy")

        events = mock_transport.events()

        # Every single event must share the same generation_id
        gen_ids = {e["generation_id"] for e in events}
        assert gen_ids == {generation_id}, (
            f"Expected all events to share generation_id {generation_id}, got {gen_ids}"
        )

    def test_generation_carries_prompt_context(self, mock_transport: CallLog) -> None:
        """$generation event (from backend) carries prompt_id, prompt_version,
        and model metadata. Behavioral events (from frontend) don't need to."""
        client = LitmusClient(api_key="ltm_pk_test_abc", sync_mode=True)

        backend_gen = client.generation(
            "s1",
            prompt_id="summarizer",
            prompt_version="v3.1",
            metadata={"model": "claude-sonnet", "token_count": 512},
        )

        frontend_gen = client.attach(backend_gen.id, "s1")
        frontend_gen.event("$accept")

        events = mock_transport.events()
        gen_event = next(e for e in events if e["type"] == "$generation")
        accept_event = next(e for e in events if e["type"] == "$accept")

        # Backend $generation has full prompt context
        assert gen_event["prompt_id"] == "summarizer"
        assert gen_event["prompt_version"] == "v3.1"
        assert gen_event["metadata"]["model"] == "claude-sonnet"
        assert gen_event["metadata"]["token_count"] == 512

        # Frontend behavioral event doesn't need prompt context,
        # it correlates via generation_id
        assert "prompt_id" not in accept_event
        assert "prompt_version" not in accept_event

    def test_exactly_one_generation_event(self, mock_transport: CallLog) -> None:
        """No matter how many attach() calls, only one $generation is emitted."""
        client = LitmusClient(api_key="ltm_pk_test_abc", sync_mode=True)

        gen = client.generation("s1", prompt_id="chat-v1")

        # Multiple "consumers" attach to the same generation
        handle_a = client.attach(gen.id, "s1")
        handle_b = client.attach(gen.id, "s1")

        handle_a.event("$view")
        handle_b.event("$copy")
        handle_a.event("$accept")

        events = mock_transport.events()
        gen_events = [e for e in events if e["type"] == "$generation"]
        assert len(gen_events) == 1

        # But all 4 events (1 gen + 3 behavioral) share the same id
        assert len(events) == 4
        assert all(e["generation_id"] == gen.id for e in events)

    def test_session_id_consistent(self, mock_transport: CallLog) -> None:
        """Both sides must use the same session_id for the join to work."""
        client = LitmusClient(api_key="ltm_pk_test_abc", sync_mode=True)

        gen = client.generation("conversation-42", prompt_id="chat")
        frontend = client.attach(gen.id, "conversation-42")
        frontend.event("$accept")

        events = mock_transport.events()
        session_ids = {e["session_id"] for e in events}
        assert session_ids == {"conversation-42"}


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
        """shutdown() -> flush() -> queue.join() is the sync primitive.
        The constructor fires a $startup event, so we expect at least 2 calls."""
        client = LitmusClient(
            api_key="ltm_pk_test_abc",
            flush_at=1,
            flush_interval=0.1,
        )
        event_id = client.track(event_type="$accept", session_id="s1")
        client.shutdown()

        # $startup + $accept = at least 2 calls
        assert len(mock_transport.calls) >= 2
        events = mock_transport.events()
        accept_event = next(e for e in events if e["type"] == "$accept")
        assert accept_event["id"] == event_id
        # $startup is also present
        assert any(e["type"] == "$startup" for e in events)

    def test_threaded_batching(self, mock_transport: CallLog) -> None:
        client = LitmusClient(
            api_key="ltm_pk_test_abc",
            flush_at=5,
            flush_interval=10,
        )
        for i in range(5):
            client.track(event_type="$view", session_id=f"s{i}")
        client.shutdown()

        assert len(mock_transport.calls) >= 1
        # 5 user events + 1 $startup = 6 total, batched in groups of 5
        events = mock_transport.events()
        view_events = [e for e in events if e["type"] == "$view"]
        assert len(view_events) == 5

    def test_threaded_retry_on_500(self) -> None:
        import litmus.request as req

        log = CallLog()
        log.add_response(500, {"error": "boom"})
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
            client.shutdown()

            # $startup hits the 500 first (retry succeeds with 202),
            # then $accept also sends. At least 3 calls total.
            assert len(log.calls) >= 3
        finally:
            req._client.close()
            req._client = old_client


class TestStartup:
    """$startup fires automatically in threaded mode with env metadata."""

    def test_startup_fires_in_threaded_mode(self, mock_transport: CallLog) -> None:
        client = LitmusClient(
            api_key="ltm_pk_test_abc",
            flush_at=1,
            flush_interval=0.1,
        )
        client.shutdown()

        events = mock_transport.events()
        startup = next((e for e in events if e["type"] == "$startup"), None)
        assert startup is not None
        assert startup["session_id"] == ""
        assert startup["metadata"]["$lib"] == "litmus-python"
        assert "platform" in startup["metadata"]
        assert "python_version" in startup["metadata"]
        assert "runtime" in startup["metadata"]

    def test_startup_skipped_in_sync_mode(self, mock_transport: CallLog) -> None:
        """sync_mode is for serverless; $startup would block the constructor."""
        LitmusClient(api_key="ltm_pk_test_abc", sync_mode=True)
        assert len(mock_transport.calls) == 0

    def test_startup_skipped_when_disabled(self) -> None:
        client = LitmusClient(api_key="ltm_pk_test_abc", disabled=True, send=False)
        # disabled clients drop everything, track() returns None
        result = client.track(event_type="$accept", session_id="s1")
        assert result is None


class TestQueueFull:
    def test_returns_none_when_full(self) -> None:
        client = LitmusClient(api_key="ltm_pk_test_abc", max_queue_size=1, send=False)
        client.send = True
        client._queue.put({"fake": True})
        result = client.track(event_type="$view", session_id="s1")
        assert result is None
