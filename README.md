# litmus-python-sdk

Python SDK for [Litmus](https://trylitmus.com) — implicit evals for AI products.

## Install

```bash
pip install litmus-python-sdk
```

## Quick start

```python
from litmus import LitmusClient

client = LitmusClient(api_key="ltm_pk_live_...")

# Track a generation and user signals
gen = client.generation("session-123", prompt_id="content_gen")
gen.event("$accept")
gen.event("$edit", edit_distance=0.3)
gen.event("$share", channel="slack")

# Custom events work too
gen.event("my_custom_signal", score=0.9)

# Flush before exit (serverless, scripts, etc.)
client.shutdown()
```

## How it works

Events are queued in memory and shipped to the Litmus ingest API on a
background thread. Batches are sent every 0.5s or when 100 events
accumulate (both configurable). The consumer retries transient failures
with exponential backoff.

For serverless environments, pass `sync_mode=True` to send inline.

All system events (`$accept`, `$edit`, `$copy`, etc.) get full autocomplete
via the `EventType` type. Custom event strings are accepted too.
