# litmus-sdk

Python SDK for [Litmus](https://trylitmus.com) - implicit evals for AI products.

## Install

```bash
pip install litmus-sdk
```

## Quick start

```python
from litmus import LitmusClient

client = LitmusClient(api_key="ltm_pk_live_...")

# Track a generation and user signals
gen = client.generation("session-123", prompt_id="content_gen")
gen.accept()
gen.edit(edit_distance=0.3)

# Flush before exit (serverless, scripts, etc.)
client.shutdown()
```

## How it works

Events are queued in memory and shipped to the Litmus ingest API on a
background thread. Batches are sent every 0.5s or when 100 events
accumulate (both configurable). The consumer retries transient failures
with exponential backoff.

For serverless environments, pass `sync_mode=True` to send inline.
