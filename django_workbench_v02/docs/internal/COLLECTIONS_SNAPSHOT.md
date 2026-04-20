# Collections Counter Snapshot

Status: Implemented  
Last updated: 2026-04-20

## Purpose

Keep `/collections/` fast by serving counters from a persisted snapshot instead of loading/processing the full report dataset on each request.

Counters now update only when refresh jobs run.

## Runtime Behavior

1. `GET /collections/` reads `CollectionCardSnapshot` from Postgres.
2. If no snapshot exists yet, page renders with a "snapshot not available" message.
3. Collection detail/copy endpoints still load dataset as needed.

## Refresh Command

```bash
uv run python manage.py refresh_collection_cards_snapshot
```

Optional (reuse in-process cache):

```bash
uv run python manage.py refresh_collection_cards_snapshot --no-force-refresh
```

## Railway Scheduler Recommendation

Create a scheduled job targeting the `web` service command:

```bash
uv run python manage.py refresh_collection_cards_snapshot
```

Suggested cadence:

1. Every 6 hours for near-live counters.
2. Increase/decrease based on acceptable staleness vs compute cost.

## Deploy Note

After first deploy with this feature, run the refresh command once so counters appear immediately.
