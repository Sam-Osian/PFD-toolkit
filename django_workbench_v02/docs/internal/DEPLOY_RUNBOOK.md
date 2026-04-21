# Deploy Runbook (Railway, v0.2)

Status: Active  
Last updated: 2026-04-21

## 1. Scope

This runbook covers:

1. Safe deploy flow for `pfd-toolkit-v02` on Railway.
2. Smoke tests after deploy.
3. Rollback and recovery, including snapshot-failure workaround.

Project:

1. `pfd-toolkit-v02`
2. Project ID: `641fab32-8cac-4134-b996-81eea714664c`

Services:

1. `web`
2. `worker`
3. `notification-dispatcher`
4. `Postgres`

## 2. Pre-Deploy Checklist

1. Confirm local tests pass.
2. Confirm migrations are generated and committed.
3. Confirm `CREDENTIAL_ENCRYPTION_KEY` is set on `web`, `worker`, `notification-dispatcher`.
4. Confirm SMTP/Auth0 vars are present on required services.
5. Confirm `RAILWAY_DOCKERFILE_PATH=Dockerfile.railway.v02` on app services.

## 3. Standard Deploy

From repo root:

```bash
railway link --project 641fab32-8cac-4134-b996-81eea714664c
railway up -s web --detach
railway up -s worker --detach
railway up -s notification-dispatcher --detach
railway service status --all --json
```

Expected end state:

1. `web`: `SUCCESS`
2. `worker`: `SUCCESS`
3. `notification-dispatcher`: `SUCCESS`
4. `Postgres`: `SUCCESS`

## 4. Snapshot Failure Recovery

Symptom:

1. Deployment fails with: `Failed to create code snapshot`.

Workaround (deploy from clean staging directory):

```bash
rm -rf /tmp/pfdtoolkit-v02-railway-deploy
mkdir -p /tmp/pfdtoolkit-v02-railway-deploy

cp -r Dockerfile.railway.v02 pyproject.toml uv.lock README.md src django_workbench_v02 /tmp/pfdtoolkit-v02-railway-deploy/
cp .railwayignore /tmp/pfdtoolkit-v02-railway-deploy/

cd /tmp/pfdtoolkit-v02-railway-deploy
railway link --project 641fab32-8cac-4134-b996-81eea714664c
railway up -s web --detach
railway up -s worker --detach
railway up -s notification-dispatcher --detach
railway service status --all --json
```

Notes:

1. This avoids snapshotting unrelated files in the full working directory.
2. Use this only when normal `railway up` from repo root fails with snapshot errors.

## 5. Smoke Test Procedure

## 5.1 CLI-Verifiable Checks

1. Service health:
```bash
railway service status --all --json
```
2. Home page responds:
```bash
curl -sS -D - -o /dev/null https://web-production-b259d.up.railway.app/ | sed -n '1,25p'
```
Expect: `HTTP/2 200`

3. Login starts Auth0 flow:
```bash
curl -sS -D - -o /dev/null https://web-production-b259d.up.railway.app/auth/login/ | sed -n '1,40p'
```
Expect: `HTTP/2 302` with `location: https://oreliandata.uk.auth0.com/authorize?...`

4. Verify live dataset metadata (row count, date range, fingerprint, package version):
```bash
railway run -s web -- uv run python manage.py report_pfd_dataset --json
```

## 5.2 Manual Browser Checks (Required)

1. Open app and login via Auth0.
2. Open a workspace where user can run workflows.
3. Save credential under `LLM Credentials`.
4. Queue a real `filter` run.
5. Confirm run reaches `succeeded`.
6. Download output artifact from run detail page.
7. Queue a run with completion notification enabled.
8. Confirm completion email arrives from `DEFAULT_FROM_EMAIL`.

## 5.3 2026-04-19 Smoke Results

Completed via CLI:

1. All services healthy (`SUCCESS`).
2. Home endpoint returns `200`.
3. Auth login endpoint returns `302` to Auth0 authorize URL.

Pending manual confirmation:

1. In-app credential save.
2. Real run execution and artifact download.
3. Completion email receipt.

## 6. Rollback

If latest deploy is unhealthy:

1. Check service/deployment status:
```bash
railway service status --all --json
railway deployment list -s web --json
```
2. Attempt redeploy/restart if deployment is restartable:
```bash
railway redeploy -s web -y
railway restart -s worker -y
```
3. If deploy object is non-restartable and snapshot path is failing, use the clean staging deploy method from section 4.

## 7. Post-Deploy Log Review

1. `web`: confirm migrations applied and gunicorn booted.
2. `worker`: confirm polling loop started without DB errors.
3. `notification-dispatcher`: confirm loop started and no SMTP config errors.

Commands:

```bash
railway logs -s web --deployment --latest --lines 200
railway logs -s worker --deployment --latest --lines 200
railway logs -s notification-dispatcher --deployment --latest --lines 200
```

## 8. Scheduler Baseline

Add or confirm these scheduled jobs in Railway:

1. Collection counters snapshot refresh every 6 hours:
```bash
uv run python manage.py refresh_collection_cards_snapshot
```
2. Lifecycle maintenance (existing):
```bash
uv run python manage.py run_lifecycle_maintenance
```
