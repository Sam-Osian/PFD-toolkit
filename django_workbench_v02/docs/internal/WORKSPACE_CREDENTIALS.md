# Workspace Credentials (v0.2)

Status: Implemented  
Last updated: 2026-04-19

## 1. Purpose

Support async server-side runs without global provider keys by storing user API keys per workspace/provider in encrypted form.

## 2. Data Model

Model: `wb_workspaces.WorkspaceCredential`

Scope uniqueness:

1. `workspace`
2. `user`
3. `provider` (`openai` or `openrouter`)

Stored fields:

1. `encrypted_api_key` (ciphertext)
2. `key_last4` (operator-safe reference only)
3. `base_url` (optional provider endpoint override)
4. `last_used_at`

## 3. Encryption

Implementation: `wb_workspaces/credentials.py`

Key resolution order:

1. `CREDENTIAL_ENCRYPTION_KEY` env var (recommended)
2. fallback deterministic key derived from `SECRET_KEY` (developer convenience)

## 4. Run Queue Flow

When queueing a real run:

1. If user submits `api_key` and chooses save, app upserts encrypted credential.
2. If user submits no key, app requires an existing credential for selected provider.
3. Key material is never copied into `InvestigationRun.input_config_json`.

## 5. Worker Flow

During real adapter execution:

1. Adapter resolves credential for `(run.workspace, run.requested_by, provider)`.
2. Decrypts key in-process and creates `LLM(...)`.
3. Logs audit events for credential use without exposing plaintext.

## 6. Audit Events

1. `workspace.credential_saved`
2. `workspace.credential_used`

Payload includes provider and `key_last4` only.
