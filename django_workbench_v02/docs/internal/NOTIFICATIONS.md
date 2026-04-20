# Completion Notifications (v0.2)

Status: Implemented  
Last updated: 2026-04-20

## 1. Purpose

Users can optionally request an email when a queued run reaches a terminal state.

Supported channel:

1. `email`

Supported triggers:

1. `success`
2. `failure` (`failed`, `timed_out`, `cancelled`)
3. `any`

## 2. Queue-Time Request

When queueing a run, users can set:

1. `request_completion_email` (boolean)
2. `notify_on` (`success`/`failure`/`any`)

If enabled, a `NotificationRequest(status=pending)` record is created.

## 3. Dispatch Model

Dispatch is decoupled from run execution:

1. Run worker only updates run lifecycle/status.
2. Notification dispatcher processes pending requests for terminal runs.
3. Email send failures do not affect run status.

Dispatcher command:

```bash
uv run python manage.py run_notification_dispatcher --once
```

Loop mode:

```bash
uv run python manage.py run_notification_dispatcher --poll-seconds 5 --max-items 50
```

## 4. Status Handling

`NotificationRequest.status` transitions:

1. `pending -> sent` on successful email delivery
2. `pending -> failed` on send error
3. `pending -> cancelled` when run terminal status does not match `notify_on`

## 5. Email Settings

Configured in `pfd_workbench_v02/settings.py`:

1. `EMAIL_BACKEND`
2. `EMAIL_HOST`
3. `EMAIL_PORT`
4. `EMAIL_USE_TLS`
5. `EMAIL_HOST_USER`
6. `EMAIL_HOST_PASSWORD`
7. `DEFAULT_FROM_EMAIL`
8. `WORKBENCH_BASE_URL`

## 6. Email Template Design

Notification emails are now sent as multipart messages:

1. Plain text fallback template:
   - `templates/wb_notifications/emails/run_status.txt`
2. HTML template:
   - `templates/wb_notifications/emails/run_status.html`

Status-specific variants are rendered from one template context:

1. `succeeded` -> "Run complete"
2. `failed` -> "Run failed"
3. `timed_out` -> "Run timed out"
4. `cancelled` -> "Run cancelled"

This keeps design consistency while adapting headline/color/copy to outcome.

## 7. Audit Events

Notification flow emits audit events:

1. `notification.requested`
2. `notification.sent`
3. `notification.failed_send_error`
4. `notification.failed_no_email`
5. `notification.cancelled_trigger_mismatch`
