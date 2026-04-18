# Auth + Permission Spine (v0.2)

Status: Implemented baseline  
Last updated: 2026-04-18

## 1. What This Delivers

1. Custom Django user model (`accounts.User`) with:
   - email as login identifier
   - first name
   - last name
   - optional Auth0 subject id (`auth0_sub`)
2. Auth0-backed login/logout/callback routes.
3. Workspace permission helpers for object-level authorization.
4. Admin bootstrap command for your single admin account.

## 2. Auth0 URL Allowlist Fields

Auth0 requires exact URL allowlists to prevent malicious redirects.

Use these values in your Auth0 Application settings:

1. Allowed Callback URLs
`http://127.0.0.1:8000/auth/callback/,https://pfdtoolkit.org/auth/callback/`

2. Allowed Logout URLs
`http://127.0.0.1:8000/,https://pfdtoolkit.org/`

3. Allowed Web Origins
`http://127.0.0.1:8000,https://pfdtoolkit.org`

If you use `www`, add `https://www.pfdtoolkit.org` variants too.

## 3. Environment Variables

Configured in `settings.py` with defaults for your tenant/domain:

1. `AUTH0_DOMAIN`
2. `AUTH0_CLIENT_ID`
3. `AUTH0_CLIENT_SECRET`
4. `AUTH0_CALLBACK_URL`
5. `AUTH0_POST_LOGOUT_REDIRECT_URI`
6. `PFD_ADMIN_EMAIL`

`api.env` at project root is auto-loaded in local development.

## 4. Admin Model

Policy baseline:

1. The configured `PFD_ADMIN_EMAIL` is auto-promoted to staff/superuser on Auth0 login.
2. Non-admin users are non-staff by default.
3. Admin UI can manually modify users, workspace memberships, and all domain models.

Bootstrap command:

```bash
uv run python manage.py ensure_admin_user \
  --email sam.osian@oreliandata.co.uk \
  --first-name Sam \
  --last-name Osian
```

You can supply `--password` only if you want local password-based admin login too.

## 5. Permission Spine

Implemented in `wb_workspaces/permissions.py`:

1. `can_view_workspace`
2. `can_edit_workspace`
3. `can_manage_members`
4. `can_manage_shares`
5. `can_run_workflows`
6. `workspace_permission_required(...)` decorator

Rules:

1. Public workspace: viewable without login.
2. Private workspace: requires membership.
3. Edit actions require `access_mode=edit`.
4. Membership/share management is owner-only (plus superuser override).
5. Run execution depends on `can_run_workflows`.

## 6. Current Endpoints

1. `/` landing
2. `/auth/login/`
3. `/auth/callback/`
4. `/auth/logout/`
5. `/workspaces/` (authenticated dashboard + create workspace)
6. `/workspaces/public/`
7. `/workspaces/<workspace_id>/`

## 7. Next Integration Work

1. Apply permission checks to all future investigation/run/share endpoints.
2. Add service-layer invariants for owner safeguards (cannot remove last owner).
3. Add audit event writes on membership and sharing changes.
4. Add integration tests for Auth0 callback and admin elevation path.
