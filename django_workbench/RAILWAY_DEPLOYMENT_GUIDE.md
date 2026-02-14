# Django Deployment Guide (Railway) for pfdtoolkit.org

This guide migrates you from:
- MkDocs at `https://pfdtoolkit.org/`
- Streamlit at `https://workbench.pfdtoolkit.org/`

to a **single Django deployment** on Railway at `https://pfdtoolkit.org/`.

## 1. Target Architecture

- One Railway service running Django + Gunicorn.
- One Railway Postgres database.
- Django serves app pages, docs pages (`/for-coders/`), `robots.txt`, and `sitemap.xml`.
- `workbench.pfdtoolkit.org` removed from DNS and Railway.

## 2. Important Pre-Cutover Risks

### Risk A: SQLite on Railway is not suitable
Your current Django settings default to SQLite (`django_workbench/pfd_workbench/settings.py`). Railway containers are ephemeral, so SQLite data can be lost.

Use Railway Postgres and `DATABASE_URL`.

### Risk B: Old MkDocs URLs will 404 after cutover
Current Django routes do not serve old root docs paths like `/llm_setup/` or `/reference/loader/`.

Before cutover, add redirects from old MkDocs paths to your Django docs routes (`/for-coders/?doc=...`), or keep equivalent pages at old paths.

## 3. Update Django Settings for Production

Edit `django_workbench/pfd_workbench/settings.py` to support env-based production config.

## 3.1 Add imports
```python
import os
import dj_database_url
```

## 3.2 Replace secret/debug/hosts
```python
SECRET_KEY = os.environ["DJANGO_SECRET_KEY"]
DEBUG = os.getenv("DJANGO_DEBUG", "False").lower() == "true"

ALLOWED_HOSTS = [
    host.strip()
    for host in os.getenv(
        "DJANGO_ALLOWED_HOSTS",
        "pfdtoolkit.org,www.pfdtoolkit.org"
    ).split(",")
    if host.strip()
]

CSRF_TRUSTED_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "DJANGO_CSRF_TRUSTED_ORIGINS",
        "https://pfdtoolkit.org,https://www.pfdtoolkit.org"
    ).split(",")
    if origin.strip()
]
```

## 3.3 Replace DATABASES
```python
DATABASES = {
    "default": dj_database_url.config(
        default=f"sqlite:///{BASE_DIR / 'db.sqlite3'}",
        conn_max_age=600,
        ssl_require=not DEBUG,
    )
}
```

## 3.4 Static files with WhiteNoise

Add WhiteNoise middleware directly after `SecurityMiddleware`:
```python
"django.middleware.security.SecurityMiddleware",
"whitenoise.middleware.WhiteNoiseMiddleware",
```

Add static config:
```python
STATIC_URL = "/static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
STATICFILES_DIRS = [BASE_DIR / "static"]
STATICFILES_STORAGE = "whitenoise.storage.CompressedManifestStaticFilesStorage"
```

## 3.5 HTTPS/proxy settings
```python
SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
SECURE_SSL_REDIRECT = os.getenv("DJANGO_SECURE_SSL_REDIRECT", "True").lower() == "true"
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True
```

## 3.6 Optional: security hardening
```python
SECURE_HSTS_SECONDS = 60 if DEBUG else 31536000
SECURE_HSTS_INCLUDE_SUBDOMAINS = not DEBUG
SECURE_HSTS_PRELOAD = not DEBUG
```

## 4. Railway Service Setup

## 4.1 Create service
- In Railway, create a new service from this repo.
- Root directory: repo root.

## 4.2 Add Postgres
- Add Railway Postgres to the project.
- Confirm `DATABASE_URL` is injected into your Django service.

## 4.3 Build/Start commands
Use these in Railway service settings:

- Build Command:
```bash
uv sync --frozen --no-dev
```

- Start Command:
```bash
uv run python django_workbench/manage.py collectstatic --noinput && uv run python django_workbench/manage.py migrate && uv run gunicorn pfd_workbench.wsgi:application --chdir django_workbench --bind 0.0.0.0:$PORT
```

## 4.4 Environment variables
Set these in Railway:

- `DJANGO_SECRET_KEY` = long random string
- `DJANGO_DEBUG` = `False`
- `DJANGO_ALLOWED_HOSTS` = `pfdtoolkit.org,www.pfdtoolkit.org`
- `DJANGO_CSRF_TRUSTED_ORIGINS` = `https://pfdtoolkit.org,https://www.pfdtoolkit.org`
- `DJANGO_SECURE_SSL_REDIRECT` = `True`
- `OPENAI_API_KEY` and any other app provider keys you use

Do **not** set `DATABASE_URL` manually if Railway already injects it from Postgres.

## 5. Pre-Go-Live Validation (Railway Preview URL)

Before moving DNS:

1. Open Railway generated domain and test:
   - `/`
   - `/explore-pfds/`
   - `/analyse-themes/`
   - `/extract-data/`
   - `/for-coders/`
   - `/robots.txt`
   - `/sitemap.xml`
2. Confirm static assets load (CSS/JS/icons).
3. Create/edit/share a workbook to confirm DB persistence.
4. Confirm `https` canonical URLs in rendered metadata.
5. Verify old MkDocs URLs are redirected (if you implemented compatibility redirects).

## 6. DNS + Domain Cutover

## 6.1 Lower TTL first
24 hours before cutover, reduce DNS TTL for:
- `pfdtoolkit.org`
- `www.pfdtoolkit.org`
- `workbench.pfdtoolkit.org`

to ~300 seconds.

## 6.2 Attach custom domains in Railway
In Railway domain settings, add:
- `pfdtoolkit.org`
- `www.pfdtoolkit.org`

Railway will give DNS targets/records. Apply exactly as shown.

## 6.3 Switch apex/root
Update DNS so `pfdtoolkit.org` points to Railway target (provider-dependent ALIAS/ANAME/CNAME flattening).

## 6.4 Switch www
Point `www.pfdtoolkit.org` to Railway as instructed.

## 6.5 Remove old workbench subdomain
Delete `workbench.pfdtoolkit.org` DNS record after confirming root works.

## 6.6 Remove old deployments
After successful cutover:
- Disable/remove MkDocs hosting deployment.
- Disable/remove Streamlit Railway service.

## 7. Rollback Plan (Keep for first 24-48h)

If severe issue appears:
1. Re-point `pfdtoolkit.org` DNS back to old MkDocs host.
2. Re-enable Streamlit service and restore `workbench.pfdtoolkit.org` DNS.
3. Investigate/fix Django service, then cut over again.

## 8. Cleanup After Successful Cutover

- Update public links in `README.md` if needed.
- Remove obsolete deployment config/docs for MkDocs + Streamlit.
- Keep one deployment runbook in-repo (this file).

## 9. Suggested Minimal Execution Order

1. Make Django production settings changes.
2. Add legacy docs URL redirects for old MkDocs paths.
3. Deploy to Railway preview URL and validate.
4. Add Railway custom domains.
5. DNS cutover for `pfdtoolkit.org` and `www`.
6. Remove `workbench.pfdtoolkit.org` DNS + service.
7. Decommission MkDocs deployment.

## 10. First-Time Django Deploy Checklist

- [ ] `DEBUG=False` in production
- [ ] secret key in env var
- [ ] Postgres configured (not SQLite)
- [ ] `collectstatic` runs successfully
- [ ] migrations run successfully
- [ ] `ALLOWED_HOSTS` and `CSRF_TRUSTED_ORIGINS` set
- [ ] HTTPS security settings enabled
- [ ] `robots.txt` and `sitemap.xml` accessible
- [ ] old docs URLs handled (redirect/compatibility)
