#!/usr/bin/env bash
set -euo pipefail

cd /app/django_workbench_v02
uv run python manage.py migrate --noinput
exec uv run gunicorn pfd_workbench_v02.wsgi:application --bind 0.0.0.0:${PORT} --workers 2 --timeout 180
