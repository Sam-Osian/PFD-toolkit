# Django Workbench (Standalone Port)

This directory contains a standalone Django recreation of the Workbench.
It is independent and does not import or reference Streamlit code.

## Run locally

1. Install dependencies (from repo root):
   - `uv sync`
2. Run migrations (first time only):
   - `UV_CACHE_DIR=/tmp/uv-cache uv run python django_workbench/manage.py migrate`
3. Start the Django server:
   - `UV_CACHE_DIR=/tmp/uv-cache uv run python django_workbench/manage.py runserver`
4. Open:
   - `http://127.0.0.1:8000/`

## Notes

- The UI and workflow mirror the Streamlit version: report loading, screener, theme discovery, feature extraction, download bundle, and undo/redo/start-over.
- Workspace state is session-backed.
- Styling and layout are ported to Django templates/static assets while preserving the same visual direction.
