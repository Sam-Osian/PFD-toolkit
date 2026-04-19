from __future__ import annotations

from django.conf import settings


BOT_USER_AGENT_MARKERS = {
    "bot",
    "crawler",
    "spider",
    "slurp",
    "headless",
    "curl",
    "wget",
    "python-requests",
}


def is_human_view_request(*, request=None) -> bool:
    if request is None:
        return True
    user_agent = (request.META.get("HTTP_USER_AGENT") or "").strip().lower()
    if not user_agent:
        # Empty user-agent can come from tests and some privacy configurations.
        return True
    return not any(marker in user_agent for marker in BOT_USER_AGENT_MARKERS)


def should_update_last_viewed(*, existing_last_viewed_at, now) -> bool:
    debounce_seconds = int(getattr(settings, "LIFECYCLE_VIEW_DEBOUNCE_SECONDS", 0))
    if debounce_seconds <= 0:
        return True
    if existing_last_viewed_at is None:
        return True
    elapsed_seconds = (now - existing_last_viewed_at).total_seconds()
    return elapsed_seconds >= debounce_seconds
