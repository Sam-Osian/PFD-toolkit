from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from typing import Callable

from django.conf import settings
from django.db import connection

logger = logging.getLogger("pfd_workbench.performance")


@dataclass
class _QueryStats:
    total_count: int = 0
    slow_count: int = 0
    total_duration_ms: float = 0.0


class _QueryTimingWrapper:
    def __init__(
        self,
        *,
        request_id: str,
        method: str,
        path: str,
        threshold_ms: float,
        stats: _QueryStats,
    ) -> None:
        self.request_id = request_id
        self.method = method
        self.path = path
        self.threshold_ms = threshold_ms
        self.stats = stats

    def __call__(self, execute: Callable, sql: str, params, many: bool, context):
        started_at = time.perf_counter()
        try:
            return execute(sql, params, many, context)
        finally:
            elapsed_ms = (time.perf_counter() - started_at) * 1000
            self.stats.total_count += 1
            self.stats.total_duration_ms += elapsed_ms
            if elapsed_ms < self.threshold_ms:
                return

            self.stats.slow_count += 1
            sql_preview = " ".join(str(sql).split())
            if len(sql_preview) > 280:
                sql_preview = f"{sql_preview[:280]}..."
            logger.warning(
                (
                    "slow_query request_id=%s method=%s path=%s duration_ms=%.2f "
                    "db_alias=%s sql=%s"
                ),
                self.request_id,
                self.method,
                self.path,
                elapsed_ms,
                getattr(context.get("connection"), "alias", "default"),
                sql_preview,
            )


class PerformanceInstrumentationMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response
        self.enabled = bool(getattr(settings, "PERF_REQUEST_LOGGING_ENABLED", True))
        self.slow_request_ms = float(getattr(settings, "PERF_SLOW_REQUEST_MS", 800))
        self.slow_query_ms = float(getattr(settings, "PERF_SLOW_QUERY_MS", 200))
        self.add_timing_header = bool(getattr(settings, "PERF_ADD_RESPONSE_TIMING_HEADER", False))

    def __call__(self, request):
        if not self.enabled:
            return self.get_response(request)

        request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex[:12])
        method = request.method
        path = request.get_full_path()
        started_at = time.perf_counter()
        stats = _QueryStats()
        query_wrapper = _QueryTimingWrapper(
            request_id=request_id,
            method=method,
            path=path,
            threshold_ms=self.slow_query_ms,
            stats=stats,
        )

        with connection.execute_wrapper(query_wrapper):
            response = self.get_response(request)

        duration_ms = (time.perf_counter() - started_at) * 1000
        if self.add_timing_header:
            response["X-Request-ID"] = request_id
            response["X-Request-Duration-Ms"] = f"{duration_ms:.2f}"

        if duration_ms >= self.slow_request_ms:
            logger.warning(
                (
                    "slow_request request_id=%s method=%s path=%s duration_ms=%.2f "
                    "query_count=%d slow_query_count=%d query_total_ms=%.2f status=%s"
                ),
                request_id,
                method,
                path,
                duration_ms,
                stats.total_count,
                stats.slow_count,
                stats.total_duration_ms,
                getattr(response, "status_code", "unknown"),
            )

        return response
