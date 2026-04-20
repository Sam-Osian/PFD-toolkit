from __future__ import annotations

from django.db import models
from django.utils import timezone


class CollectionCardSnapshot(models.Model):
    key = models.CharField(max_length=64, unique=True, default="default")
    cards_json = models.JSONField(default=list)
    generated_at = models.DateTimeField(default=timezone.now)
    source_row_count = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(default=timezone.now, editable=False)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        indexes = [
            models.Index(fields=["key"], name="idx_col_snapshot_key"),
            models.Index(fields=["generated_at"], name="idx_col_snapshot_gen"),
        ]

    def __str__(self) -> str:
        return f"{self.key} @ {self.generated_at.isoformat()}"
