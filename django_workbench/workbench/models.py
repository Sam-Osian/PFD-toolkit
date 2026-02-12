from django.db import models
from uuid import uuid4


class Workbook(models.Model):
    share_number = models.PositiveIntegerField(unique=True, db_index=True)
    public_id = models.UUIDField(default=uuid4, unique=True, editable=False, db_index=True)
    edit_token = models.UUIDField(default=uuid4, editable=False)
    title = models.CharField(max_length=120)
    snapshot = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at"]

    def __str__(self) -> str:
        return self.title or f"Workbook {self.public_id}"
