import django.db.models.deletion
import django.utils.timezone
import uuid
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("wb_workspaces", "0006_workspace_current_revision"),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name="WorkspaceLLMSetting",
            fields=[
                ("id", models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                (
                    "provider",
                    models.CharField(
                        choices=[("openai", "OpenAI"), ("openrouter", "OpenRouter")],
                        default="openai",
                        max_length=16,
                    ),
                ),
                ("model_name", models.CharField(default="gpt-4.1-mini", max_length=255)),
                ("max_parallel_workers", models.PositiveSmallIntegerField(default=1)),
                ("created_at", models.DateTimeField(default=django.utils.timezone.now, editable=False)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                (
                    "user",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="workspace_llm_settings",
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
                (
                    "workspace",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="llm_settings",
                        to="wb_workspaces.workspace",
                    ),
                ),
            ],
            options={},
        ),
        migrations.AddConstraint(
            model_name="workspacellmsetting",
            constraint=models.UniqueConstraint(
                fields=("workspace", "user"),
                name="uniq_workspace_llm_setting_scope",
            ),
        ),
        migrations.AddConstraint(
            model_name="workspacellmsetting",
            constraint=models.CheckConstraint(
                check=models.Q(max_parallel_workers__gte=1)
                & models.Q(max_parallel_workers__lte=32),
                name="chk_workspace_llm_setting_workers_bounds",
            ),
        ),
        migrations.AddIndex(
            model_name="workspacellmsetting",
            index=models.Index(fields=["workspace", "user"], name="idx_ws_llm_scope"),
        ),
        migrations.AddIndex(
            model_name="workspacellmsetting",
            index=models.Index(fields=["provider"], name="idx_ws_llm_provider"),
        ),
    ]
