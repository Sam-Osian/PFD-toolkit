import django.db.models.deletion
import django.utils.timezone
import uuid
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("wb_workspaces", "0007_workspacellmsetting"),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name="UserLLMCredential",
            fields=[
                ("id", models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                (
                    "provider",
                    models.CharField(
                        choices=[("openai", "OpenAI"), ("openrouter", "OpenRouter")],
                        max_length=16,
                    ),
                ),
                ("encrypted_api_key", models.TextField()),
                ("base_url", models.URLField(blank=True)),
                ("key_last4", models.CharField(max_length=4)),
                ("last_used_at", models.DateTimeField(blank=True, null=True)),
                ("created_at", models.DateTimeField(default=django.utils.timezone.now, editable=False)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                (
                    "user",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="llm_credentials",
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
            ],
        ),
        migrations.CreateModel(
            name="UserLLMSetting",
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
                    models.OneToOneField(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="llm_setting",
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
            ],
        ),
        migrations.AddConstraint(
            model_name="userllmcredential",
            constraint=models.UniqueConstraint(
                fields=("user", "provider"),
                name="uniq_user_llm_credential_scope",
            ),
        ),
        migrations.AddConstraint(
            model_name="userllmsetting",
            constraint=models.CheckConstraint(
                check=models.Q(max_parallel_workers__gte=1) & models.Q(max_parallel_workers__lte=32),
                name="chk_user_llm_setting_workers_bounds",
            ),
        ),
        migrations.AddIndex(
            model_name="userllmcredential",
            index=models.Index(fields=["user"], name="idx_user_llm_cred_user"),
        ),
        migrations.AddIndex(
            model_name="userllmcredential",
            index=models.Index(fields=["provider"], name="idx_user_llm_cred_provider"),
        ),
        migrations.AddIndex(
            model_name="userllmsetting",
            index=models.Index(fields=["provider"], name="idx_user_llm_provider"),
        ),
    ]
