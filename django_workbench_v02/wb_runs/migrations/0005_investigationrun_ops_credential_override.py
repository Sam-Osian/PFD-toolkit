from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("wb_workspaces", "0009_alter_userllmcredential_provider_and_more"),
        ("wb_runs", "0004_investigationrun_approval_note_and_more"),
    ]

    operations = [
        migrations.AddField(
            model_name="investigationrun",
            name="ops_override_base_url",
            field=models.URLField(blank=True),
        ),
        migrations.AddField(
            model_name="investigationrun",
            name="ops_override_encrypted_api_key",
            field=models.TextField(blank=True),
        ),
        migrations.AddField(
            model_name="investigationrun",
            name="ops_override_key_last4",
            field=models.CharField(blank=True, max_length=4),
        ),
        migrations.AddField(
            model_name="investigationrun",
            name="ops_override_provider",
            field=models.CharField(
                blank=True,
                choices=[("local_ollama", "Local (Ollama)"), ("openai", "OpenAI"), ("openrouter", "OpenRouter")],
                max_length=16,
            ),
        ),
    ]
