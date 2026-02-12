# Generated manually to backfill share_number for existing workbook rows.
from django.db import migrations, models


def backfill_workbook_share_numbers(apps, schema_editor):
    Workbook = apps.get_model("workbench", "Workbook")
    base = 100000
    for offset, workbook in enumerate(Workbook.objects.order_by("id"), start=1):
        workbook.share_number = base + offset
        workbook.save(update_fields=["share_number"])


def noop_reverse(apps, schema_editor):
    return None


class Migration(migrations.Migration):

    dependencies = [
        ("workbench", "0002_workbook_snapshot_delete_workbookrevision"),
    ]

    operations = [
        migrations.AddField(
            model_name="workbook",
            name="share_number",
            field=models.PositiveIntegerField(db_index=True, null=True),
        ),
        migrations.RunPython(backfill_workbook_share_numbers, noop_reverse),
        migrations.AlterField(
            model_name="workbook",
            name="share_number",
            field=models.PositiveIntegerField(db_index=True, unique=True),
        ),
    ]
