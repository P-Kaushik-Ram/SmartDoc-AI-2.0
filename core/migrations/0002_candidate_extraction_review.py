from django.db import migrations, models


def mark_existing_extractions(apps, schema_editor):
    Candidate = apps.get_model('core', 'Candidate')
    for candidate in Candidate.objects.exclude(extracted_text='').iterator():
        candidate.extraction_status = 'completed'
        candidate.extraction_method = 'legacy_extraction'
        candidate.save(update_fields=['extraction_status', 'extraction_method'])


class Migration(migrations.Migration):
    dependencies = [('core', '0001_initial')]

    operations = [
        migrations.AddField(
            model_name='candidate',
            name='extraction_status',
            field=models.CharField(
                choices=[('pending', 'Pending'), ('completed', 'Completed'), ('needs_review', 'Needs review'), ('failed', 'Failed')],
                default='pending', max_length=20,
            ),
        ),
        migrations.AddField(
            model_name='candidate',
            name='extraction_method',
            field=models.CharField(blank=True, max_length=40),
        ),
        migrations.AddField(
            model_name='candidate',
            name='extraction_error',
            field=models.CharField(blank=True, max_length=500),
        ),
        migrations.AddField(
            model_name='candidate',
            name='reviewed_text',
            field=models.TextField(blank=True),
        ),
        migrations.RunPython(mark_existing_extractions, migrations.RunPython.noop),
    ]
