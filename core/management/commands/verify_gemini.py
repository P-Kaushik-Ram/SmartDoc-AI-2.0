"""Metadata-only verification: never sends documents or generates tokens."""
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from core import gemini_ocr


class Command(BaseCommand):
    help = 'List approved free-tier Gemini models visible to the configured key; no OCR request.'

    def handle(self, *args, **options):
        if not settings.GEMINI_API_KEY:
            raise CommandError('Set GEMINI_API_KEY in your local .env file. Never paste it into chat.')
        try:
            with gemini_ocr.create_client() as client:
                available = {model.name.removeprefix('models/') for model in client.models.list()
                    if 'generateContent' in (model.supported_actions or [])}
            approved = sorted(available & gemini_ocr.FREE_TIER_MODELS)
            if not approved:
                raise CommandError('No approved free-tier model is accessible. Keep using local OCR; do not enable billing.')
            self.stdout.write('Visible approved models: ' + ', '.join(approved))
            self.stdout.write('Set GEMINI_MODEL to one of these names and restart. Confirm the API project remains on the Free tier in AI Studio; model listing cannot verify billing status or quota.')
        except CommandError:
            raise
        except Exception as exc:
            raise CommandError(gemini_ocr.safe_error(exc)) from None
