import io
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from PIL import Image
from django.core.files.uploadedfile import SimpleUploadedFile
from django.core.management import call_command, CommandError
from django.test import SimpleTestCase, TestCase, Client, override_settings
from django.urls import reverse

from .pdf_test_fixtures import pdf_bytes
from . import gemini_ocr as gemini
from .utils import DocumentProcessingError, ExtractionResult
from .workspaces import DocumentForm

TEXT = 'English தமிழ் हिन्दी తెలుగు മലയാളം\nSymbol | Frequency\nA | 6\nTotal | 10'
REAL_CREATE_CLIENT = gemini.create_client


def upload(name='sample.png'):
    buffer = io.BytesIO()
    Image.new('RGB', (40, 60), 'white').save(buffer, format='PNG')
    return SimpleUploadedFile(name, buffer.getvalue())


@override_settings(GEMINI_API_KEY='test-only-placeholder', GEMINI_MODEL='gemini-2.5-flash')
class GeminiServiceTests(SimpleTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'sample.png'
        self.path.write_bytes(upload().read())
        self.client = mock.MagicMock()
        self.client.__enter__.return_value = self.client
        self.client.models.get.return_value = SimpleNamespace(supported_actions=['generateContent'])
        self.client.models.generate_content.return_value = SimpleNamespace(text=TEXT, candidates=[])
        self.factory = mock.patch('core.gemini_ocr.create_client', return_value=self.client).start()
        self.addCleanup(mock.patch.stopall)

    def run_ocr(self, **kwargs):
        return gemini.extract_gemini_document(self.path, cloud_consent=True, **kwargs)

    def test_configuration_detection_does_not_initialize_client(self):
        self.assertEqual(gemini.configuration_message(), '')
        self.factory.assert_not_called()

    def test_sdk_client_retries_disabled_and_tls_verified(self):
        # Invoke the actual factory while intercepting the SDK constructor.
        from google import genai
        with mock.patch.object(genai, 'Client') as constructor:
            REAL_CREATE_CLIENT()
            options = constructor.call_args.kwargs['http_options']
            self.assertEqual(options.retry_options.attempts, 1)
            self.assertTrue(options.client_args['verify'].check_hostname)
            self.assertFalse(constructor.call_args.kwargs['vertexai'])

    @mock.patch('core.handwriting.recognize_handwriting_image')
    def test_gemini_does_not_run_local_recognizer(self, recognize):
        self.run_ocr()
        recognize.assert_not_called()

    def test_truncated_response_is_flagged(self):
        self.client.models.generate_content.return_value.candidates = [SimpleNamespace(finish_reason='MAX_TOKENS')]
        self.assertTrue(any('truncated' in w for w in self.run_ocr().warnings))

    def test_invalid_language_blocks_network(self):
        with self.assertRaises(DocumentProcessingError):
            self.run_ocr(languages=['invalid'])
        self.factory.assert_not_called()

    def test_wrong_image_extension_blocks_network(self):
        self.path = self.path.rename(self.path.with_suffix('.jpg'))
        with self.assertRaises(DocumentProcessingError):
            self.run_ocr()
        self.factory.assert_not_called()

    @override_settings(GEMINI_API_KEY='')
    def test_missing_key(self):
        with self.assertRaisesMessage(DocumentProcessingError, 'GEMINI_API_KEY'):
            self.run_ocr()
        self.factory.assert_not_called()

    @override_settings(GEMINI_MODEL='')
    def test_missing_model(self):
        with self.assertRaisesMessage(DocumentProcessingError, 'GEMINI_MODEL'):
            self.run_ocr()

    @override_settings(GEMINI_MODEL='paid-or-unknown-model')
    def test_no_paid_model_or_fallback(self):
        with self.assertRaisesMessage(DocumentProcessingError, 'not approved'):
            self.run_ocr()
        self.factory.assert_not_called()

    def test_consent_enforced_at_service_boundary(self):
        with self.assertRaisesMessage(DocumentProcessingError, 'consent'):
            gemini.extract_gemini_document(self.path)
        self.factory.assert_not_called()

    def test_model_access_verified_before_generation(self):
        self.run_ocr()
        self.client.models.get.assert_called_once_with(model='gemini-2.5-flash')
        self.assertEqual(self.client.models.generate_content.call_args.kwargs['model'], 'gemini-2.5-flash')

    def test_incompatible_model(self):
        self.client.models.get.return_value.supported_actions = []
        with self.assertRaisesMessage(DocumentProcessingError, 'does not support'):
            self.run_ocr()
        self.client.models.generate_content.assert_not_called()

    def test_unicode_response_and_table_values_unchanged(self):
        self.assertEqual(self.run_ocr().text, '[Page 1]\n' + TEXT)

    def test_request_image_and_faithful_instruction(self):
        self.run_ocr(languages=['tam', 'hin'])
        args = self.client.models.generate_content.call_args.kwargs
        self.assertEqual(args['contents'][0], 'Selected languages: Tamil, Hindi.')
        self.assertEqual(args['contents'][1].inline_data.mime_type, 'image/jpeg')
        self.assertIn('Do not translate', args['config'].system_instruction)
        self.assertIn('do not invent missing cells', args['config'].system_instruction)
        self.assertEqual(args['config'].http_options.retry_options.attempts, 1)
        self.assertLessEqual(args['config'].http_options.timeout, 30000)

    def test_empty_response(self):
        self.client.models.generate_content.return_value.text = ''
        with self.assertRaisesMessage(DocumentProcessingError, 'no transcription'):
            self.run_ocr()

    def test_timeout(self):
        self.client.models.generate_content.side_effect = TimeoutError('private')
        with self.assertRaisesMessage(DocumentProcessingError, 'timed out'):
            self.run_ocr()
        self.assertEqual(self.client.models.generate_content.call_count, 1)

    def error(self, code):
        exception = RuntimeError('private credential must not leak')
        exception.code = code
        self.client.models.get.side_effect = exception
        return exception

    def test_invalid_key_redacted(self):
        self.error(403)
        with self.assertRaisesMessage(DocumentProcessingError, 'API key') as error:
            self.run_ocr()
        self.assertNotIn('private', str(error.exception))

    def test_model_unavailable(self):
        self.error(404)
        with self.assertRaisesMessage(DocumentProcessingError, 'model is unavailable'):
            self.run_ocr()

    def test_rate_limit(self):
        self.error(429)
        with self.assertRaisesMessage(DocumentProcessingError, 'rate limit'):
            self.run_ocr()

    def test_quota_exhaustion(self):
        self.error(429)
        with self.assertRaisesMessage(DocumentProcessingError, 'quota'):
            self.run_ocr()

    def test_internet_unavailable(self):
        self.client.models.get.side_effect = ConnectionError('secret')
        with self.assertRaisesMessage(DocumentProcessingError, 'internet access'):
            self.run_ocr()

    def test_service_failure(self):
        self.error(500)
        with self.assertRaisesMessage(DocumentProcessingError, 'No automatic retry'):
            self.run_ocr()

    def test_corrupted_image_never_sent(self):
        self.path.write_bytes(b'\x89PNG\r\n\x1a\ninvalid')
        with self.assertRaises(DocumentProcessingError):
            self.run_ocr()
        self.factory.assert_not_called()

    def pdf(self, pages):
        self.path = self.path.with_suffix('.pdf')
        self.path.write_bytes(pdf_bytes([''] * pages, page_size=(80, 80)))

    def test_pdf_order_and_one_call_per_page(self):
        self.pdf(2)
        result = self.run_ocr()
        self.assertEqual(result.page_count, 2)
        self.assertEqual(self.client.models.generate_content.call_count, 2)
        self.assertLess(result.text.index('[Page 1]'), result.text.index('[Page 2]'))

    def test_partial_pdf_stops_without_retries(self):
        self.pdf(3)
        self.client.models.generate_content.side_effect = [SimpleNamespace(text=TEXT, candidates=[]), TimeoutError()]
        result = self.run_ocr()
        self.assertEqual(self.client.models.generate_content.call_count, 2)
        self.assertIn('[Page 2: not processed]', result.text)
        self.assertIn('[Page 3: not processed]', result.text)
        self.assertTrue(any('Page 2 failed' in w for w in result.warnings))

    def test_pdf_limit_before_network(self):
        self.pdf(6)
        with self.assertRaisesMessage(DocumentProcessingError, '1 to 5 pages'):
            self.run_ocr()
        self.factory.assert_not_called()

    def test_encrypted_pdf_is_rejected_before_network(self):
        from reportlab.lib.pdfencrypt import StandardEncryption

        self.path = self.path.with_suffix('.pdf')
        encryption = StandardEncryption('', ownerPassword='fixture-owner')
        self.path.write_bytes(pdf_bytes(['Synthetic protected document.'], encrypt=encryption))
        with self.assertRaisesMessage(DocumentProcessingError, 'unencrypted PDFs'):
            self.run_ocr()
        self.factory.assert_not_called()

    def test_invalid_pdf_before_network(self):
        self.path = self.path.with_suffix('.pdf')
        self.path.write_bytes(b'%PDF-invalid')
        with self.assertRaises(DocumentProcessingError):
            self.run_ocr()
        self.factory.assert_not_called()

    def test_total_time_budget(self):
        self.pdf(2)
        with mock.patch('core.gemini_ocr.time.monotonic', side_effect=[0, 1, 121]):
            result = self.run_ocr()
        self.assertEqual(self.client.models.generate_content.call_count, 1)
        self.assertIn('[Page 2: not processed]', result.text)

    def test_jpeg(self):
        self.path = self.path.with_suffix('.jpg')
        Image.new('RGB', (30, 30)).save(self.path)
        self.assertIn(TEXT, self.run_ocr().text)

    def test_listing_only_no_transcription(self):
        self.client.models.list.return_value = [SimpleNamespace(name='models/gemini-2.5-flash', supported_actions=['generateContent'])]
        output = io.StringIO()
        call_command('verify_gemini', stdout=output)
        self.assertIn('gemini-2.5-flash', output.getvalue())
        self.client.models.generate_content.assert_not_called()

    def test_no_approved_model_access(self):
        self.client.models.list.return_value = []
        with self.assertRaisesMessage(CommandError, 'No approved'):
            call_command('verify_gemini')


class GeminiWorkspaceTests(TestCase):
    def post(self, **kwargs):
        return self.client.post(reverse('document_studio'), {'mode':'handwriting', 'engine':'gemini', 'document':upload(), **kwargs}, HTTP_ACCEPT='application/json')

    @mock.patch('core.workspaces.extract_gemini_document')
    def test_no_consent_no_cloud_call(self, extract):
        self.assertEqual(self.post().status_code, 400)
        extract.assert_not_called()

    @mock.patch('core.workspaces.extract_gemini_document', return_value=ExtractionResult(TEXT, 'handwriting_gemini', [], 1))
    def test_explicit_cloud_dispatch_and_unicode_editor(self, extract):
        response = self.post(cloud_consent='on', languages=['tam', 'hin'])
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['text'], TEXT)
        self.assertEqual(response.json()['language_names'], ['Tamil', 'Hindi'])
        self.assertTrue(extract.call_args.kwargs['cloud_consent'])

    def test_each_and_mixed_language_form(self):
        for selected in [['eng'], ['tam'], ['hin'], ['tel'], ['mal'], ['eng','tam','hin','tel','mal']]:
            with self.subTest(languages=selected):
                form = DocumentForm({'mode':'handwriting','engine':'gemini','cloud_consent':'on','languages':selected}, {'document':upload()})
                self.assertTrue(form.is_valid(), form.errors)
                self.assertEqual(form.cleaned_data['languages'], selected)

    def test_trocr_non_english_rejected(self):
        self.assertEqual(self.post(engine='trocr', languages=['tam']).status_code, 400)

    @mock.patch('core.workspaces.extract_gemini_document')
    def test_printed_never_dispatches_cloud(self, extract):
        with mock.patch('core.workspaces.extract_document', return_value=ExtractionResult('local', 'ocr', [], 1)):
            self.post(mode='printed', cloud_consent='on')
        extract.assert_not_called()

    @mock.patch('core.workspaces.extract_gemini_document')
    def test_resume_never_dispatches_cloud(self, extract):
        with mock.patch('core.workspaces.process_upload', return_value=ExtractionResult('Alex Python engineer', 'local', [], 1)):
            self.client.post(reverse('resume_analyzer'), {'document':upload(), 'engine':'gemini', 'cloud_consent':'on'})
        extract.assert_not_called()

    @mock.patch('core.workspaces.extract_handwritten_document', return_value=ExtractionResult('local', 'local', [], 1))
    @mock.patch('core.workspaces.extract_gemini_document')
    def test_local_engines_do_not_call_gemini(self, cloud, local):
        for engine in ['trocr', 'tesseract', 'hybrid']:
            self.assertEqual(self.post(engine=engine).status_code, 200)
        self.assertEqual(local.call_count, 3)
        cloud.assert_not_called()

    @override_settings(GEMINI_API_KEY='')
    def test_missing_configuration_visible_and_local_choices_preserved(self):
        response = self.client.get(reverse('document_studio') + '?mode=handwriting')
        for value in ['GEMINI_API_KEY', 'Gemini AI', 'TrOCR Base', 'Tesseract', 'cloud_consent']:
            self.assertContains(response, value)
        self.assertContains(response, "Google's servers")

    def test_csrf_required_for_cloud(self):
        response = Client(enforce_csrf_checks=True).post(reverse('document_studio'), {'mode':'handwriting','engine':'gemini','cloud_consent':'on','document':upload()})
        self.assertEqual(response.status_code, 403)

    @mock.patch('core.workspaces.extract_gemini_document')
    def test_invalid_upload_no_cloud(self, cloud):
        self.assertEqual(self.post(document=SimpleUploadedFile('bad.exe', b'invalid'), cloud_consent='on').status_code, 400)
        cloud.assert_not_called()
