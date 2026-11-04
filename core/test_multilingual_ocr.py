"""P6 regressions. Mocked contract tests; real accuracy evidence is in docs/p6-real-ocr-results.json."""
import io
import os
import tempfile
from pathlib import Path
from unittest import mock

from PIL import Image
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import SimpleTestCase
from django.urls import reverse
from .pdf_test_fixtures import pdf_bytes
from . import utils
from .workspaces import DocumentForm

TEXT = 'தமிழ் மொழி\nहिन्दी भाषा\nతెలుగు భాష\nമലയാളം ഭാഷ\nWelcome <script>'


def image_bytes():
    output = io.BytesIO()
    Image.new('RGB', (100, 80), 'white').save(output, format='PNG')
    return output.getvalue()


class MultilingualEngineTests(SimpleTestCase):
    def check_engine(self, languages):
        engine = mock.Mock()
        engine.image_to_osd.side_effect = RuntimeError('No orientation hint')
        engine.image_to_string.return_value = TEXT
        with mock.patch('core.utils._tesseract', return_value=engine) as setup:
            result = utils._ocr_image(Image.new('RGB', (100, 80)), languages)
        self.assertEqual(result, TEXT)
        setup.assert_called_once_with(languages)
        call = engine.image_to_string.call_args.kwargs
        self.assertEqual(call['lang'], '+'.join(languages))
        self.assertEqual(call['timeout'], utils.OCR_TIMEOUT_SECONDS)
        self.assertEqual(call['config'], '--psm 3')

    def test_english(self): self.check_engine(['eng'])
    def test_tamil(self): self.check_engine(['tam'])
    def test_hindi(self): self.check_engine(['hin'])
    def test_telugu(self): self.check_engine(['tel'])
    def test_malayalam(self): self.check_engine(['mal'])
    def test_english_tamil(self): self.check_engine(['eng', 'tam'])
    def test_english_hindi(self): self.check_engine(['eng', 'hin'])
    def test_english_telugu(self): self.check_engine(['eng', 'tel'])
    def test_english_malayalam(self): self.check_engine(['eng', 'mal'])

    def test_rejects_arbitrary_strings_and_arguments(self):
        for value in ['tam', 'eng+tam', ['eng+tam'], ['--psm 0'], ['eng', 'fra'], [None], [1], {}, [''], '']:
            with self.subTest(value=value), self.assertRaises(utils.DocumentProcessingError):
                utils.validate_ocr_languages(value)

    def test_empty_selection_defaults_to_english(self):
        for value in (None, [], ()):
            self.assertEqual(utils.validate_ocr_languages(value), ['eng'])

    def test_deduplicates_and_preserves_selection_order(self):
        self.assertEqual(utils.validate_ocr_languages(['hin', 'eng', 'hin']), ['hin', 'eng'])

    @mock.patch('pytesseract.get_tesseract_version', return_value='5.5')
    @mock.patch('pytesseract.get_languages', return_value=['eng', 'osd'])
    def test_missing_language_identifies_pack(self, languages, version):
        with self.assertRaisesRegex(utils.DocumentProcessingError, r'Tamil \(tam\).*missing'):
            utils._tesseract(['eng', 'tam'])

    @mock.patch('pytesseract.get_tesseract_version', return_value='5.5')
    @mock.patch('pytesseract.get_languages', return_value=['eng', 'tam'])
    def test_checks_actual_traineddata_even_if_language_listing_cached(self, languages, version):
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {'TESSDATA_PREFIX': directory}):
            Path(directory, 'eng.traineddata').touch()
            with self.assertRaisesRegex(utils.DocumentProcessingError, 'Tamil'):
                utils._tesseract(['tam'])

    def test_data_directory_with_spaces_uses_environment(self):
        with tempfile.TemporaryDirectory(prefix='language data ') as directory:
            with mock.patch.dict(os.environ, {'TESSDATA_PREFIX': directory}):
                self.assertEqual(utils._tessdata_directory(), Path(directory))
                self.assertEqual(utils._tessdata_config(), '')

    def test_missing_data_directory_is_actionable(self):
        with mock.patch.dict(os.environ, {'TESSDATA_PREFIX': 'C:/missing-p6-data'}):
            with self.assertRaisesRegex(utils.DocumentProcessingError, 'TESSDATA_PREFIX'):
                utils._tessdata_directory()

    def test_timeout_is_actionable(self):
        engine = mock.Mock()
        engine.image_to_string.side_effect = RuntimeError('Tesseract process timeout')
        with mock.patch('core.utils._tesseract', return_value=engine):
            with self.assertRaisesRegex(utils.DocumentProcessingError, '30-second page limit'):
                utils._ocr_image(Image.new('RGB', (100, 80)), ['tam'])

    def test_corrupt_language_data_does_not_fall_back(self):
        import pytesseract
        engine = mock.Mock()
        engine.image_to_string.side_effect = pytesseract.TesseractError(1, 'Failed loading language tam traineddata')
        with mock.patch('core.utils._tesseract', return_value=engine):
            with self.assertRaisesRegex(utils.DocumentProcessingError, 'selected language data.*tam'):
                utils._ocr_image(Image.new('RGB', (100, 80)), ['tam'])

    def test_orientation_correction_is_preserved(self):
        engine = mock.Mock()
        engine.image_to_osd.return_value = 'Rotate: 90'
        engine.image_to_string.return_value = 'தமிழ்'
        with mock.patch('core.utils._tesseract', return_value=engine):
            utils._ocr_image(Image.new('RGB', (100, 80)), ['tam'])
        self.assertEqual(engine.image_to_string.call_args.args[0].size, (80, 100))

    def test_pdf_unicode_only_pages_and_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'scan.pdf'
            path.write_bytes(pdf_bytes(['', '']))
            with mock.patch('core.utils._tesseract'), mock.patch('core.utils._ocr_image', side_effect=['தமிழ் மொழி', 'हिन्दी भाषा']) as ocr:
                result = utils.extract_document(path, ['tam', 'hin'])
            self.assertEqual(result.text, '[Page 1]\nதமிழ் மொழி\n\n[Page 2]\nहिन्दी भाषा')
            self.assertEqual(result.page_count, 2)
            self.assertEqual(result.method, 'pdf_ocr')
            self.assertTrue(all(call.args[1] == ['tam', 'hin'] for call in ocr.call_args_list))

    def test_pdf_missing_pack_cannot_be_hidden_by_digital_text(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'digital.pdf'
            path.write_bytes(pdf_bytes(['A sufficiently long selectable English document for testing.']))
            with mock.patch('core.utils._tesseract', side_effect=utils.DocumentProcessingError('Tamil (tam) language data is missing')):
                with self.assertRaisesRegex(utils.DocumentProcessingError, 'Tamil'):
                    utils.extract_document(path, ['tam'])

    def test_blank_image_has_required_message(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'blank.png'; path.write_bytes(image_bytes())
            with mock.patch('core.utils._tesseract'), mock.patch('core.utils._ocr_image', return_value=''):
                with self.assertRaisesRegex(utils.DocumentProcessingError, 'No readable text was detected'):
                    utils.extract_document(path, ['tam'])

    def test_unicode_selectable_pdf_text_is_reliable(self):
        self.assertTrue(utils._has_reliable_digital_text('தமிழ் மொழி அனைவருக்கும் வணக்கம் தமிழ்நாடு கல்வி வளர்ச்சி'))

    def test_digital_docx_does_not_require_tesseract(self):
        from docx import Document
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'notes.docx'
            doc = Document(); doc.add_paragraph('தமிழ் மொழி'); doc.save(path)
            with mock.patch('core.utils._tesseract') as engine:
                result = utils.extract_document(path, ['tam'])
            self.assertEqual(result.text, 'தமிழ் மொழி')
            engine.assert_not_called()


class MultilingualWorkspaceTests(SimpleTestCase):
    def post(self, languages=None, content=None, name='sample.png', mode='printed', accept='application/json'):
        data = {'mode': mode, 'document': SimpleUploadedFile(name, image_bytes() if content is None else content)}
        if languages is not None: data['languages'] = languages
        return self.client.post(reverse('document_studio'), data, HTTP_ACCEPT=accept)

    def test_selector_defaults_to_english_and_lists_five_languages(self):
        response = self.client.get(reverse('document_studio'))
        self.assertContains(response, 'Select the languages present in your document.')
        self.assertEqual(response.context['form']['languages'].value(), ['eng'])
        for name in utils.PRINTED_OCR_LANGUAGES.values(): self.assertContains(response, name)

    def test_multiple_languages_reach_service_and_return_unicode_metadata(self):
        with mock.patch('core.workspaces.extract_document', return_value=utils.ExtractionResult(TEXT, 'test', [], 1)) as extract:
            response = self.post(['eng', 'tam', 'tam'])
        self.assertEqual(response.status_code, 200)
        self.assertEqual(extract.call_args.kwargs['languages'], ['eng', 'tam'])
        self.assertEqual(response.json()['languages'], ['eng', 'tam'])
        self.assertEqual(response.json()['language_names'], ['English', 'Tamil'])
        self.assertEqual(response.json()['filename'], 'sample.png')
        self.assertEqual(response.json()['text'], TEXT)

    def test_empty_selection_reaches_service_as_english(self):
        with mock.patch('core.workspaces.extract_document', return_value=utils.ExtractionResult('English', 'test', [], 1)) as extract:
            response = self.post([])
        self.assertEqual(response.status_code, 200)
        self.assertEqual(extract.call_args.kwargs['languages'], ['eng'])

    def test_invalid_selection_rejected_before_processing(self):
        with mock.patch('core.workspaces.extract_document') as extract:
            response = self.post(['eng+tam --psm 0'])
        self.assertEqual(response.status_code, 400); extract.assert_not_called()

    def test_invalid_upload(self):
        self.assertEqual(self.post(['tam'], b'fake', 'bad.exe').status_code, 400)

    def test_corrupted_image(self):
        self.assertEqual(self.post(['tam'], b'\x89PNG\r\n\x1a\ncorrupt').status_code, 400)

    def test_missing_language_error_is_visible(self):
        with mock.patch('core.workspaces.extract_document', side_effect=utils.DocumentProcessingError('Tamil (tam) language data is missing')):
            response = self.post(['tam'])
        self.assertEqual(response.status_code, 400)
        self.assertIn('Tamil (tam)', str(response.json()['errors']))

    def test_unicode_html_escaping_and_edit_controls(self):
        with mock.patch('core.workspaces.extract_document', return_value=utils.ExtractionResult(TEXT, 'test', [], 1)):
            response = self.post(['tam', 'hin', 'tel', 'mal'], accept='text/html')
        for text in ['தமிழ்', 'हिन्दी', 'తెలుగు', 'മലയാളം', '&lt;script&gt;', 'data-copy="extracted-text"', 'data-download="extracted-text"', 'Analyze another document']:
            self.assertContains(response, text)
        self.assertNotContains(response, '<script>')

    def test_english_comparison_keeps_existing_processor_contract(self):
        with mock.patch('core.workspaces.extract_handwritten_document', return_value=utils.ExtractionResult('Notes', 'handwriting', [], 1)) as handwritten, mock.patch('core.workspaces.extract_document') as printed:
            response = self.post(['eng'], mode='handwriting')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(handwritten.call_args.kwargs, {})
        printed.assert_not_called()

    def test_handwriting_has_language_and_engine_selectors(self):
        response = self.client.get(reverse('document_studio') + '?mode=handwriting')
        self.assertContains(response, 'ocr-language-selector')
        self.assertContains(response, 'Gemini AI')
        self.assertContains(response, 'TrOCR Result')
