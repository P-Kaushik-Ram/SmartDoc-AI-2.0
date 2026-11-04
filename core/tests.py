import tempfile
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

from docx import Document
from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.urls import reverse
from PIL import Image, ImageDraw, ImageFont

from .pdf_test_fixtures import pdf_bytes
from .forms import CandidateForm
from .models import Candidate, JobDescription, MatchResult
from .structured_extraction import (
    apply_profile_corrections, extract_structured_profile, profile_to_form_initial,
)
from .utils import (
    DocumentProcessingError, ExtractionResult, _tesseract, _ocr_image, extract_handwritten_document,
    compute_match, extract_document, extract_skills_from_text, get_recommendation,
)
from .semantic_matching import (
    SemanticMatchingError, SemanticModelUnavailable, _get_model,
    clear_model_cache, compute_hybrid_match, get_weights, validate_candidate_for_matching,
)
from .analytics import document_method_counts, paired_match_averages, score_distribution, skill_frequencies


class MatchingServiceTests(TestCase):
    def test_skill_extraction_is_deterministic_and_case_insensitive(self):
        self.assertEqual(
            extract_skills_from_text('Python, Django and PostgreSQL. PYTHON developer.'),
            ['django', 'postgresql', 'python'],
        )

    def test_required_skill_coverage_has_no_arbitrary_bonus(self):
        score, matching, missing = compute_match(['python', 'docker', 'excel'], ['python', 'django'])
        self.assertEqual(score, 50.0)
        self.assertEqual(matching, ['python'])
        self.assertEqual(missing, ['django'])

    def test_empty_requirements_return_zero(self):
        self.assertEqual(compute_match(['python'], []), (0.0, [], []))

    def test_recommendation_thresholds(self):
        self.assertEqual(get_recommendation(70), 'highly_recommended')
        self.assertEqual(get_recommendation(40), 'recommended')
        self.assertEqual(get_recommendation(39.9), 'not_suitable')


class AnalyticsCalculationTests(TestCase):
    def test_score_distribution_uses_real_values_and_ignores_invalid_scores(self):
        result = score_distribution([0, 19.9, 20, 59, 60, 79.9, 80, 100, 101, None, 'bad'])
        self.assertEqual([item['count'] for item in result], [2, 1, 1, 2, 2])

    def test_skill_frequencies_prefer_reviewed_profile_and_exclude_unreviewed_handwriting(self):
        user = User.objects.create_user(username='analytics-user')
        Candidate.objects.create(
            name='Automatic', email='auto-analytics@example.com', mobile='1', experience='1 year',
            resume='resumes/auto.pdf', extraction_status='completed', extracted_skills='Python, Django',
        )
        Candidate.objects.create(
            name='Reviewed', email='reviewed-analytics@example.com', mobile='2', experience='2 years',
            resume='resumes/reviewed.pdf', extraction_status='completed', extracted_skills='python, sql',
            reviewed_text='Corrected skills list', reviewed_profile={'skills': [
                {'value': 'Python'}, {'value': 'Excel'}, {'value': 'python'},
            ]}, uploaded_by=user,
        )
        Candidate.objects.create(
            name='Handwriting', email='handwriting-analytics@example.com', mobile='3', experience='1 year',
            resume='resumes/handwriting.png', processing_mode=Candidate.PROCESSING_HANDWRITING,
            extraction_status='needs_review', extracted_skills='python', extracted_text='Python? maybe',
        )
        Candidate.objects.create(
            name='Needs review', email='review-needed-analytics@example.com', mobile='4', experience='1 year',
            resume='resumes/needs-review.pdf', extraction_status='needs_review', extracted_skills='Java',
        )
        result = skill_frequencies(Candidate.objects.all())
        counts = {item['label']: item['count'] for item in result['skills']}
        self.assertEqual(counts, {'Python': 2, 'Django': 1, 'Excel': 1})
        self.assertEqual(result['reviewed_candidates'], 1)
        self.assertEqual(result['automatic_candidates'], 1)
        self.assertEqual(result['excluded_unreviewed_handwriting'], 1)
        self.assertEqual(result['eligible_candidates'], 2)

    def test_document_methods_and_paired_method_averages(self):
        methods = document_method_counts([('pdf_text', 2), ('pdf_text', 1), ('', 1)])
        self.assertEqual(methods, [
            {'label': 'Digital PDF text', 'count': 3},
            {'label': 'Method not recorded', 'count': 1},
        ])
        comparison = paired_match_averages([(40, 60), (80, 70), (None, 20), (101, 50)])
        self.assertEqual(comparison, {'count': 2, 'baseline_average': 60.0, 'hybrid_average': 65.0})


class SemanticMatchingTests(TestCase):
    @staticmethod
    def candidate(**overrides):
        values = {
            'extraction_status': 'completed', 'processing_mode': 'standard',
            'reviewed_text': '', 'reviewed_profile': {}, 'extraction_warnings': '',
            'extracted_skills': 'python', 'extracted_text': '[Page 2]\nBuilt Python APIs and deployed services.',
        }
        values.update(overrides)
        obj = types.SimpleNamespace(**values)
        obj.get_effective_text = lambda: obj.reviewed_text.strip() or obj.extracted_text
        obj.get_skills_list = lambda: [item.strip().lower() for item in obj.extracted_skills.split(',') if item.strip()]
        return obj

    @staticmethod
    def job(**overrides):
        values = {
            'required_skills': 'python, django',
            'description': 'Build and maintain scalable backend services.',
            'get_skills_list': lambda: ['python', 'django'],
        }
        values.update(overrides)
        return types.SimpleNamespace(**values)

    def test_model_is_loaded_once_locally_on_cpu(self):
        class FakeSentenceTransformer:
            def __init__(self, path, **kwargs):
                self.path, self.kwargs = path, kwargs

        fake_module = types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer)
        clear_model_cache()
        with (
            mock.patch.dict(sys.modules, {'sentence_transformers': fake_module}),
            mock.patch.dict('os.environ', {
                'SMARTDOC_SEMANTIC_DEVICE': 'cpu',
                'SMARTDOC_SEMANTIC_MODEL_PATH': r'C:\local-models\all-MiniLM-L6-v2',
            }),
        ):
            first = _get_model()
            second = _get_model()
        self.assertIs(first, second)
        self.assertEqual(first.path, r'C:\local-models\all-MiniLM-L6-v2')
        self.assertEqual(first.kwargs['device'], 'cpu')
        self.assertIs(first.kwargs['local_files_only'], True)
        clear_model_cache()

    def test_invalid_configured_model_path_keeps_actionable_warning(self):
        class MissingLocalModel:
            def __init__(self, path, **kwargs):
                raise OSError('model files are missing')

        clear_model_cache()
        fake_module = types.SimpleNamespace(SentenceTransformer=MissingLocalModel)
        with (
            mock.patch.dict(sys.modules, {'sentence_transformers': fake_module}),
            mock.patch.dict('os.environ', {
                'SMARTDOC_SEMANTIC_MODEL_PATH': r'C:\missing\all-MiniLM-L6-v2',
                'SMARTDOC_SEMANTIC_DEVICE': 'cpu',
            }),
        ):
            with self.assertRaisesRegex(SemanticModelUnavailable, 'not available locally'):
                _get_model()
        clear_model_cache()

    def test_missing_optional_package_has_clear_error(self):
        clear_model_cache()
        with mock.patch.dict(sys.modules, {'sentence_transformers': None}):
            with self.assertRaisesRegex(SemanticModelUnavailable, 'optional requirements-semantic.txt'):
                _get_model()

    def test_mocked_hybrid_score_persists_comparable_components_and_real_passage(self):
        class FixedVectorModel:
            def encode(self, texts, **kwargs):
                return [[1.0, 0.0] for _ in texts]

        candidate = self.candidate()
        with mock.patch('core.semantic_matching._get_model', return_value=FixedVectorModel()):
            result = compute_hybrid_match(candidate, self.job())
        self.assertEqual(result['baseline_score'], 50.0)
        self.assertEqual(result['semantic_score'], 100.0)
        self.assertEqual(result['score'], 70.0)
        self.assertEqual(result['weights'], {'skill_coverage': 0.6, 'semantic_relevance': 0.4})
        self.assertEqual(result['evidence'][0]['passage'], 'Built Python APIs and deployed services.')
        self.assertEqual(result['evidence'][0]['page'], 2)
        self.assertIn('not confidence', result['interpretation'])

    def test_corrected_text_is_the_reported_source(self):
        class FixedVectorModel:
            def encode(self, texts, **kwargs):
                return [[1.0, 0.0] for _ in texts]

        candidate = self.candidate(reviewed_text='Reviewed correction: Django API testing.', extracted_text='Original text')
        with mock.patch('core.semantic_matching._get_model', return_value=FixedVectorModel()):
            result = compute_hybrid_match(candidate, self.job())
        self.assertEqual(result['text_source'], 'reviewed_text_and_corrected_profile')
        self.assertIn('Reviewed correction', result['evidence'][0]['passage'])

    def test_failed_or_incomplete_extraction_is_not_a_negative_match(self):
        candidate = self.candidate(extraction_status='failed')
        with self.assertRaisesRegex(SemanticMatchingError, 'not evidence of missing qualifications'):
            validate_candidate_for_matching(candidate)

    def test_unreviewed_handwriting_is_rejected_before_model_use(self):
        candidate = self.candidate(processing_mode='handwriting', reviewed_text='')
        with self.assertRaisesRegex(SemanticMatchingError, 'manually reviewed'):
            compute_hybrid_match(candidate, self.job())

    def test_configurable_weights_are_normalized(self):
        with mock.patch.dict('os.environ', {
            'SMARTDOC_MATCH_SKILL_WEIGHT': '3', 'SMARTDOC_MATCH_SEMANTIC_WEIGHT': '1',
        }):
            self.assertEqual(get_weights(), {'skill_coverage': 0.75, 'semantic_relevance': 0.25})
        with mock.patch.dict('os.environ', {
            'SMARTDOC_MATCH_SKILL_WEIGHT': '0', 'SMARTDOC_MATCH_SEMANTIC_WEIGHT': '0',
        }), self.assertRaises(SemanticMatchingError):
            get_weights()


class CandidateUploadFormTests(TestCase):
    def _form(self, filename, content):
        upload = SimpleUploadedFile(filename, content, content_type='application/octet-stream')
        return CandidateForm(data={
            'name': 'Test Candidate', 'email': 'test@example.com',
            'mobile': '1234567890', 'experience': '1 year',
        }, files={'resume': upload})

    def test_rejects_extension_that_does_not_match_content(self):
        form = self._form('resume.pdf', b'not a pdf')
        self.assertFalse(form.is_valid())
        self.assertIn('resume', form.errors)

    def test_rejects_unsupported_extension(self):
        form = self._form('resume.exe', b'MZ fake executable')
        self.assertFalse(form.is_valid())
        self.assertIn('resume', form.errors)

    def test_rejects_empty_files(self):
        form = self._form('empty.pdf', b'')
        self.assertFalse(form.is_valid())
        self.assertIn('resume', form.errors)

    def test_rejects_files_over_the_upload_limit(self):
        form = self._form('large.pdf', b'%PDF-' + b'x' * CandidateForm.MAX_UPLOAD_BYTES)
        self.assertFalse(form.is_valid())
        self.assertIn('resume', form.errors)


class StructuredResumeExtractionTests(TestCase):
    def test_extracts_contact_skills_education_projects_work_certifications_and_dates(self):
        text = '''[Page 1]
Taylor Morgan
taylor.morgan@example.com
+1 (415) 555-0182
EDUCATION
BSc Computer Science, North State University, 2018 - 2022
SKILLS
Python, Django, PostgreSQL, Docker
PROJECTS
Hiring analytics dashboard — built a Django application.
WORK EXPERIENCE
Software Engineer at Example Corp, Jan 2023 - Present (3 years)
INTERNSHIPS
Engineering Intern, Summer 2021
CERTIFICATIONS
AWS Certified Developer, 2024
ACHIEVEMENTS
Dean's List, 2020
'''
        profile = extract_structured_profile(text)
        self.assertEqual(profile['full_name']['value'], 'Taylor Morgan')
        self.assertEqual(profile['full_name']['status'], 'extracted')
        self.assertEqual(profile['email']['value'], 'taylor.morgan@example.com')
        self.assertIn('555-0182', profile['phone']['value'])
        self.assertEqual([item['value'] for item in profile['skills']], ['python', 'django', 'postgresql', 'docker'])
        for field in ('education', 'projects', 'work_experience', 'internships', 'certifications', 'achievements'):
            self.assertTrue(profile[field], field)
        self.assertTrue(any(item['page'] == 1 for item in profile['education'][0]['sources']))
        self.assertIn('2023', profile['work_experience'][0]['dates'][0])
        self.assertTrue(any('years' in item['value'].lower() for item in profile['dates']))

    def test_missing_fields_are_reported_as_not_found_not_negative_qualifications(self):
        profile = extract_structured_profile('A short description with no resume section headings.')
        self.assertEqual(profile['email']['status'], 'missing')
        self.assertEqual(profile['phone']['status'], 'missing')
        self.assertEqual(profile['education'], [])
        self.assertEqual(profile['field_status']['education'], 'missing')
        self.assertEqual(profile['email']['review_status'], 'not_found')

    def test_multiple_contacts_and_possible_names_are_marked_ambiguous(self):
        profile = extract_structured_profile(
            'Taylor Morgan\ntaylor@example.com\nTaylor.Morgan@example.org\n'
            '+1 415 555 0101\n+1 415 555 0102\nAlex Morgan\n'
        )
        self.assertEqual(profile['email']['status'], 'ambiguous')
        self.assertEqual(profile['phone']['status'], 'ambiguous')
        self.assertEqual(profile['full_name']['status'], 'ambiguous')

    def test_name_detection_handles_labeled_and_contact_row_headers_with_ocr_noise(self):
        labeled = extract_structured_profile(
            '[Page 1]\n❖  Full Name: Priya R. Shah  ✦\n'
            '✉ priya@example.test | ☎ +1 415 555 0123\n'
            'EDUCATION\n• BSc Computer Science, Fictional University, 2020 - 2024\n'
            'SKILLS\n▪ Python • Django • PostgreSQL\n'
            'PROJECTS\n✦ Built an accessibility dashboard.\n'
            'CERTIFICATIONS\n◆ Cloud Fundamentals, 2023\n'
        )
        self.assertEqual(labeled['full_name']['value'], 'Priya R. Shah')
        self.assertEqual(labeled['full_name']['status'], 'extracted')
        self.assertEqual(labeled['email']['value'], 'priya@example.test')
        self.assertEqual(labeled['phone']['status'], 'extracted')
        self.assertTrue(labeled['education'])
        self.assertTrue(labeled['skills'])
        self.assertTrue(labeled['projects'])
        self.assertTrue(labeled['certifications'])

        inline = extract_structured_profile(
            '[Page 1]\n✦  Morgan Lee | morgan.lee@example.test | +1 415 555 0134 ✦\n'
            'EXPERIENCE\n• QA Analyst at Fictional Labs, 2022 - Present\n'
        )
        self.assertEqual(inline['full_name']['value'], 'Morgan Lee')
        self.assertEqual(inline['full_name']['status'], 'extracted')
        self.assertEqual(inline['email']['value'], 'morgan.lee@example.test')
        self.assertEqual(inline['work_experience'][0]['value'], 'QA Analyst at Fictional Labs, 2022 - Present')

    def test_repeated_ocr_header_name_is_not_marked_ambiguous(self):
        profile = extract_structured_profile(
            'Avery Chen\nAvery Chen\navery@example.test\nSKILLS\nPython'
        )
        self.assertEqual(profile['full_name']['value'], 'Avery Chen')
        self.assertEqual(profile['full_name']['status'], 'extracted')

    def test_malformed_and_oversized_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            extract_structured_profile(None)
        with self.assertRaises(ValueError):
            extract_structured_profile('x' * 200001)

    def test_manual_corrections_are_separate_and_retain_unchanged_evidence(self):
        original = extract_structured_profile('[Page 2]\nTaylor Morgan\ntaylor@example.com\nSKILLS\nPython, Django')
        values = profile_to_form_initial(original)
        values.update({'full_name': 'Taylor A. Morgan', 'skills': 'Python\nDjango\nSQL'})
        reviewed = apply_profile_corrections(values, original)
        self.assertEqual(original['full_name']['value'], 'Taylor Morgan')
        self.assertEqual(reviewed['full_name']['origin'], 'manual_correction')
        self.assertEqual(reviewed['email']['origin'], 'automatic_value_reviewed')
        self.assertEqual(reviewed['email']['sources'][0]['page'], 2)
        self.assertEqual([item['value'] for item in reviewed['skills']], ['Python', 'Django', 'SQL'])
        self.assertEqual(reviewed['skills'][-1]['origin'], 'manual_correction')

class DocumentExtractionTests(TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _make_pdf(self, name, pages):
        path = self.root / name
        path.write_bytes(pdf_bytes(pages))
        return path

    def _make_printed_text_image(self, name='printed.png'):
        path = self.root / name
        image = Image.new('RGB', (800, 220), 'white')
        ImageDraw.Draw(image).text((30, 70), 'Printed resume text: Python and Django', fill='black')
        image.save(path)
        return path

    def test_digitally_generated_pdf_does_not_run_ocr(self):
        path = self._make_pdf('digital.pdf', ['Experienced Python and Django engineer with ten years building reliable software.'])
        with mock.patch('core.utils._ocr_image') as ocr:
            result = extract_document(path)
        self.assertEqual(result.method, 'pdf_text')
        self.assertEqual(result.page_count, 1)
        self.assertIn('Python', result.text)
        ocr.assert_not_called()

    def test_scanned_pdf_uses_ocr_and_keeps_page_boundary(self):
        path = self._make_pdf('scanned.pdf', [''])
        with mock.patch('core.utils._ocr_image', return_value='Printed Python and Django resume text.') as ocr:
            result = extract_document(path)
        self.assertEqual(result.method, 'pdf_ocr')
        self.assertEqual(result.page_count, 1)
        self.assertIn('[Page 1]', result.text)
        self.assertIn('Printed Python', result.text)
        ocr.assert_called_once()

    def test_mixed_pdf_extracts_digital_page_and_ocr_page(self):
        path = self._make_pdf('mixed.pdf', [
            'Digital page with sufficient readable text for direct extraction and no OCR.', '',
        ])
        with mock.patch('core.utils._ocr_image', return_value='Scanned page says Java and SQL.') as ocr:
            result = extract_document(path)
        self.assertEqual(result.method, 'pdf_text_ocr')
        self.assertEqual(result.page_count, 2)
        self.assertIn('[Page 1]', result.text)
        self.assertIn('[Page 2]', result.text)
        self.assertIn('Digital page', result.text)
        self.assertIn('Scanned page', result.text)
        ocr.assert_called_once()

    def test_docx_extracts_paragraphs_and_tables(self):
        path = self.root / 'candidate.docx'
        document = Document()
        document.add_paragraph('Resume paragraph: Python and Django.')
        table = document.add_table(rows=1, cols=2)
        table.cell(0, 0).text = 'Skill'
        table.cell(0, 1).text = 'PostgreSQL'
        document.save(path)
        result = extract_document(path)
        self.assertEqual(result.method, 'docx_text')
        self.assertIsNone(result.page_count)
        self.assertIn('Resume paragraph', result.text)
        self.assertIn('Skill | PostgreSQL', result.text)

    def test_printed_text_image_uses_ocr(self):
        path = self._make_printed_text_image()
        with mock.patch('core.utils._ocr_image', return_value='Printed resume text: Python and Django') as ocr:
            result = extract_document(path)
        self.assertEqual(result.method, 'tesseract_ocr')
        self.assertEqual(result.page_count, 1)
        self.assertIn('Python', result.text)
        ocr.assert_called_once()

    def test_rejects_password_protected_and_owner_encrypted_pdfs(self):
        from reportlab.lib.pdfencrypt import StandardEncryption

        path = self.root / 'encrypted.pdf'
        for user_password in ('fixture-passphrase', ''):
            with self.subTest(has_user_password=bool(user_password)):
                encryption = StandardEncryption(user_password, ownerPassword='fixture-owner')
                path.write_bytes(pdf_bytes(['Synthetic protected document.'], encrypt=encryption))
                with self.assertRaisesRegex(DocumentProcessingError, 'Password-protected'):
                    extract_document(path)

    def test_rejects_unsupported_corrupt_and_empty_files(self):
        unsupported = self.root / 'notes.txt'
        unsupported.write_text('not a resume', encoding='utf-8')
        corrupt_pdf = self.root / 'corrupt.pdf'
        corrupt_pdf.write_bytes(b'%PDF-not a valid PDF')
        corrupt_docx = self.root / 'corrupt.docx'
        corrupt_docx.write_bytes(b'PK\x03\x04not a zip archive')
        corrupt_png = self.root / 'corrupt.png'
        corrupt_png.write_bytes(b'\x89PNG\r\n\x1a\nnot a png')
        empty = self.root / 'empty.pdf'
        empty.write_bytes(b'')
        for path in [unsupported, corrupt_pdf, corrupt_docx, corrupt_png, empty]:
            with self.subTest(path=path.name), self.assertRaises(DocumentProcessingError):
                extract_document(path)


class HandwritingServiceTests(TestCase):
    def setUp(self):
        from . import handwriting
        handwriting.reset_model_cache()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        from . import handwriting
        handwriting.reset_model_cache()
        self.temp_dir.cleanup()

    def _make_pdf(self, name, pages):
        path = self.root / name
        path.write_bytes(pdf_bytes(pages))
        return path

    def _make_printed_text_image(self):
        path = self.root / 'printed.png'
        image = Image.new('RGB', (800, 220), 'white')
        ImageDraw.Draw(image).text((30, 70), 'Printed resume text: Python and Django', fill='black')
        image.save(path)
        return path

    def _two_line_image(self):
        image = Image.new('RGB', (600, 180), 'white')
        draw = ImageDraw.Draw(image)
        draw.text((20, 25), 'Handwritten first line', fill='black')
        draw.text((20, 105), 'Handwritten second line', fill='black')
        return image

    def test_preprocessing_splits_lines_and_normalizes_for_model(self):
        from .handwriting import preprocess_handwriting_image
        lines = preprocess_handwriting_image(self._two_line_image())
        self.assertEqual(len(lines), 2)
        self.assertTrue(all(line.mode == 'RGB' for line in lines))
        self.assertTrue(all(line.width > line.height for line in lines))

    def test_preprocessing_splits_ruled_table_rows_and_cells_without_whole_table_crop(self):
        from .handwriting import preprocess_handwriting_image
        # Draw generic labels and cells; this tests geometry, not a target
        # transcription or any candidate-specific words/values.
        canvas = Image.new('RGB', (300, 230), 'white')
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default()
        draw.text((20, 8), 'Q) ABC', font=font, fill='black')
        draw.text((18, 56), 'Entropy', font=font, fill='black')
        draw.text((90, 89), 'Symbol', font=font, fill='black')
        draw.text((190, 89), 'Freq', font=font, fill='black')
        for y, left, right in ((128, 'A', '6'), (165, 'B', '2'), (200, 'C', '2')):
            draw.text((100, y), left, font=font, fill='black')
            draw.text((205, y), right, font=font, fill='black')
        draw.line((160, 83, 160, 218), fill='black', width=2)
        draw.line((72, 108, 275, 108), fill='black', width=2)
        draw.line((160, 218, 275, 218), fill='black', width=2)
        draw.line((20, 220, 260, 212), fill='black', width=2)

        page = canvas.resize((900, 690), Image.Resampling.LANCZOS)
        line_images = preprocess_handwriting_image(page)
        self.assertEqual(len(line_images), 10)
        self.assertTrue(all(line.width < page.width and line.height < page.height for line in line_images))

    def test_missing_model_dependencies_are_reported(self):
        import builtins
        from . import handwriting
        original_import = builtins.__import__

        def no_transformers(name, *args, **kwargs):
            if name == 'transformers' or name.startswith('transformers.'):
                raise ImportError('not installed')
            return original_import(name, *args, **kwargs)

        with mock.patch('builtins.__import__', side_effect=no_transformers):
            with self.assertRaisesRegex(DocumentProcessingError, 'optional PyTorch and Transformers'):
                handwriting._load_backend()

    def test_model_initializes_once_and_uses_local_cached_weights(self):
        from . import handwriting
        torch = types.ModuleType('torch')
        torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        processor_class = mock.Mock()
        processor = mock.Mock()
        processor_class.from_pretrained.return_value = processor
        model_class = mock.Mock()
        model = mock.Mock()
        model_class.from_pretrained.return_value = model
        transformers = types.ModuleType('transformers')
        transformers.TrOCRProcessor = processor_class
        transformers.VisionEncoderDecoderModel = model_class
        with mock.patch.dict(sys.modules, {'torch': torch, 'transformers': transformers, 'sentencepiece': types.ModuleType('sentencepiece')}), \
             mock.patch.dict('os.environ', {'SMARTDOC_HANDWRITING_DEVICE': 'cpu', 'SMARTDOC_HANDWRITING_MODEL_PATH': ''}):
            first = handwriting._load_backend()
            second = handwriting._load_backend()
        self.assertIs(first, second)
        self.assertEqual(first[3], 'cpu')
        processor_class.from_pretrained.assert_called_once_with(handwriting.MODEL_ID, local_files_only=True)
        model_class.from_pretrained.assert_called_once_with(handwriting.MODEL_ID, local_files_only=True)
        model.to.assert_called_once_with('cpu')
        model.eval.assert_called_once()

    def test_unavailable_local_model_reports_explicit_download_command(self):
        from . import handwriting
        torch = types.ModuleType('torch')
        torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        processor_class = mock.Mock()
        processor_class.from_pretrained.side_effect = OSError('not found in the cached files')
        transformers = types.ModuleType('transformers')
        transformers.TrOCRProcessor = processor_class
        transformers.VisionEncoderDecoderModel = mock.Mock()
        with mock.patch.dict(sys.modules, {'torch': torch, 'transformers': transformers, 'sentencepiece': types.ModuleType('sentencepiece')}), \
             mock.patch.dict('os.environ', {'SMARTDOC_HANDWRITING_DEVICE': 'cpu'}):
            with self.assertRaisesRegex(DocumentProcessingError, f'hf download {handwriting.MODEL_ID}'):
                handwriting._load_backend()

    def test_explicit_local_model_directory_is_used_without_network_loading(self):
        from . import handwriting
        torch = types.ModuleType('torch')
        torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        processor_class = mock.Mock()
        processor_class.from_pretrained.return_value = mock.Mock()
        model_class = mock.Mock()
        model = mock.Mock()
        model_class.from_pretrained.return_value = model
        transformers = types.ModuleType('transformers')
        transformers.TrOCRProcessor = processor_class
        transformers.VisionEncoderDecoderModel = model_class
        model_dir = self.root / 'trocr-small-handwritten'
        model_dir.mkdir()
        expected_path = str(model_dir.resolve())
        with mock.patch.dict(sys.modules, {'torch': torch, 'transformers': transformers, 'sentencepiece': types.ModuleType('sentencepiece')}), \
             mock.patch.dict('os.environ', {
                 'SMARTDOC_HANDWRITING_DEVICE': 'cpu',
                 'SMARTDOC_HANDWRITING_MODEL_PATH': str(model_dir),
             }):
            backend = handwriting._load_backend()
        self.assertEqual(backend[3], 'cpu')
        processor_class.from_pretrained.assert_called_once_with(expected_path, local_files_only=True)
        model_class.from_pretrained.assert_called_once_with(expected_path, local_files_only=True)

    def test_missing_explicit_model_directory_reports_setup_command(self):
        from . import handwriting
        with mock.patch.dict('os.environ', {
            'SMARTDOC_HANDWRITING_MODEL_PATH': str(self.root / 'missing-checkpoint'),
        }):
            with self.assertRaisesRegex(DocumentProcessingError, 'configured TrOCR model directory does not exist') as error:
                handwriting._load_backend()
        self.assertIn(f'hf download {handwriting.MODEL_ID}', str(error.exception))

    def test_incomplete_local_checkpoint_reports_missing_files_helpfully(self):
        from . import handwriting
        torch = types.ModuleType('torch')
        torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        processor_class = mock.Mock()
        processor_class.from_pretrained.side_effect = OSError(
            'The directory does not appear to have a file named preprocessor_config.json.'
        )
        transformers = types.ModuleType('transformers')
        transformers.TrOCRProcessor = processor_class
        transformers.VisionEncoderDecoderModel = mock.Mock()
        model_dir = self.root / 'incomplete-model'
        model_dir.mkdir()
        with mock.patch.dict(sys.modules, {
            'torch': torch, 'transformers': transformers,
            'sentencepiece': types.ModuleType('sentencepiece'),
        }), mock.patch.dict('os.environ', {
            'SMARTDOC_HANDWRITING_MODEL_PATH': str(model_dir),
            'SMARTDOC_HANDWRITING_DEVICE': 'cpu',
        }):
            with self.assertRaisesRegex(DocumentProcessingError, 'checkpoint is incomplete or unavailable locally') as error:
                handwriting._load_backend()
        self.assertIn('No model download was attempted', str(error.exception))

    def test_missing_sentencepiece_dependency_is_actionable(self):
        from . import handwriting

        torch = types.ModuleType('torch')
        transformers = types.ModuleType('transformers')
        transformers.TrOCRProcessor = mock.Mock()
        transformers.VisionEncoderDecoderModel = mock.Mock()

        with mock.patch.dict(sys.modules, {
            'torch': torch, 'transformers': transformers, 'sentencepiece': None,
        }), mock.patch.dict('os.environ', {'SMARTDOC_HANDWRITING_MODEL_PATH': ''}):
            with self.assertRaisesRegex(DocumentProcessingError, 'SentencePiece is missing'):
                handwriting._load_backend()

    def test_recognition_uses_mocked_inference_for_each_detected_line(self):
        from . import handwriting
        torch = mock.Mock()
        torch.inference_mode.return_value = mock.MagicMock()
        pixel_values = mock.Mock()
        pixel_values.to.return_value = pixel_values
        processor = mock.Mock()
        processor.return_value.pixel_values = pixel_values
        processor.batch_decode.side_effect = [['recognized first'], ['recognized second']]
        model = mock.Mock()
        model.generate.side_effect = ['ids-1', 'ids-2']
        with mock.patch('core.handwriting._load_backend', return_value=(torch, processor, model, 'cpu')):
            recognized = handwriting.recognize_handwriting_image(self._two_line_image())
        self.assertEqual(recognized, 'recognized first\nrecognized second')
        self.assertEqual(model.generate.call_count, 2)

    def test_document_processor_routes_image_and_pdf_pages_to_handwriting_service(self):
        image_path = self.root / 'handwritten.png'
        self._two_line_image().save(image_path)
        with mock.patch('core.handwriting.recognize_handwriting_image', return_value='Recognized words') as recognize:
            image_result = extract_handwritten_document(image_path)
        self.assertEqual(image_result.method, 'handwriting_trocr')
        self.assertEqual(image_result.page_count, 1)
        self.assertIn('[Page 1]', image_result.text)
        recognize.assert_called_once()

        pdf_path = self._make_pdf('handwritten.pdf', [''])
        with mock.patch('core.handwriting.recognize_handwriting_image', return_value='Handwritten PDF line') as recognize:
            pdf_result = extract_handwritten_document(pdf_path)
        self.assertEqual(pdf_result.page_count, 1)
        self.assertIn('Handwritten PDF line', pdf_result.text)
        recognize.assert_called_once()

    def test_docx_is_rejected_in_handwriting_mode(self):
        path = self.root / 'not-supported.docx'
        path.write_bytes(b'PK\x03\x04not-a-docx')
        with self.assertRaisesRegex(DocumentProcessingError, 'DOCX is not supported'):
            extract_handwritten_document(path)

    def test_inference_errors_are_reported(self):
        from . import handwriting
        torch = mock.Mock()
        processor = mock.Mock()
        processor.return_value.pixel_values.to.return_value = mock.Mock()
        model = mock.Mock()
        model.generate.side_effect = RuntimeError('out of memory')
        with mock.patch('core.handwriting._load_backend', return_value=(torch, processor, model, 'cpu')):
            with self.assertRaisesRegex(DocumentProcessingError, 'failed during inference'):
                handwriting.recognize_handwriting_image(self._two_line_image())

    def test_missing_tesseract_is_reported_actionably(self):
        with mock.patch.dict(os.environ, {'TESSERACT_CMD': ''}), \
             mock.patch('core.utils.shutil.which', return_value=None), \
             mock.patch('core.utils.Path.is_file', return_value=False):
            with self.assertRaisesRegex(DocumentProcessingError, 'not installed or is not on PATH'):
                _tesseract()

    def test_tesseract_cmd_environment_override_is_configured(self):
        import pytesseract
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / 'tesseract.exe'
            executable.touch()
            original_command = pytesseract.pytesseract.tesseract_cmd
            try:
                with mock.patch.dict(os.environ, {'TESSERACT_CMD': str(executable)}), \
                     mock.patch('pytesseract.get_tesseract_version', return_value='5.0'), \
                     mock.patch('pytesseract.get_languages', return_value=['eng']):
                    self.assertIs(_tesseract(), pytesseract)
                    self.assertEqual(pytesseract.pytesseract.tesseract_cmd, str(executable))
            finally:
                pytesseract.pytesseract.tesseract_cmd = original_command

    def test_verified_windows_path_is_used_when_not_on_path(self):
        import pytesseract
        configured_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        original_command = pytesseract.pytesseract.tesseract_cmd
        try:
            with (
                mock.patch.dict(os.environ, {'TESSERACT_CMD': ''}),
                mock.patch('core.utils.shutil.which', return_value=None),
                mock.patch('core.utils.os.name', 'nt'),
                mock.patch('core.utils.Path.is_file', return_value=True),
                mock.patch('pytesseract.get_tesseract_version', return_value='5.0'),
                mock.patch('pytesseract.get_languages', return_value=['eng']),
            ):
                _tesseract()
                self.assertEqual(pytesseract.pytesseract.tesseract_cmd, configured_path)
        finally:
            pytesseract.pytesseract.tesseract_cmd = original_command

    def test_missing_tesseract_cmd_executable_fails_before_invocation(self):
        import pytesseract
        with mock.patch.dict(os.environ, {'TESSERACT_CMD': r'C:\missing\tesseract.exe'}), \
             mock.patch('pytesseract.get_tesseract_version') as version:
            with self.assertRaisesRegex(DocumentProcessingError, 'configured by TESSERACT_CMD was not found'):
                _tesseract()
        version.assert_not_called()

    def test_missing_english_language_data_is_reported(self):
        with mock.patch('pytesseract.get_tesseract_version', return_value='5.0'), \
             mock.patch('pytesseract.get_languages', return_value=['osd']):
            with self.assertRaisesRegex(DocumentProcessingError, r'English \(eng\) language data is missing'):
                _tesseract()

    def test_ocr_execution_failure_is_not_silently_returned_as_empty_text(self):
        path = self._make_printed_text_image()
        fake_tesseract = mock.Mock()
        fake_tesseract.image_to_osd.side_effect = RuntimeError('orientation unavailable')
        fake_tesseract.image_to_string.side_effect = RuntimeError('OCR engine error')
        with mock.patch('core.utils._tesseract', return_value=fake_tesseract):
            with Image.open(path) as image, self.assertRaisesRegex(DocumentProcessingError, 'OCR could not process'):
                _ocr_image(image)

    def test_ocr_uses_automatic_page_segmentation_for_distinct_resume_headers(self):
        path = self._make_printed_text_image()
        fake_tesseract = mock.Mock()
        fake_tesseract.image_to_osd.side_effect = RuntimeError('orientation unavailable')
        fake_tesseract.image_to_string.return_value = 'Fictional Name\nPython resume text'
        with mock.patch('core.utils._tesseract', return_value=fake_tesseract):
            with Image.open(path) as image:
                text = _ocr_image(image)
        self.assertIn('Fictional Name', text)
        self.assertEqual(fake_tesseract.image_to_string.call_args.kwargs['config'], '--psm 3')

    def test_scanned_pdf_without_tesseract_fails_with_clear_error(self):
        path = self._make_pdf('scanned.pdf', [''])
        with mock.patch('core.utils._ocr_image', side_effect=DocumentProcessingError('Tesseract is not installed')):
            with self.assertRaisesRegex(DocumentProcessingError, 'Tesseract is not installed'):
                extract_document(path)


class DocumentAssistantTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='assistant-user', password='Strong-test-password-123')
        self.client.force_login(self.user)
        self.candidate = Candidate.objects.create(
            name='Alex Candidate', email='assistant@example.com', mobile='123', experience='3 years',
            resume='resumes/alex.pdf', extraction_status='completed',
            extracted_text='[Page 1]\nPython engineer with three years building Django APIs.\n'
                           '[Page 2]\nBuilt a reporting project using SQL.',
        )
        self.job = JobDescription.objects.create(
            title='Backend engineer', required_skills='Python, SQL', experience='3 years',
            description='Build backend services.', created_by=self.user,
        )

    def test_assistant_requires_authentication_and_renders(self):
        self.client.logout()
        self.assertEqual(self.client.get(reverse('ai_document_assistant')).status_code, 302)
        self.client.force_login(self.user)
        response = self.client.get(reverse('ai_document_assistant'), {'candidate': self.candidate.pk})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'AI Document Assistant')

    @mock.patch('core.views.get_answer_provider')
    def test_answer_uses_retrieved_passages_and_validated_page_sources(self, provider_factory):
        provider_factory.return_value.answer.return_value = 'Python is explicit in the resume [S1]. [S99]'
        response = self.client.post(reverse('ai_document_assistant'), {
            'candidate': self.candidate.pk, 'question': 'What Python experience is stated?',
        })
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Python is explicit')
        self.assertNotContains(response, '[S99]')
        self.assertContains(response, 'Page 1')
        sent_messages = provider_factory.return_value.answer.call_args.args[0]
        self.assertIn('<untrusted_document_text>', sent_messages[-1]['content'])
        self.assertIn('ignore', sent_messages[0]['content'].lower())
        self.assertEqual({source['id'] for source in response.context['sources']}, {'S1'})

    @mock.patch('core.views.get_answer_provider')
    def test_skills_answer_deduplicates_repeated_items(self, provider_factory):
        self.candidate.extracted_text = '[Page 1]\nSkills: Python, SQL.'
        self.candidate.save(update_fields=['extracted_text'])
        provider_factory.return_value.answer.return_value = 'Python, SQL, Python [S1].\nPython [S1].'
        response = self.client.post(reverse('ai_document_assistant'), {
            'candidate': self.candidate.pk, 'question': 'What skills does Alex have?',
        })
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Python, SQL [S1]')
        self.assertNotContains(response, 'Python, SQL, Python')

    @mock.patch('core.views.get_answer_provider')
    def test_skill_question_retrieves_values_under_skills_heading_not_heading_alone(self, provider_factory):
        from .document_assistant import retrieve_evidence
        self.candidate.extracted_text = (
            '[Page 1]\nEDUCATION\nComputer Science degree\n'
            'SKILLS\nProgramming Languages: Python, Java\n'
            'Web Technologies: React, Node.js\nDATA SCIENCE & ML\nTensorFlow, PyTorch\n'
            'PROJECTS\nBuilt a resume application.'
        )
        self.candidate.save(update_fields=['extracted_text'])
        sources = retrieve_evidence(self.candidate, 'What skills are listed in the resume?')
        source_texts = [source['text'] for source in sources]
        self.assertEqual(source_texts, [
            'Programming Languages: Python, Java',
            'Web Technologies: React, Node.js',
            'TensorFlow, PyTorch',
        ])
        provider_factory.return_value.answer.return_value = 'Python, React, TensorFlow [S1]'
        response = self.client.post(reverse('ai_document_assistant'), {
            'candidate': self.candidate.pk, 'question': 'What skills are listed in the resume?',
        })
        self.assertContains(response, 'Python, React, TensorFlow [S1] [S2] [S3]')
        self.assertEqual({source['id'] for source in response.context['sources']}, {'S1', 'S2', 'S3'})

    @mock.patch('core.views.get_answer_provider')
    def test_supported_uncited_answer_gets_only_a_retrieved_source_citation(self, provider_factory):
        self.candidate.extracted_text = '[Page 1]\nSkills: Python and Django.'
        self.candidate.save(update_fields=['extracted_text'])
        provider_factory.return_value.answer.return_value = 'Python and Django'
        response = self.client.post(reverse('ai_document_assistant'), {
            'candidate': self.candidate.pk, 'question': 'What skills are listed?',
        })
        self.assertContains(response, 'Python and Django [S1]')
        self.assertEqual({source['id'] for source in response.context['sources']}, {'S1'})

    @mock.patch('core.views.get_answer_provider')
    def test_answer_with_unsupported_skill_returns_insufficient_evidence(self, provider_factory):
        provider_factory.return_value.answer.return_value = 'Kubernetes is listed [S1].'
        response = self.client.post(reverse('ai_document_assistant'), {
            'candidate': self.candidate.pk, 'question': 'What skills does Alex have?',
        })
        self.assertContains(response, 'do not provide enough evidence')
        self.assertNotContains(response, 'Kubernetes')
        self.assertEqual(response.context['sources'], [])

    @mock.patch('core.views.get_answer_provider')
    def test_category_headings_are_not_accepted_as_the_skill_list(self, provider_factory):
        self.candidate.extracted_text = (
            '[Page 1]\nSKILLS\nProgramming Languages: Python, Java\n'
            'Web Technologies: React, Node.js\n'
        )
        self.candidate.save(update_fields=['extracted_text'])
        provider_factory.return_value.answer.return_value = (
            'Skills explicitly stated: Programming Languages, Web Technologies [S1] [S2].'
        )
        response = self.client.post(reverse('ai_document_assistant'), {
            'candidate': self.candidate.pk, 'question': 'List the skills explicitly stated.',
        })
        self.assertContains(response, 'do not provide enough evidence')
        self.assertNotContains(response, 'Programming Languages, Web Technologies')

    @mock.patch('core.views.get_answer_provider')
    def test_skills_prompt_requires_explicit_unique_cited_evidence_and_ignores_history(self, provider_factory):
        provider_factory.return_value.answer.return_value = 'Python [S1].'
        self.client.post(reverse('ai_document_assistant'), {
            'candidate': self.candidate.pk, 'question': 'What skills are listed?',
        })
        messages = provider_factory.return_value.answer.call_args.args[0]
        self.assertIn('explicitly present', messages[0]['content'])
        self.assertIn('include each name once', messages[0]['content'])
        self.assertIn('Every factual sentence must end', messages[0]['content'])
        self.assertFalse(any('Python [S1].' in message['content'] for message in messages))

    @mock.patch('core.views.get_answer_provider')
    def test_corrected_text_is_authoritative_and_scoped_history_is_saved(self, provider_factory):
        self.candidate.reviewed_text = '[Page 3]\nCorrected Java experience and verified certification.'
        self.candidate.save(update_fields=['reviewed_text'])
        provider_factory.return_value.answer.return_value = 'Java is stated [S1].'
        response = self.client.post(reverse('ai_document_assistant'), {
            'candidate': self.candidate.pk, 'question': 'What language is stated?',
        })
        self.assertContains(response, 'Page 3')
        self.assertNotContains(response, 'Python engineer with three years')
        self.assertEqual(len([k for k in self.client.session.keys() if k.startswith('smartdoc_assistant_')]), 1)

    @mock.patch('core.views.get_answer_provider')
    def test_unreviewed_handwriting_is_never_sent_to_provider(self, provider_factory):
        self.candidate.processing_mode = Candidate.PROCESSING_HANDWRITING
        self.candidate.extracted_text = 'Unverified recognized Python experience.'
        self.candidate.save(update_fields=['processing_mode', 'extracted_text'])
        response = self.client.post(reverse('ai_document_assistant'), {
            'candidate': self.candidate.pk, 'question': 'Summarize this resume',
        })
        self.assertContains(response, 'must be manually reviewed')
        provider_factory.assert_not_called()

    @mock.patch('core.views.get_answer_provider')
    def test_comparison_requires_two_candidates_and_job(self, provider_factory):
        response = self.client.post(reverse('ai_document_assistant'), {
            'candidate': self.candidate.pk, 'question': 'Compare these candidates',
        })
        self.assertContains(response, 'Select a second candidate and a job')
        provider_factory.assert_not_called()

    @mock.patch('core.views.get_answer_provider')
    def test_comparison_uses_both_candidates_and_selected_job_as_evidence(self, provider_factory):
        other = Candidate.objects.create(
            name='Sam Candidate', email='sam-assistant@example.com', mobile='456', experience='2 years',
            resume='resumes/sam.docx', extraction_status='completed',
            extracted_text='[Page 4]\nJava developer with two years building web services.',
        )
        provider_factory.return_value.answer.return_value = 'Python [S1]; Java [S2].'
        response = self.client.post(reverse('ai_document_assistant'), {
            'candidate': self.candidate.pk, 'compare_candidate': other.pk, 'job': self.job.pk,
            'question': 'Compare these candidates for this role',
        })
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Alex Candidate')
        self.assertContains(response, 'Sam Candidate')
        self.assertContains(response, 'Page 4')
        prompt = provider_factory.return_value.answer.call_args.args[0][-1]['content']
        self.assertIn('Backend engineer', prompt)
        self.assertIn('Python engineer', prompt)
        self.assertIn('Java developer', prompt)

    @mock.patch('core.views.get_answer_provider')
    def test_failed_document_does_not_reach_model(self, provider_factory):
        self.candidate.extraction_status = 'failed'
        self.candidate.save(update_fields=['extraction_status'])
        response = self.client.post(reverse('ai_document_assistant'), {
            'candidate': self.candidate.pk, 'question': 'Summarize this resume',
        })
        self.assertContains(response, 'not marked complete')
        provider_factory.assert_not_called()

    def test_provider_is_disabled_by_default_and_rejects_remote_ollama(self):
        from .assistant_providers import AssistantUnavailable, OllamaProvider, get_answer_provider
        with mock.patch.dict(os.environ, {'SMARTDOC_ASSISTANT_PROVIDER': 'disabled'}):
            with self.assertRaises(AssistantUnavailable):
                get_answer_provider()
        with self.assertRaises(AssistantUnavailable):
            OllamaProvider(endpoint='http://example.com:11434')

    def test_local_provider_bypasses_proxies_and_blocks_bad_responses_and_timeouts(self):
        from urllib.request import ProxyHandler
        from .assistant_providers import AssistantUnavailable, OllamaProvider
        with mock.patch('core.assistant_providers.build_opener') as build_opener:
            response = mock.MagicMock()
            response.__enter__.return_value.read.return_value = b'{"message":{"content":"Python [S1]"}}'
            build_opener.return_value.open.return_value = response
            self.assertEqual(OllamaProvider().answer([{'role': 'user', 'content': 'Question'}]), 'Python [S1]')
            self.assertIsInstance(build_opener.call_args.args[0], ProxyHandler)
            self.assertEqual(build_opener.call_args.args[0].proxies, {})

        with mock.patch('core.assistant_providers.build_opener') as build_opener:
            response = mock.MagicMock()
            response.__enter__.return_value.read.return_value = b'{malformed-json'
            build_opener.return_value.open.return_value = response
            with self.assertRaisesRegex(AssistantUnavailable, 'unavailable or timed out'):
                OllamaProvider().answer([])

        with mock.patch('core.assistant_providers.build_opener') as build_opener:
            build_opener.return_value.open.side_effect = TimeoutError()
            with self.assertRaisesRegex(AssistantUnavailable, 'unavailable or timed out'):
                OllamaProvider().answer([])

    @mock.patch('core.views.get_answer_provider')
    def test_assistant_escapes_model_html_and_scopes_history_by_candidate(self, provider_factory):
        other = Candidate.objects.create(
            name='Morgan Candidate', email='morgan-assistant@example.com', mobile='456', experience='1 year',
            resume='resumes/morgan.pdf', extraction_status='completed',
            extracted_text='[Page 1]\nMorgan has verified Go experience.',
        )
        provider_factory.return_value.answer.return_value = '<b>Python</b> is explicitly stated [S1]'
        first = self.client.post(reverse('ai_document_assistant'), {
            'candidate': self.candidate.pk, 'question': 'What is the candidate name?',
        })
        self.assertContains(first, '&lt;b&gt;Python&lt;/b&gt;', html=False)
        self.assertNotContains(first, '<b>Python</b>', html=False)
        provider_factory.return_value.answer.return_value = 'Morgan has Go experience [S1].'
        self.client.post(reverse('ai_document_assistant'), {
            'candidate': other.pk, 'question': 'What language is stated?',
        })
        prompt = provider_factory.return_value.answer.call_args.args[0]
        self.assertNotIn('Python engineer', prompt[-1]['content'])
        self.assertEqual(len([key for key in self.client.session.keys() if key.startswith('smartdoc_assistant_')]), 2)


class RecruitmentViewTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='recruiter', password='Strong-test-password-123')
        self.client.force_login(self.user)

    def test_dashboard_requires_authentication(self):
        self.client.logout()
        response = self.client.get(reverse('dashboard'))
        self.assertEqual(response.status_code, 302)

    def test_dashboard_loads(self):
        response = self.client.get(reverse('dashboard'))
        self.assertEqual(response.status_code, 200)

    def test_failed_extraction_is_not_matched(self):
        candidate = Candidate.objects.create(
            name='Candidate', email='candidate@example.com', mobile='123',
            experience='1 year', resume='resumes/missing.pdf',
            extraction_status='failed', extraction_error='OCR unavailable',
        )
        job = JobDescription.objects.create(
            title='Python developer', required_skills='python',
            experience='1 year', description='Build Python services', created_by=self.user,
        )
        response = self.client.post(reverse('run_match', args=[candidate.pk]), {'job': job.pk})
        self.assertEqual(response.status_code, 302)
        self.assertFalse(MatchResult.objects.filter(candidate=candidate, job=job).exists())

    def test_hybrid_model_failure_does_not_fall_back_or_overwrite_prior_result(self):
        candidate = Candidate.objects.create(
            name='Reviewed Candidate', email='reviewed@example.com', mobile='123',
            experience='2 years', resume='resumes/reviewed.pdf',
            extraction_status='completed', extracted_text='Reviewed Python experience.',
            extracted_skills='python',
        )
        job = JobDescription.objects.create(
            title='Python role', required_skills='python, django',
            experience='2 years', description='Build backend services.', created_by=self.user,
        )
        prior = MatchResult.objects.create(
            candidate=candidate, job=job, match_percentage=50,
            matching_skills='python', missing_skills='django', recommendation='recommended',
            explanation={'method': 'baseline', 'baseline_score': 50, 'semantic_score': None},
        )
        with mock.patch('core.views.compute_hybrid_match', side_effect=SemanticModelUnavailable('local model missing')):
            response = self.client.post(reverse('run_match', args=[candidate.pk]), {
                'job': job.pk, 'method': 'hybrid',
            })
        self.assertEqual(response.status_code, 302)
        prior.refresh_from_db()
        self.assertEqual(prior.match_percentage, 50)
        self.assertEqual(prior.explanation['method'], 'baseline')

    @unittest.skipUnless(
        bool(os.environ.get('SMARTDOC_SEMANTIC_MODEL_PATH', '').strip())
        and Path(os.environ['SMARTDOC_SEMANTIC_MODEL_PATH'].strip()).is_dir(),
        'Configured local semantic model is not installed.',
    )
    def test_real_local_model_runs_through_hybrid_matching_view(self):
        candidate = Candidate.objects.create(
            name='Local Model Candidate', email='local-model@example.com', mobile='123',
            experience='5 years', resume='resumes/local-model.pdf',
            extraction_status='completed',
            extracted_text='Built Python and Django REST APIs. Led SQL database optimization projects.',
            extracted_skills='python,django',
        )
        job = JobDescription.objects.create(
            title='Local model backend role', required_skills='python,django,sql',
            experience='3 years',
            description='Develop and maintain backend APIs. Experience with relational databases and production services.',
            created_by=self.user,
        )
        clear_model_cache()
        try:
            response = self.client.post(reverse('run_match', args=[candidate.pk]), {
                'job': job.pk, 'method': 'hybrid',
            })
            self.assertEqual(response.status_code, 302)
            result = MatchResult.objects.get(candidate=candidate, job=job)
            self.assertEqual(result.explanation['method'], 'hybrid')
            self.assertEqual(result.explanation['model'], 'sentence-transformers/all-MiniLM-L6-v2')
            self.assertTrue(result.explanation['evidence'])
        finally:
            clear_model_cache()

    def test_candidate_matching_requires_authentication(self):
        candidate = Candidate.objects.create(
            name='Private Candidate', email='private@example.com', mobile='123',
            experience='1 year', resume='resumes/private.pdf', extraction_status='completed',
            extracted_text='Python experience.', extracted_skills='python',
        )
        self.client.logout()
        response = self.client.post(reverse('run_match', args=[candidate.pk]), {})
        self.assertEqual(response.status_code, 302)


class DocumentWorkflowTests(TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.settings_override = override_settings(MEDIA_ROOT=self.temp_dir.name)
        self.settings_override.enable()
        self.user = User.objects.create_user(username='document-reviewer', password='Strong-test-password-123')
        self.client.force_login(self.user)

    def tearDown(self):
        self.settings_override.disable()
        self.temp_dir.cleanup()

    @staticmethod
    def _pdf_bytes(text='A digital PDF resume containing readable Python and Django experience.'):
        return pdf_bytes([text])

    def _candidate_upload(self, email='new@example.com', content=None):
        return self.client.post(reverse('candidate_create'), {
            'name': 'New Candidate', 'email': email, 'mobile': '555-0101',
            'experience': '2 years', 'processing_mode': Candidate.PROCESSING_STANDARD,
            'resume': SimpleUploadedFile('resume.pdf', content or self._pdf_bytes(), content_type='application/pdf'),
        })

    def test_upload_persists_extraction_metadata_and_skills(self):
        result = ExtractionResult(
            '[Page 1]\nTaylor Morgan\ntaylor@example.com\n+1 415 555 0101\n'
            'EDUCATION\nBSc Computing, 2018 - 2022\nSKILLS\nPython, Django',
            'pdf_text', [], 1,
        )
        with mock.patch('core.views.extract_document', return_value=result):
            response = self._candidate_upload()
        candidate = Candidate.objects.get(email='new@example.com')
        self.assertEqual(response.status_code, 302)
        self.assertEqual(candidate.extraction_status, 'completed')
        self.assertEqual(candidate.extraction_method, 'pdf_text')
        self.assertEqual(candidate.extraction_page_count, 1)
        self.assertIsNotNone(candidate.extraction_processed_at)
        self.assertEqual(candidate.get_skills_list(), ['django', 'python'])
        self.assertEqual(candidate.structured_extraction['full_name']['value'], 'Taylor Morgan')
        self.assertEqual(candidate.structured_extraction['education'][0]['sources'][0]['page'], 1)
        detail = self.client.get(reverse('candidate_detail', args=[candidate.pk]))
        self.assertContains(detail, 'Structured resume profile')
        self.assertContains(detail, 'BSc Computing')
        self.assertContains(detail, 'Automatic · verify')

    def test_handwriting_upload_is_review_required_and_does_not_infer_skills(self):
        candidate_result = ExtractionResult(
            '[Page 1]\nMaybe Python, maybe Django.', 'handwriting_trocr',
            ['Handwriting recognition is best-effort.'], 1,
        )
        image = Image.new('RGB', (40, 20), 'white')
        import io
        content = io.BytesIO()
        image.save(content, format='PNG')
        with mock.patch('core.views.extract_handwritten_document', return_value=candidate_result) as process:
            response = self.client.post(reverse('candidate_create'), {
                'name': 'Handwriting Candidate', 'email': 'written@example.com', 'mobile': '555-0105',
                'experience': '1 year', 'processing_mode': Candidate.PROCESSING_HANDWRITING,
                'resume': SimpleUploadedFile('written.png', content.getvalue(), content_type='image/png'),
            })
        candidate = Candidate.objects.get(email='written@example.com')
        self.assertEqual(response.status_code, 302)
        process.assert_called_once()
        self.assertEqual(candidate.processing_mode, Candidate.PROCESSING_HANDWRITING)
        self.assertEqual(candidate.extraction_method, 'handwriting_trocr')
        self.assertEqual(candidate.extraction_status, 'needs_review')
        self.assertEqual(candidate.extracted_skills, '')
        self.assertIn('Maybe Python', candidate.extracted_text)
        self.assertIn('Review extracted text', self.client.get(reverse('candidate_detail', args=[candidate.pk])).content.decode())

        self.client.post(reverse('save_document_review', args=[candidate.pk]), {
            'reviewed_text': 'Verified Python and Django experience.'
        })
        candidate.refresh_from_db()
        self.assertEqual(candidate.extracted_skills, 'django, python')
        self.assertEqual(candidate.extraction_status, 'completed')
        self.assertIn('Maybe Python', candidate.extracted_text)
        self.assertEqual(candidate.reviewed_text, 'Verified Python and Django experience.')
        self.assertEqual(candidate.reviewed_profile['source_type'], 'reviewed_text')
        self.assertEqual(candidate.reviewed_profile['skills'], [])

    def test_handwriting_mode_retry_uses_saved_mode_and_requires_authentication(self):
        candidate = Candidate.objects.create(
            name='Retry Handwriting', email='retry-written@example.com', mobile='555-0106',
            experience='1 year', resume='resumes/retry.png', processing_mode=Candidate.PROCESSING_HANDWRITING,
            extraction_status='failed', extraction_error='Model unavailable', uploaded_by=self.user,
        )
        with mock.patch('core.views.extract_handwritten_document', return_value=ExtractionResult(
            'Readable handwritten line.', 'handwriting_trocr', ['Review output.'], 1,
        )) as process:
            response = self.client.post(reverse('retry_candidate_extraction', args=[candidate.pk]))
        candidate.refresh_from_db()
        process.assert_called_once()
        self.assertEqual(response.status_code, 302)
        self.assertEqual(candidate.extraction_status, 'needs_review')
        self.assertEqual(candidate.get_skills_list(), [])
        self.client.logout()
        self.assertEqual(self.client.post(reverse('retry_candidate_extraction', args=[candidate.pk])).status_code, 302)

    def test_failed_upload_is_saved_with_actionable_error(self):
        response = self._candidate_upload(content=b'%PDF-corrupted file bytes')
        candidate = Candidate.objects.get(email='new@example.com')
        self.assertEqual(response.status_code, 302)
        self.assertEqual(candidate.extraction_status, 'failed')
        self.assertIn('corrupted', candidate.extraction_error.lower())
        self.assertEqual(Candidate.objects.count(), 1)

    def test_unexpected_extraction_error_does_not_expose_server_path(self):
        with mock.patch('core.views.extract_document', side_effect=OSError('cannot read C:\\private\\smartdoc\\resume.pdf')):
            response = self._candidate_upload()
        candidate = Candidate.objects.get(email='new@example.com')
        self.assertEqual(response.status_code, 302)
        self.assertEqual(candidate.extraction_status, 'failed')
        self.assertNotIn('C:\\private', candidate.extraction_error)
        self.assertIn('unexpectedly', candidate.extraction_error)

    def test_review_corrections_drive_skills_and_candidate_matching(self):
        candidate = Candidate.objects.create(
            name='Reviewed Candidate', email='reviewed@example.com', mobile='555-0102',
            experience='3 years', resume='resumes/reviewed.pdf', extracted_text='Original text mentions Java.',
            extracted_skills='java', extraction_status='completed', uploaded_by=self.user,
        )
        job = JobDescription.objects.create(
            title='Django Engineer', required_skills='django', experience='3 years',
            description='Build with Django.', created_by=self.user,
        )
        response = self.client.post(reverse('save_document_review', args=[candidate.pk]), {
            'reviewed_text': 'Reviewed Candidate\nCorrected text: I build production services with Django.'
        })
        candidate.refresh_from_db()
        self.assertEqual(response.status_code, 302)
        self.assertEqual(candidate.extracted_text, 'Original text mentions Java.')
        self.assertIn('Django', candidate.get_effective_text())
        self.assertEqual(candidate.get_skills_list(), ['django'])
        self.assertEqual(candidate.reviewed_profile['source_type'], 'reviewed_text')
        self.assertEqual(candidate.reviewed_profile['full_name']['review_status'], 'derived_from_reviewed_text')

        response = self.client.post(reverse('run_match', args=[candidate.pk]), {'job': job.pk})
        result = MatchResult.objects.get(candidate=candidate, job=job)
        self.assertEqual(response.status_code, 302)
        self.assertEqual(result.match_percentage, 100.0)
        self.assertEqual(result.get_matching_skills_list(), ['django'])

    def test_structured_profile_view_saves_manual_corrections_separately(self):
        original = extract_structured_profile('[Page 1]\nTaylor Morgan\ntaylor@example.com\nSKILLS\nPython, Django')
        candidate = Candidate.objects.create(
            name='Record Name', email='record@example.com', mobile='555-0110', experience='2 years',
            resume='resumes/structured.pdf', extracted_text='Original resume source.',
            extraction_status='completed', structured_extraction=original, uploaded_by=self.user,
        )
        response = self.client.post(reverse('save_structured_profile', args=[candidate.pk]), {
            'full_name': 'Taylor A. Morgan', 'email': 'taylor@example.com', 'phone': '',
            'education': 'BSc Computing', 'skills': 'Python\nDjango\nSQL',
            'projects': '', 'work_experience': '', 'internships': '', 'certifications': '',
            'achievements': '', 'dates': '2018 - 2022',
        })
        candidate.refresh_from_db()
        self.assertEqual(response.status_code, 302)
        self.assertEqual(candidate.name, 'Record Name')
        self.assertEqual(candidate.structured_extraction, original)
        self.assertEqual(candidate.reviewed_profile['full_name']['origin'], 'manual_correction')
        self.assertEqual(candidate.reviewed_profile['email']['sources'][0]['page'], 1)
        self.assertEqual(candidate.get_skills_list(), ['python', 'django', 'sql'])

        job = JobDescription.objects.create(
            title='Python SQL role', required_skills='python, sql', experience='2 years',
            description='Use Python and SQL.', created_by=self.user,
        )
        self.client.post(reverse('run_match', args=[candidate.pk]), {'job': job.pk})
        result = MatchResult.objects.get(candidate=candidate, job=job)
        self.assertEqual(result.match_percentage, 100.0)

        self.client.logout()
        response = self.client.post(reverse('save_structured_profile', args=[candidate.pk]), {})
        self.assertEqual(response.status_code, 302)

    def test_handwriting_structured_fields_are_locked_until_transcription_review(self):
        candidate = Candidate.objects.create(
            name='Unreviewed Writer', email='writer@example.com', mobile='555-0111', experience='1 year',
            resume='resumes/handwritten.png', processing_mode=Candidate.PROCESSING_HANDWRITING,
            extraction_status='needs_review', extracted_text='Possibly Python', uploaded_by=self.user,
        )
        response = self.client.post(reverse('save_structured_profile', args=[candidate.pk]), {
            'full_name': '', 'email': '', 'phone': '', 'education': '', 'skills': 'Python',
            'projects': '', 'work_experience': '', 'internships': '', 'certifications': '',
            'achievements': '', 'dates': '',
        })
        candidate.refresh_from_db()
        self.assertEqual(response.status_code, 302)
        self.assertEqual(candidate.get_skills_list(), [])
        self.assertEqual(candidate.reviewed_profile, {})

    def test_retry_reprocesses_failed_document_and_is_authenticated_post_only(self):
        candidate = Candidate.objects.create(
            name='Retry Candidate', email='retry@example.com', mobile='555-0103',
            experience='1 year', resume='resumes/retry.pdf', extraction_status='failed',
            extraction_error='Tesseract unavailable', uploaded_by=self.user,
        )
        self.assertEqual(self.client.get(reverse('retry_candidate_extraction', args=[candidate.pk])).status_code, 405)
        result = ExtractionResult('Retry recovered readable Python skills.', 'pdf_text', [], 1)
        with mock.patch('core.views.extract_document', return_value=result):
            response = self.client.post(reverse('retry_candidate_extraction', args=[candidate.pk]))
        candidate.refresh_from_db()
        self.assertEqual(response.status_code, 302)
        self.assertEqual(candidate.extraction_status, 'completed')
        self.assertEqual(candidate.extraction_error, '')
        self.assertEqual(candidate.get_skills_list(), ['python'])

        self.client.logout()
        response = self.client.post(reverse('retry_candidate_extraction', args=[candidate.pk]))
        self.assertEqual(response.status_code, 302)

    def test_one_failed_candidate_does_not_change_other_candidate_results(self):
        existing = Candidate.objects.create(
            name='Existing Candidate', email='existing@example.com', mobile='555-0104',
            experience='5 years', resume='resumes/existing.pdf', extracted_text='Existing Python text',
            extracted_skills='python', extraction_status='completed', uploaded_by=self.user,
        )
        previous_text = existing.extracted_text
        response = self._candidate_upload(email='bad@example.com', content=b'%PDF-not-valid')
        existing.refresh_from_db()
        failed = Candidate.objects.get(email='bad@example.com')
        self.assertEqual(response.status_code, 302)
        self.assertEqual(Candidate.objects.count(), 2)
        self.assertEqual(existing.extracted_text, previous_text)
        self.assertEqual(existing.extraction_status, 'completed')
        self.assertEqual(failed.extraction_status, 'failed')


class LoginViewTests(TestCase):
    def test_login_page_renders_and_valid_credentials_sign_in(self):
        User.objects.create_user(username='ui-recruiter', password='Strong-test-password-123')

        response = self.client.get(reverse('login'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'INTELLIGENT DOCUMENTS. EXPLAINABLE INSIGHTS.')

        response = self.client.post(reverse('login'), {
            'username': 'ui-recruiter',
            'password': 'Strong-test-password-123',
        })
        self.assertRedirects(response, reverse('dashboard'))
        self.assertIn('_auth_user_id', self.client.session)

    def test_logout_requires_post_and_clears_authentication(self):
        user = User.objects.create_user(username='logout-recruiter', password='Strong-test-password-123')
        self.client.force_login(user)
        self.assertEqual(self.client.get(reverse('logout')).status_code, 405)
        response = self.client.post(reverse('logout'))
        self.assertRedirects(response, reverse('login'))
        self.assertNotIn('_auth_user_id', self.client.session)


class DeploymentSecurityTests(TestCase):
    def test_production_settings_reject_missing_or_development_secret(self):
        from django.core.exceptions import ImproperlyConfigured
        from resume_screener.settings import validate_secret_key
        for secret in ('', 'dev-only-change-me-before-deployment'):
            with self.subTest(secret='empty' if not secret else 'development-default'):
                with self.assertRaises(ImproperlyConfigured):
                    validate_secret_key(secret, debug=False)
        self.assertEqual(validate_secret_key('a-private-random-secret', debug=False), 'a-private-random-secret')
        self.assertEqual(validate_secret_key('dev-only-change-me-before-deployment', debug=True), 'dev-only-change-me-before-deployment')


class ApplicationPageRenderingTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='ui-reviewer', password='Strong-test-password-123')
        self.client.force_login(self.user)
        self.job = JobDescription.objects.create(
            title='Product Engineer', required_skills='python, django',
            experience='3 years', description='Build product features.', created_by=self.user,
        )
        self.candidate = Candidate.objects.create(
            name='Jamie Candidate', email='jamie@example.com', mobile='555-0100',
            experience='4 years', resume='resumes/jamie.pdf', extracted_text='Python and Django',
            extracted_skills='python, django', extraction_status='completed',
            extraction_method='pdf_text', uploaded_by=self.user,
        )
        MatchResult.objects.create(
            candidate=self.candidate, job=self.job, match_percentage=100,
            matching_skills='python, django', recommendation='highly_recommended',
        )

    def test_existing_application_pages_render(self):
        urls = [
            reverse('dashboard'), reverse('job_list'), reverse('job_create'),
            reverse('job_detail', args=[self.job.pk]), reverse('job_edit', args=[self.job.pk]),
            reverse('job_delete', args=[self.job.pk]), reverse('candidate_list'),
            reverse('candidate_create'), reverse('candidate_detail', args=[self.candidate.pk]),
            reverse('candidate_edit', args=[self.candidate.pk]),
            reverse('candidate_delete', args=[self.candidate.pk]), reverse('ranking'),
        ]
        for url in urls:
            with self.subTest(url=url):
                self.assertEqual(self.client.get(url).status_code, 200)

    def test_dashboard_shows_database_backed_values_and_real_match_result(self):
        response = self.client.get(reverse('dashboard'))
        self.assertContains(response, 'Jamie Candidate')
        self.assertContains(response, 'Product Engineer')
        self.assertContains(response, '100.0%')
        self.assertContains(response, 'data-chart-data="score-chart-data"')

    def test_dashboard_metrics_separate_failed_review_and_pending_documents(self):
        Candidate.objects.create(
            name='Failed', email='failed-dashboard@example.com', mobile='1', experience='0 years',
            resume='resumes/failed.pdf', extraction_status='failed', extraction_error='Unreadable',
        )
        Candidate.objects.create(
            name='Needs review', email='review-dashboard@example.com', mobile='2', experience='0 years',
            resume='resumes/review.pdf', extraction_status='needs_review', extraction_warnings='Verify text',
        )
        Candidate.objects.create(
            name='Pending', email='pending-dashboard@example.com', mobile='3', experience='0 years',
            resume='resumes/pending.pdf', extraction_status='pending',
        )
        response = self.client.get(reverse('dashboard'))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context['documents_processed'], 3)
        self.assertEqual(response.context['successful_extractions'], 1)
        self.assertEqual(response.context['failed_extractions'], 1)
        self.assertEqual(response.context['documents_needing_review'], 1)
        self.assertEqual(response.context['pending_documents'], 1)
        self.assertEqual(response.context['total_matches'], 1)

    def test_dashboard_filters_and_score_distribution_use_saved_records(self):
        date = self.candidate.uploaded_at.date().isoformat()
        response = self.client.get(reverse('dashboard'), {
            'job': self.job.pk, 'method': 'baseline', 'status': 'completed', 'date_from': date, 'date_to': date,
        })
        self.assertEqual(response.context['total_candidates'], 1)
        self.assertEqual(response.context['total_matches'], 1)
        self.assertEqual([row['count'] for row in response.context['score_distribution']], [0, 0, 0, 0, 1])
        self.assertContains(response, 'Jamie Candidate')

    def test_dashboard_shows_comparable_saved_hybrid_components(self):
        result = MatchResult.objects.get(candidate=self.candidate, job=self.job)
        result.match_percentage = 70
        result.explanation = {'method': 'hybrid', 'baseline_score': 50, 'semantic_score': 100}
        result.save(update_fields=['match_percentage', 'explanation'])
        response = self.client.get(reverse('dashboard'), {'method': 'hybrid'})
        self.assertEqual(response.context['method_comparison'], {
            'count': 1, 'baseline_average': 50.0, 'hybrid_average': 70.0,
        })
        self.assertEqual(response.context['total_matches'], 1)

    def test_dashboard_empty_dataset_has_zero_metrics_and_no_score_chart(self):
        Candidate.objects.all().delete()
        JobDescription.objects.all().delete()
        MatchResult.objects.all().delete()
        response = self.client.get(reverse('dashboard'))
        self.assertEqual(response.context['total_candidates'], 0)
        self.assertEqual(response.context['total_jobs'], 0)
        self.assertEqual(response.context['total_matches'], 0)
        self.assertFalse(response.context['skill_stats']['skills'])
        self.assertContains(response, 'No scores in this scope')

    def test_excel_export_still_returns_a_workbook(self):
        response = self.client.get(reverse('export_excel'))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response['Content-Type'],
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        )
