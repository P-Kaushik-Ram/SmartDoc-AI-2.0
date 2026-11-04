import io
import tempfile
from pathlib import Path
from unittest import mock

from django.conf import settings
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import SimpleTestCase, TestCase
from django.urls import reverse
from docx import Document
from PIL import Image

from . import handwriting, hybrid_ocr
from .utils import COMMON_SKILLS, DocumentProcessingError, ExtractionResult, compute_match, extract_skills_from_text

JD = 'Python, Django, PostgreSQL, Git, HTML, CSS, C++, Java, C'


class SkillAccuracyTests(SimpleTestCase):
    def test_entire_existing_vocabulary_remains_supported(self):
        self.assertEqual(set(extract_skills_from_text(', '.join(COMMON_SKILLS))), set(COMMON_SKILLS))

    def test_token_boundaries(self):
        for text, expected in [('C', ['c']), ('C++', ['c++']), ('C#', ['c#']),
                               ('JavaScript', ['javascript']), ('Java', ['java']),
                               ('PostgreSQL', ['postgresql']), ('SQL', ['sql']),
                               ('JavaScript PostgreSQL C++ C#', ['c#', 'c++', 'javascript', 'postgresql']),
                               ('scalable javabeans nosql', []), ('', [])]:
            with self.subTest(text=text):
                self.assertEqual(extract_skills_from_text(text), expected)

    def test_case_duplicates_and_punctuation(self):
        self.assertEqual(extract_skills_from_text('(c++), C++; [C#]; c, JAVA/javascript; SQL/PostgreSQL.'),
                         ['c', 'c#', 'c++', 'java', 'javascript', 'postgresql', 'sql'])
        self.assertEqual(compute_match(['Python', 'python'], ['PYTHON', 'python', 'Java'], precision=2),
                         (50.0, ['python'], ['java']))

    def test_empty_and_no_matches(self):
        self.assertEqual(compute_match(['python'], []), (0.0, [], []))
        self.assertEqual(compute_match([], ['java']), (0.0, [], ['java']))

    def test_all_nine_required_skills(self):
        skills = extract_skills_from_text(JD)
        self.assertEqual(len(skills), 9)
        score, matched, missing = compute_match(['CSS', 'Django', 'Git', 'HTML', 'PostgreSQL', 'Python'], skills, precision=2)
        self.assertEqual(score, 66.67)
        self.assertEqual(missing, ['c', 'c++', 'java'])
        self.assertEqual(len(matched), 6)


class ResumeAccuracyIntegrationTests(TestCase):
    def resume(self, text):
        document = Document()
        document.add_paragraph(text)
        stream = io.BytesIO()
        document.save(stream)
        return SimpleUploadedFile('resume.docx', stream.getvalue())

    def test_actual_upload_controls_score_and_denominator(self):
        for text, score, count in [('Python Django PostgreSQL Git HTML CSS', 66.67, 6),
                                   ('C++', 11.11, 1), ('No technical experience.', 0.0, 0)]:
            with self.subTest(text=text):
                response = self.client.post(reverse('resume_analyzer'), {'document': self.resume(text), 'job_description': JD})
                result = response.context['result']
                self.assertEqual(result['score'], score)
                self.assertEqual(len(result['requirements']), 9)
                self.assertEqual(len(result['matched']), count)
                self.assertContains(response, 'Job Skill Coverage')
                self.assertContains(response, 'Total detected required skills: 9')
                self.assertIn('Job Skill Coverage', response.context['report'])
                self.assertTrue(all(skill in ' '.join(result['job']['recommendations']).casefold() for skill in result['missing']))

    def test_empty_jd_generates_automatic_dashboard(self):
        response = self.client.post(reverse('resume_analyzer'), {'document': self.resume('Python'), 'job_description': ''})
        self.assertIsNone(response.context['result']['job'])
        self.assertFalse(response.context['form'].errors)
        self.assertContains(response, 'Resume Analytics Dashboard')

    def test_get_forms_are_unbound_and_confirmation_is_not_in_title(self):
        for name in ['document_studio', 'resume_analyzer']:
            response = self.client.get(reverse(name))
            self.assertFalse(response.context['form'].is_bound)
            self.assertNotContains(response, 'This field is required.')
        html = self.client.get(reverse('document_studio')).content.decode()
        self.assertEqual(html.count('data-confirm-replace'), 1)
        self.assertNotIn('<dialog', html.split('</title>')[0])

    def test_engine_outputs_are_escaped_in_html_and_separate_in_json(self):
        result = ExtractionResult('', 'handwriting_hybrid', [], 1,
                                  {'trocr': {'text': '</textarea><script>bad()</script>', 'error': '', 'warnings': []},
                                   'tesseract': {'text': 'Other result', 'error': '', 'warnings': []}})
        for json_request in [False, True]:
            with mock.patch('core.workspaces.extract_handwritten_document', return_value=result):
                response = self.client.post(reverse('document_studio'),
                    {'mode': 'handwriting', 'document': SimpleUploadedFile('notes.png', b'image')},
                    HTTP_ACCEPT='application/json' if json_request else 'text/html')
            self.assertEqual(response.status_code, 200)
            if json_request:
                self.assertEqual(response.json()['text'], '')
                self.assertEqual(response.json()['engine_results']['tesseract']['text'], 'Other result')
            else:
                self.assertContains(response, 'Use TrOCR Result')
                self.assertContains(response, 'Use Tesseract Result')
                self.assertContains(response, '&lt;script&gt;bad()&lt;/script&gt;')
                self.assertNotContains(response, '<script>bad()</script>')


class HybridHandwritingTests(SimpleTestCase):
    def test_noisy_fixture_has_three_lines(self):
        with Image.open(Path(settings.BASE_DIR) / 'demo/handwriting/handwriting_test_sample.png') as image:
            self.assertEqual(len(handwriting.preprocess_handwriting_image(image)), 3)

    def test_table_cells_reconstruct_rows_without_inventing_values(self):
        torch = mock.Mock()
        torch.inference_mode.return_value = mock.MagicMock()
        processor = mock.Mock()
        processor.batch_decode.side_effect = [['A'], ['6'], ['B']]
        model = mock.Mock()
        crops = [Image.new('RGB', (80, 30), 'white') for _ in range(3)]
        with mock.patch.object(handwriting, 'preprocess_handwriting_image', return_value=(crops, [(10, 0, 2), (10, 1, 2), (40, 0, 2)])), \
             mock.patch.object(handwriting, '_load_backend', return_value=(torch, processor, model, 'cpu')):
            self.assertEqual(handwriting.recognize_handwriting(crops[0]), 'A\t6\nB\t')

    def extract(self, trocr, tesseract):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'notes.png'
            Image.new('RGB', (100, 100), 'white').save(path)
            with mock.patch.object(handwriting, 'recognize_handwriting_image', side_effect=trocr), \
                 mock.patch.object(hybrid_ocr, 'recognize_tesseract_handwriting', side_effect=tesseract):
                return hybrid_ocr.extract_handwritten_document(path)

    def test_both_engines_are_separate_and_final_text_starts_empty(self):
        result = self.extract(lambda image: 'Alpha', lambda image: 'Beta')
        self.assertEqual(result.text, '')
        self.assertEqual(result.engine_results['trocr']['text'], '[Page 1]\nAlpha')
        self.assertEqual(result.engine_results['tesseract']['text'], '[Page 1]\nBeta')

    def test_either_engine_failure_preserves_other_output(self):
        def fail(image):
            raise DocumentProcessingError('Engine unavailable')
        for first, second, available in [(fail, lambda im: 'Other', 'tesseract'),
                                          (lambda im: 'Other', fail, 'trocr')]:
            with self.subTest(available=available):
                result = self.extract(first, second)
                self.assertIn('Other', result.engine_results[available]['text'])
                self.assertIn('Engine unavailable', ' '.join(result.warnings))

    def test_all_empty_results_fail_actionably(self):
        with self.assertRaisesRegex(DocumentProcessingError, 'No readable handwriting'):
            self.extract(lambda image: '', lambda image: '')

    @mock.patch('core.handwriting._load_backend')
    @mock.patch('core.hybrid_ocr._tesseract')
    def test_blank_never_invokes_either_engine(self, tesseract, backend):
        with self.assertRaises(DocumentProcessingError):
            hybrid_ocr.extract_handwritten_document(Path(settings.BASE_DIR) / 'demo/handwriting/blank.png')
        tesseract.assert_not_called()
        backend.assert_not_called()
