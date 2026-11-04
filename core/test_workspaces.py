"""Workspace integration tests; existing processing has its own regression suite."""
import io
from pathlib import Path
from unittest import mock

from docx import Document
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client, TestCase
from django.urls import reverse

from .models import Candidate, JobDescription, MatchResult
from .pdf_test_fixtures import pdf_bytes
from .utils import DocumentProcessingError, ExtractionResult


def resume_file():
    document = Document()
    document.add_paragraph('Alex Example — Software engineer with Python and Django project experience.')
    content = io.BytesIO()
    document.save(content)
    return SimpleUploadedFile('resume.docx', content.getvalue())



class WorkspaceTests(TestCase):
    def test_empty_submission_returns_validation_errors(self):
        response = self.client.post(reverse('document_studio'), {}, HTTP_ACCEPT='application/json')
        self.assertEqual(response.status_code, 400)
        self.assertIn('document', response.json()['errors'])
        response = self.client.post(reverse('resume_analyzer'), {})
        self.assertContains(response, 'This field is required.')

    def test_primary_pages_are_public_and_independent_of_recruitment(self):
        for name in ['home', 'resume_analyzer', 'document_studio', 'about']:
            with self.subTest(name=name):
                response = self.client.get(reverse(name))
                self.assertEqual(response.status_code, 200)
                self.assertContains(response, 'SmartDoc')
                self.assertContains(response, reverse('resume_analyzer'))
                self.assertContains(response, reverse('document_studio'))
        self.assertRedirects(self.client.get(reverse('dashboard')), '/login/?next=/dashboard/')

    def test_real_docx_resume_analysis_has_explainable_score_and_no_db_writes(self):
        response = self.client.post(reverse('resume_analyzer'), {
            'document': resume_file(), 'job_description': 'We require Python, Django, and Docker.',
        })
        self.assertEqual(response.status_code, 200)
        result = response.context['result']
        self.assertEqual(result['score'], 66.67)
        self.assertEqual(result['matched'], ['django', 'python'])
        self.assertEqual(result['missing'], ['docker'])
        self.assertContains(response, 'Download report')
        self.assertIn('66.67%', response.context['report'])
        self.assertFalse(Candidate.objects.exists())
        self.assertFalse(JobDescription.objects.exists())
        self.assertFalse(MatchResult.objects.exists())

    def test_unrecognized_job_skills_do_not_produce_a_score(self):
        response = self.client.post(reverse('resume_analyzer'), {
            'document': resume_file(), 'job_description': 'An enthusiastic colleague.',
        })
        self.assertIsNone(response.context['result']['job']['score'])
        self.assertContains(response, 'Resume Analytics Dashboard')
        self.assertContains(response, 'No supported skills were found')

    def test_real_digital_pdf_extraction(self):
        content = pdf_bytes(['SmartDoc converts this real digital document into editable text.'])
        response = self.client.post(reverse('document_studio'), {
            'document': SimpleUploadedFile('notes.pdf', content), 'mode': 'printed',
        }, HTTP_ACCEPT='application/json')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['method'], 'pdf_text')
        self.assertIn('real digital document', response.json()['text'])

    def test_printed_and_handwriting_use_distinct_processors_and_remove_temp_files(self):
        seen = []

        def extract(path, **kwargs):
            self.assertTrue(Path(path).is_file())
            seen.append(path)
            return ExtractionResult('Recognized text', 'test', ['Review required'], 1)

        for mode, selected, other in [
            ('printed', 'extract_document', 'extract_handwritten_document'),
            ('handwriting', 'extract_handwritten_document', 'extract_document'),
        ]:
            with self.subTest(mode=mode), mock.patch('core.workspaces.' + selected, side_effect=extract) as processor, mock.patch('core.workspaces.' + other) as unused:
                response = self.client.post(reverse('document_studio'), {
                    'document': SimpleUploadedFile('notes.png', b'placeholder'), 'mode': mode,
                }, HTTP_ACCEPT='application/json')
                self.assertEqual(response.status_code, 200)
                processor.assert_called_once()
                unused.assert_not_called()
                self.assertEqual(response.json()['warnings'], ['Review required'])
        self.assertTrue(all(not Path(path).exists() for path in seen))

    def test_validation_blocks_invalid_mode_docx_handwriting_and_bad_signature(self):
        for mode, upload in [
            ('invalid', resume_file()), ('handwriting', resume_file()),
            ('printed', SimpleUploadedFile('fake.pdf', b'not a PDF')),
            ('printed', SimpleUploadedFile('empty.png', b'')),
            ('printed', SimpleUploadedFile('notes.exe', b'not supported')),
            ('printed', SimpleUploadedFile('large.pdf', b'x' * (10 * 1024 * 1024 + 1))),
        ]:
            with self.subTest(mode=mode, file=upload.name):
                response = self.client.post(reverse('document_studio'), {
                    'document': upload, 'mode': mode,
                }, HTTP_ACCEPT='application/json')
                self.assertEqual(response.status_code, 400)
                self.assertIn('errors', response.json())

    def test_processing_failure_does_not_expose_private_paths(self):
        with mock.patch('core.workspaces.extract_document', side_effect=RuntimeError('C:/private/secret')):
            with self.assertLogs('core.workspaces', level='ERROR'):
                response = self.client.post(reverse('document_studio'), {
                    'document': resume_file(), 'mode': 'printed',
                }, HTTP_ACCEPT='application/json')
        self.assertEqual(response.status_code, 400)
        self.assertNotContains(response, 'C:/private/secret', status_code=400)

    def test_known_handwriting_setup_failure_is_actionable(self):
        with mock.patch('core.workspaces.extract_handwritten_document', side_effect=DocumentProcessingError('Install local TrOCR weights.')):
            response = self.client.post(reverse('document_studio'), {
                'document': SimpleUploadedFile('notes.png', b'placeholder'), 'mode': 'handwriting',
            }, HTTP_ACCEPT='application/json')
        self.assertContains(response, 'Install local TrOCR weights.', status_code=400)

    def test_temp_file_removed_after_failure(self):
        from .workspaces import process_upload
        seen = []
        def fail(path):
            seen.append(path)
            raise DocumentProcessingError('Unreadable file')
        with mock.patch('core.workspaces.extract_document', side_effect=fail):
            with self.assertRaises(DocumentProcessingError):
                process_upload(resume_file())
        self.assertFalse(Path(seen[0]).exists())

    def test_html_fallback_and_output_escaping(self):
        with mock.patch('core.workspaces.extract_document', return_value=ExtractionResult('</textarea><script>alert(1)</script>', 'test', [], 1)):
            response = self.client.post(reverse('document_studio'), {'document': resume_file(), 'mode': 'printed'})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '&lt;script&gt;alert(1)&lt;/script&gt;')
        self.assertNotContains(response, '<script>alert(1)</script>')

    def test_csrf_required_on_processing(self):
        response = Client(enforce_csrf_checks=True).post(reverse('document_studio'), {'document': resume_file(), 'mode': 'printed'})
        self.assertEqual(response.status_code, 403)
