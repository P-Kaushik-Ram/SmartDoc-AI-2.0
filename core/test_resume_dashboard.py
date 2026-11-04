"""P5 analytics unit and real-file integration coverage."""
from pathlib import Path
from unittest import mock

from django.conf import settings
from django.core import signing
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client, SimpleTestCase, TestCase
from django.urls import reverse

from .resume_dashboard import analyze, build_report, categorize
from .test_workspaces import resume_file
from .utils import ExtractionResult, extract_skills_from_text

SAMPLE = (Path(settings.BASE_DIR) / 'demo/p5/resume-source.txt').read_text(encoding='utf-8')


class AnalyticsParserTests(SimpleTestCase):
    def test_personal_information_and_links(self):
        personal = {row['label']: row['value'] for row in analyze(SAMPLE)['personal']}
        self.assertEqual(personal['Name'], 'Alex Morgan')
        self.assertEqual(personal['Email'], 'alex@example.com')
        self.assertIn('555-0147', personal['Phone'])
        self.assertEqual(personal['Location'], 'Pune, India')
        self.assertIn('linkedin.com/in/', personal['LinkedIn'])
        self.assertIn('github.com/', personal['GitHub'])
        self.assertEqual(personal['Portfolio'], 'https://alex.example.com')

    def test_project_or_section_is_not_candidate_name(self):
        for text in ['Projects\nAlex Dashboard\nBuilt Python tools', 'Technical Skills\nPython Django', 'Document Management System\nProjects\nProject: Tools']:
            with self.subTest(text=text):
                self.assertFalse(analyze(text)['personal'][0]['value'])

    def test_cv_header_and_explicit_name_label(self):
        for text in ['Curriculum Vitae\nAlex Morgan\nSkills\nPython', 'Resume\nName: Alex Morgan\nSkills\nPython']:
            self.assertEqual(analyze(text)['personal'][0]['value'], 'Alex Morgan')

    def test_unlabeled_single_header_website(self):
        personal = {row['label']: row['value'] for row in analyze('Alex Morgan\nhttps://alex.example.com\nProjects\nProject: Tool\nBuilt Django tools')['personal']}
        self.assertEqual(personal['Portfolio'], 'https://alex.example.com')

    def test_skill_categories_deduplicate_and_keep_one_category(self):
        skills = extract_skills_from_text('C++ C# JavaScript Python Python Django PostgreSQL HTML CSS Git')
        categories = categorize(skills + skills)
        self.assertEqual(sum(row['count'] for row in categories), len(skills))
        all_skills = [s for row in categories for s in row['skills']]
        self.assertIn('C++', all_skills)
        self.assertIn('C#', all_skills)
        self.assertIn('PostgreSQL', all_skills)
        self.assertNotIn('C', all_skills)
        self.assertNotIn('Java', all_skills)

    def test_education_and_explicit_graduation(self):
        entry = analyze(SAMPLE)['education'][0]
        self.assertEqual(entry['institution'], 'Example University')
        self.assertEqual(entry['specialization'], 'Computer Science')
        self.assertEqual(entry['year'], '2025')
        text = SAMPLE.replace('Graduation: 2025', 'Attended 2021 - 2025')
        self.assertEqual(analyze(text)['education'][0]['year'], '')

    def test_multiple_education_entries(self):
        result = analyze('Education\nBachelor of Science\nExample University\nGraduation: 2022\nMaster of Science\nOther University\nExpected 2026')
        self.assertEqual(len(result['education']), 2)
        self.assertEqual(result['education'][1]['year'], '2026')

    def test_experience_is_separate_from_project_dates(self):
        result = analyze(SAMPLE.replace('Project: Portfolio', 'Project: Portfolio | 2020 - 2026'))
        self.assertEqual(result['experience_count'], 1)
        self.assertEqual(result['experience'][0]['organization'], 'Example Labs')
        self.assertIn('2024', result['experience'][0]['dates'])
        self.assertNotIn('2020', result['experience'][0]['dates'])

    def test_projects_and_only_explicit_technologies(self):
        result = analyze(SAMPLE)
        self.assertEqual(result['project_count'], 2)
        self.assertEqual(result['projects'][1]['technologies'], ['CSS', 'HTML'])
        self.assertNotIn('Python', result['projects'][1]['technologies'])

    def test_ambiguous_sections_preserve_source_and_unknown_counts(self):
        result = analyze('Projects\nWorked on various things.\nExperience\nHelped a team.\nEducation\nStudied science.')
        self.assertIsNone(result['project_count'])
        self.assertIsNone(result['experience_count'])
        self.assertEqual(result['projects_source'], 'Worked on various things.')
        self.assertEqual(result['education'], [])

    def test_fresher_can_have_full_completeness_without_employment(self):
        text = SAMPLE.split('Internships')[0]
        result = analyze(text)
        self.assertEqual(result['completeness'], 100)
        self.assertIsNone(result['experience_count'])
        self.assertTrue(all(row['detected'] for row in result['rubric']))

    def test_empty_text_has_no_fabricated_data(self):
        result = analyze('')
        self.assertEqual(result['skills'], [])
        self.assertEqual(result['completeness'], 0)
        self.assertTrue(all(not row['value'] for row in result['personal']))
        self.assertIsNone(result['project_count'])

    def test_missing_sections_have_explainable_recommendations(self):
        result = analyze('Alex Morgan\nPython')
        self.assertEqual(result['completeness'], 17)
        self.assertTrue(any('email' in tip for tip in result['recommendations']))
        self.assertIsNone(result['job'])

    def test_optional_matching_boundaries_and_numerator(self):
        result = analyze('C++ C# JavaScript Python Django PostgreSQL HTML CSS Git', job_description='C C++ C# Java JavaScript Python Django PostgreSQL HTML CSS Git')
        self.assertEqual(result['missing'], ['c', 'java'])
        self.assertEqual(len(result['matched']), 9)
        self.assertEqual(len(result['requirements']), 11)
        self.assertEqual(result['score'], 81.82)

    def test_unrecognized_job_skills_not_zero_score(self):
        result = analyze(SAMPLE, job_description='Enthusiasm and care')
        self.assertIsNone(result['job']['score'])
        self.assertEqual(result['completeness'], 100)

    def test_report_contains_sections_unicode_and_optional_job(self):
        report = build_report(analyze(SAMPLE, filename='résumé.docx', job_description='C++ C# Python Docker'))
        for phrase in ['résumé.docx', 'C++', 'C#', 'Resume Completeness: 100%', 'Education', 'Experience', 'Projects', 'Job Skill Coverage: 75.0%', '3 of 4', 'six equal criteria']:
            self.assertIn(phrase, report)


class AnalyticsIntegrationTests(TestCase):
    def test_upload_without_jd_generates_real_docx_dashboard(self):
        path = Path(settings.BASE_DIR) / 'demo/p5/analytics-resume.docx'
        response = self.client.post(reverse('resume_analyzer'), {'document': SimpleUploadedFile(path.name, path.read_bytes())})
        self.assertContains(response, 'Resume Analytics Dashboard')
        self.assertEqual(response.context['result']['project_count'], 2)
        self.assertEqual(response.context['result']['completeness'], 100)
        self.assertIsNone(response.context['result']['job'])
        self.assertContains(response, 'skill-chart-data')
        self.assertIn('no-store', response['Cache-Control'])
        self.assertNotIn('sessionid', response.cookies)

    def test_real_pdf_resume_dashboard(self):
        path = Path(settings.BASE_DIR) / 'demo/p5/analytics-resume.pdf'
        response = self.client.post(reverse('resume_analyzer'), {'document': SimpleUploadedFile(path.name, path.read_bytes())})
        self.assertEqual(response.context['result']['method'], 'pdf_text')
        self.assertEqual(response.context['result']['project_count'], 2)
        self.assertEqual(response.context['result']['completeness'], 100)

    def test_existing_demo_docx(self):
        path = Path(settings.BASE_DIR) / 'demo/sample-resume.docx'
        response = self.client.post(reverse('resume_analyzer'), {'document': SimpleUploadedFile(path.name, path.read_bytes())})
        self.assertEqual(response.context['result']['personal'][0]['value'], 'Alex Morgan')
        self.assertIn('python', response.context['result']['skills'])

    def test_job_submission_reuses_signed_result_without_upload_or_db(self):
        response = self.client.post(reverse('resume_analyzer'), {'document': resume_file()})
        token = response.context['analysis_token']
        with mock.patch('core.workspaces.process_upload') as processor:
            response = self.client.post(reverse('resume_analyzer'), {'analysis_token': token, 'job_description': 'Python Django Docker'})
        processor.assert_not_called()
        self.assertEqual(response.context['result']['score'], 66.67)
        self.assertContains(response, '2 of 3 detected job-description skills matched.')
        self.assertIn('Job Skill Coverage: 66.67%', response.context['report'])

    def test_invalid_or_expired_token_is_rejected(self):
        for token in ['tampered', signing.dumps({'text':'Python'}, salt='wrong')]:
            response = self.client.post(reverse('resume_analyzer'), {'analysis_token': token})
            self.assertContains(response, 'expired or is invalid')
            self.assertNotIn('result', response.context)
        with mock.patch('core.workspaces.signing.loads', side_effect=signing.SignatureExpired()):
            response = self.client.post(reverse('resume_analyzer'), {'analysis_token': 'token'})
        self.assertContains(response, 'expired or is invalid')

    def test_bad_uploads_are_handled(self):
        for name, content in [('fake.pdf', b'garbage'), ('fake.docx', b'PK\x03\x04broken'), ('notes.exe', b'x'), ('empty.pdf', b''), ('huge.pdf', b'x' * (10*1024*1024+1))]:
            with self.subTest(name=name):
                response = self.client.post(reverse('resume_analyzer'), {'document': SimpleUploadedFile(name, content)})
                self.assertNotIn('result', response.context)
                self.assertTrue(response.context['form'].errors)

    def test_unreadable_resume(self):
        with mock.patch('core.workspaces.extract_document', return_value=ExtractionResult('  ', 'test', [], 1)):
            response = self.client.post(reverse('resume_analyzer'), {'document': resume_file()})
        self.assertContains(response, 'No readable text')
        self.assertNotIn('result', response.context)

    def test_missing_fields_and_html_escaping(self):
        with mock.patch('core.workspaces.extract_document', return_value=ExtractionResult('</textarea><script>bad()</script>', 'test', [], 1)):
            response = self.client.post(reverse('resume_analyzer'), {'document': resume_file()})
        self.assertContains(response, 'Not detected')
        self.assertContains(response, 'Not determined')
        self.assertContains(response, '&lt;script&gt;bad()&lt;/script&gt;')
        self.assertNotContains(response, '<script>bad()</script>')
        self.assertNotContains(response, 'skill-chart-data')

    def test_csrf_required_on_upload_and_optional_matching(self):
        client = Client(enforce_csrf_checks=True)
        for data in [{'document': resume_file()}, {'analysis_token': 'token', 'job_description':'Python'}]:
            self.assertEqual(client.post(reverse('resume_analyzer'), data).status_code, 403)

    def test_no_sensitive_exception_logging(self):
        with mock.patch('core.workspaces.process_upload', side_effect=RuntimeError('PRIVATE RESUME')), self.assertLogs('core.workspaces', level='ERROR') as logs:
            response = self.client.post(reverse('resume_analyzer'), {'document': resume_file()})
        self.assertNotIn('PRIVATE RESUME', ''.join(logs.output))
        self.assertContains(response, 'Processing failed unexpectedly')
