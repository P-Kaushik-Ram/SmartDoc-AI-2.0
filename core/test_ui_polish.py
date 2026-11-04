"""Presentation contracts for the served public workspaces."""
from django.test import SimpleTestCase
from django.urls import reverse


class PublicWorkspacePolishTests(SimpleTestCase):
    def test_shared_navigation_has_mobile_control_and_current_page(self):
        for route in ['home', 'resume_analyzer', 'document_studio', 'about']:
            with self.subTest(route=route):
                response = self.client.get(reverse(route))
                self.assertContains(response, 'aria-controls="main-navigation"')
                self.assertContains(response, 'id="main-navigation"')
                self.assertContains(response, f'href="{reverse(route)}" aria-current="page"', count=1)
                self.assertContains(response, 'Skip to content')

    def test_home_and_about_explain_optional_cloud_processing(self):
        home = self.client.get(reverse('home'))
        self.assertContains(home, 'Explore Resume Analyzer')
        self.assertContains(home, 'Local tools or Gemini with explicit consent.')
        self.assertNotContains(home, 'No paid APIs.')
        about = self.client.get(reverse('about'))
        self.assertContains(about, 'explicit cloud-processing consent')

    def test_upload_workspaces_offer_file_removal_and_live_feedback(self):
        for route in ['resume_analyzer', 'document_studio']:
            response = self.client.get(reverse(route))
            self.assertContains(response, 'data-remove-file hidden')
            self.assertContains(response, 'aria-live="polite"')
            self.assertContains(response, 'csrfmiddlewaretoken')
