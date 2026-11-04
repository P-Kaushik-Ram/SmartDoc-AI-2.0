"""Template integration contracts; runtime theme behavior is exercised in JS/browser tests."""
from django.test import SimpleTestCase
from django.urls import reverse


class DualThemeTemplateTests(SimpleTestCase):
    def test_theme_bootstrap_precedes_styles_in_each_layout(self):
        for route in ['home', 'resume_analyzer', 'document_studio', 'about', 'login']:
            with self.subTest(route=route):
                response = self.client.get(reverse(route))
                html = response.content.decode()
                self.assertIn('data-theme="light"', html)
                self.assertLess(html.index('core/js/theme.js'), html.index('core/css/themes.css'))
                self.assertIn('data-theme-toggle', html)
                self.assertIn('aria-label="Switch to dark mode"', html)

    def test_home_uses_decorative_local_illustration_and_real_workspace_links(self):
        response = self.client.get(reverse('home'))
        self.assertContains(response, 'class="intelligence-art" aria-hidden="true"')
        self.assertContains(response, 'class="feature-card"', count=4)
        self.assertContains(response, '?mode=printed')
        self.assertContains(response, '?mode=handwriting')

    def test_handwriting_keeps_consent_and_engine_controls(self):
        response = self.client.get(reverse('document_studio'), {'mode':'handwriting'})
        self.assertContains(response, 'name="cloud_consent"')
        self.assertFalse(response.context['form']['cloud_consent'].value())
        for engine in ['gemini','trocr','tesseract','hybrid']:
            self.assertContains(response, f'value="{engine}"')
