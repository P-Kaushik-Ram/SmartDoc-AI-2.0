"""Geometry and blank-image regressions use synthetic fixtures."""
from pathlib import Path
from unittest import mock

from django.conf import settings
from django.test import SimpleTestCase
from PIL import Image, ImageDraw

from .handwriting import preprocess_handwriting_image, recognize_handwriting
from .utils import DocumentProcessingError


class HandwritingUpgradeTests(SimpleTestCase):
    def sample(self, name):
        with Image.open(Path(settings.BASE_DIR) / 'demo/handwriting' / f'{name}.png') as source:
            return source.copy()

    def test_fixture_lines_and_rotated_lines(self):
        for name, count in [('single', 1), ('cursive', 3), ('border', 3),
                            ('rotated', 3), ('sideways', 1), ('low_quality', 1)]:
            with self.subTest(name=name):
                lines = preprocess_handwriting_image(self.sample(name))
                self.assertEqual(len(lines), count)
                self.assertTrue(all(line.width > line.height for line in lines))

    @mock.patch('core.handwriting._load_backend')
    def test_blank_and_transparent_images_do_not_load_model(self, backend):
        for image in [self.sample('blank'), Image.new('RGBA', (500, 250), (0, 0, 0, 0))]:
            with self.subTest(mode=image.mode), self.assertRaisesRegex(DocumentProcessingError, 'No text lines'):
                recognize_handwriting(image)
        backend.assert_not_called()

    @mock.patch('core.handwriting._load_backend')
    def test_border_only_does_not_reach_model(self, backend):
        image = Image.new('RGB', (900, 600), 'white')
        ImageDraw.Draw(image).rectangle((0, 0, 899, 599), outline='black', width=5)
        with self.assertRaises(DocumentProcessingError):
            recognize_handwriting(image)
        backend.assert_not_called()

    @mock.patch('core.handwriting._load_backend')
    def test_unsupported_languages_do_not_load_models(self, backend):
        for language in ['ta', 'hi', 'unknown']:
            with self.subTest(language=language), self.assertRaisesRegex(DocumentProcessingError, 'Only English'):
                recognize_handwriting(self.sample('single'), language=language)
        backend.assert_not_called()

    def test_exif_orientation_at_service_boundary(self):
        image = self.sample('single').rotate(90, expand=True)
        image.getexif()[274] = 6
        self.assertEqual(len(preprocess_handwriting_image(image)), 1)

    def test_opposite_skew_does_not_recognize_resampled_paper_edge(self):
        image = self.sample('cursive').rotate(-4, expand=True, fillcolor='white')
        self.assertEqual(len(preprocess_handwriting_image(image)), 3)

    def test_grayscale_strokes_are_not_binarized(self):
        import numpy as np
        line = preprocess_handwriting_image(self.sample('single'))[0]
        self.assertGreater(len(np.unique(np.asarray(line))), 20)

    def test_invalid_and_tiny_inputs_are_actionable(self):
        for image in [None, Image.new('RGB', (3, 3))]:
            with self.assertRaises(DocumentProcessingError):
                preprocess_handwriting_image(image)
