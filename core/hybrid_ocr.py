"""Independent local handwriting engines; no automatic fusion or quality ranking."""
import logging
from pathlib import Path

from . import handwriting
from .utils import (
    DocumentProcessingError, ExtractionResult, MAX_DOCUMENT_PAGES, MAX_IMAGE_PIXELS,
    OCR_TIMEOUT_SECONDS, _open_pdf, _pdf_page, _render_pdf_page, _tesseract, _validate_file,
)

logger = logging.getLogger(__name__)
REVIEW_WARNING = 'Handwriting recognition may contain errors. Please verify the extracted text against the original document.'
TABLE_WARNING = 'Table formatting is not guaranteed. Verify row/column alignment and every value against the original.'


def _pages(path, suffix):
    """Decode/render each page once; each recognizer receives the same source."""
    from PIL import Image, ImageOps
    if suffix == '.pdf':
        try:
            with _open_pdf(path) as (document, page_count, _):
                if not 1 <= page_count <= MAX_DOCUMENT_PAGES:
                    raise DocumentProcessingError(f'PDF must contain between 1 and {MAX_DOCUMENT_PAGES} pages.')
                for index in range(page_count):
                    with _pdf_page(document, index) as page, _render_pdf_page(page) as image:
                        yield index + 1, image
        except DocumentProcessingError:
            raise
        except Exception as exc:
            raise DocumentProcessingError('The PDF is corrupted or could not be opened.') from exc
    else:
        try:
            with Image.open(path) as source:
                if source.format != ('PNG' if suffix == '.png' else 'JPEG'):
                    raise DocumentProcessingError('The image file content does not match its extension.')
                if source.width * source.height > MAX_IMAGE_PIXELS:
                    raise DocumentProcessingError('Image dimensions exceed the 40-megapixel processing limit.')
                image = ImageOps.exif_transpose(source).convert('RGBA')
            with image:
                yield 1, image
        except DocumentProcessingError:
            raise
        except Exception as exc:
            raise DocumentProcessingError('The image is invalid or could not be processed.') from exc


def recognize_tesseract_handwriting(image, languages=None):
    """Independent page OCR with its own grayscale/noise/layout preprocessing."""
    import numpy as np
    from PIL import Image, ImageFilter, ImageOps
    from scipy import ndimage

    gray = ImageOps.grayscale(image)
    if image.mode == 'RGBA':
        paper = Image.new('RGBA', image.size, 'white')
        gray = ImageOps.grayscale(Image.alpha_composite(paper, image))
    # Broad local illumination subtraction is used ONLY for blank screening.
    background = np.asarray(gray.filter(ImageFilter.GaussianBlur(max(2, min(25, max(gray.size) * .015)))), dtype=float)
    ink = background - np.asarray(gray, dtype=float) > 12
    ink, edges = handwriting._remove_edge_artifacts(ink)
    labels, _ = ndimage.label(ink)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    if int(sizes[sizes >= 3].sum()) < 30:
        return ''
    gray, ink, edges = handwriting._orient_page(gray, ink, edges)
    # Preserve stroke intensities; never binarize model/engine input.
    pixels = np.asarray(gray).copy()
    pixels[edges] = 255
    gray = ImageOps.autocontrast(Image.fromarray(pixels))
    gray = ImageOps.expand(gray, border=12, fill='white')
    from .utils import validate_ocr_languages
    selected = validate_ocr_languages(languages)
    tesseract = _tesseract(selected) if languages else _tesseract()
    try:
        return tesseract.image_to_string(gray, lang='+'.join(selected), config='--psm 6', timeout=OCR_TIMEOUT_SECONDS).strip()
    except Exception as exc:
        raise DocumentProcessingError('Tesseract handwriting OCR failed or timed out. Try a smaller or clearer page.') from exc


def extract_handwritten_document(file_path, engine='hybrid', languages=None):
    path = Path(str(file_path))
    suffix = _validate_file(path)
    if suffix == '.docx':
        raise DocumentProcessingError('Handwriting mode accepts PDF, PNG, JPG, and JPEG files.')
    outputs = {key: {'text': '', 'error': '', 'warnings': []} for key in ('trocr', 'tesseract')}
    texts = {key: [] for key in outputs}
    count = 0
    engines = {'trocr': handwriting.recognize_handwriting_image, 'tesseract': recognize_tesseract_handwriting}
    if engine not in {'hybrid', 'trocr', 'tesseract'}:
        raise DocumentProcessingError('Choose a supported local handwriting engine.')
    if engine != 'hybrid':
        engines = {engine: engines[engine]}
        outputs = {engine: outputs[engine]}
    for number, image in _pages(path, suffix):
        count += 1
        for key, recognize in engines.items():
            try:
                text = (recognize(image, languages=languages) if key == 'tesseract' and languages else recognize(image)).strip()
                if text:
                    texts[key].append(f'[Page {number}]\n{text}')
                else:
                    outputs[key]['warnings'].append(f'Page {number}: no text recognized.')
            except DocumentProcessingError as exc:
                outputs[key]['warnings'].append(f'Page {number}: {exc}')
            except Exception:
                logger.exception('%s handwriting engine failed on page %s', key, number)
                outputs[key]['warnings'].append(f'Page {number}: recognition failed unexpectedly.')
    for key, output in outputs.items():
        output['text'] = '\n\n'.join(texts[key])
        if not output['text']:
            output['error'] = ' '.join(output['warnings']) or 'No text recognized.'
    if not any(output['text'] for output in outputs.values()):
        details = ' '.join(f'{key}: {value["error"]}' for key, value in outputs.items())
        raise DocumentProcessingError(f'No readable handwriting was extracted. {details}')
    warnings = [REVIEW_WARNING, TABLE_WARNING]
    for key, output in outputs.items():
        warnings.extend(f'{key.upper()}: {message}' for message in output['warnings'])
    # Start with an empty final editor. The user explicitly chooses an engine;
    # this is not a score, confidence estimate, or claim one engine is better.
    if engine != 'hybrid':
        return ExtractionResult(outputs[engine]['text'], 'handwriting_' + engine, warnings, count)
    return ExtractionResult('', 'handwriting_hybrid', warnings, count, outputs)
