"""Bounded document extraction and explainable baseline matching services."""
from dataclasses import dataclass, field
import math
import os
import re
import shutil
from pathlib import Path
import zipfile
from contextlib import contextmanager
from threading import Lock

import pypdfium2 as pdfium
from docx import Document

MAX_DOCUMENT_BYTES = 10 * 1024 * 1024
MAX_DOCUMENT_PAGES = 50
MAX_IMAGE_PIXELS = 40_000_000
MAX_RENDER_PIXELS = 16_000_000
MAX_RENDER_EDGE = 4_000
OCR_TIMEOUT_SECONDS = 30
WINDOWS_TESSERACT_PATH = Path(r'C:\Program Files\Tesseract-OCR\tesseract.exe')


class DocumentProcessingError(ValueError):
    """A user-actionable validation, extraction, or OCR failure."""


@dataclass
class ExtractionResult:
    text: str
    method: str
    warnings: list[str]
    page_count: int | None
    engine_results: dict = field(default_factory=dict)

    # Preserve the former three-value unpacking API while exposing metadata.
    def __iter__(self):
        yield self.text
        yield self.method
        yield self.warnings

    def __getitem__(self, index):
        return (self.text, self.method, self.warnings)[index]

    def __len__(self):
        return 3


COMMON_SKILLS = [
    'python', 'java', 'javascript', 'typescript', 'c', 'c++', 'c#', 'ruby', 'php', 'swift',
    'kotlin', 'go', 'rust', 'scala', 'r', 'matlab', 'perl', 'bash', 'shell',
    'html', 'css', 'react', 'angular', 'vue', 'node', 'nodejs', 'express', 'django',
    'flask', 'fastapi', 'spring', 'laravel', 'rails', 'bootstrap', 'jquery', 'sass',
    'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'sqlite', 'oracle', 'elasticsearch',
    'cassandra', 'firebase', 'dynamodb', 'aws', 'azure', 'gcp', 'docker', 'kubernetes',
    'jenkins', 'git', 'github', 'gitlab', 'ci/cd', 'terraform', 'ansible', 'linux',
    'nginx', 'apache', 'machine learning', 'deep learning', 'nlp', 'tensorflow',
    'pytorch', 'keras', 'scikit-learn', 'pandas', 'numpy', 'matplotlib', 'data analysis',
    'data science', 'power bi', 'tableau', 'spark', 'hadoop', 'rest api', 'graphql',
    'microservices', 'agile', 'scrum', 'jira', 'excel', 'word', 'photoshop', 'figma',
    'postman', 'selenium', 'junit', 'pytest',
]


def _validate_file(path: Path) -> str:
    if not path.exists() or not path.is_file():
        raise DocumentProcessingError('The uploaded document is unavailable.')
    size = path.stat().st_size
    if size <= 0:
        raise DocumentProcessingError('The uploaded document is empty.')
    if size > MAX_DOCUMENT_BYTES:
        raise DocumentProcessingError('Files must be 10 MB or smaller.')

    suffix = path.suffix.lower()
    allowed = {'.pdf', '.docx', '.jpg', '.jpeg', '.png'}
    if suffix not in allowed:
        raise DocumentProcessingError('Unsupported document type. Upload PDF, DOCX, JPG, JPEG, or PNG.')
    with path.open('rb') as source:
        header = source.read(16)
    signatures = {
        '.pdf': header.startswith(b'%PDF-'),
        '.docx': header.startswith(b'PK\x03\x04'),
        '.jpg': header.startswith(b'\xff\xd8\xff'),
        '.jpeg': header.startswith(b'\xff\xd8\xff'),
        '.png': header.startswith(b'\x89PNG\r\n\x1a\n'),
    }
    if not signatures[suffix]:
        raise DocumentProcessingError('The file content does not match its extension.')
    return suffix


PRINTED_OCR_LANGUAGES = {
    'eng': 'English', 'tam': 'Tamil', 'hin': 'Hindi',
    'tel': 'Telugu', 'mal': 'Malayalam',
}
NO_READABLE_TEXT = 'No readable text was detected. Try a clearer document or verify the selected languages.'


def validate_ocr_languages(languages=None):
    """Accept only individual allowlisted codes, preserving order and removing duplicates."""
    if languages is None or languages == [] or languages == ():
        return ['eng']
    if not isinstance(languages, (list, tuple)) or any(
        not isinstance(code, str) or code not in PRINTED_OCR_LANGUAGES for code in languages
    ):
        raise DocumentProcessingError('Select valid OCR languages: English, Tamil, Hindi, Telugu, or Malayalam.')
    return list(dict.fromkeys(languages))


# Configure the offline fallback once, before workers start. Avoid putting quoted
# Windows paths in pytesseract's config string (its Windows shlex retains quotes).
LOCAL_TESSDATA = Path(__file__).resolve().parent.parent / 'tessdata'
if LOCAL_TESSDATA.is_dir():
    os.environ.setdefault('TESSDATA_PREFIX', str(LOCAL_TESSDATA))


def _tessdata_directory():
    directory = os.environ.get('TESSDATA_PREFIX', '').strip()
    if directory and not Path(directory).is_dir():
        raise DocumentProcessingError('The configured TESSDATA_PREFIX directory is invalid.')
    return Path(directory) if directory else None


def _tessdata_config():
    # Tesseract inherits the administrator-controlled TESSDATA_PREFIX, including
    # paths with spaces. No user-controlled options are ever constructed.
    _tessdata_directory()
    return ''


def _tesseract(languages=None):
    """Return pytesseract after checking the binary and selected trained data."""
    selected = validate_ocr_languages(languages)
    try:
        import pytesseract
    except ImportError as exc:
        raise DocumentProcessingError(
            'OCR support is unavailable because pytesseract is not installed. Install the '
            'project requirements, then install the Tesseract OCR application and English (eng) data.'
        ) from exc

    configured_path = os.environ.get('TESSERACT_CMD', '').strip()
    if configured_path:
        if not Path(configured_path).is_file():
            raise DocumentProcessingError(
                'The Tesseract executable configured by TESSERACT_CMD was not found. '
                'Check the path and retry.'
            )
        pytesseract.pytesseract.tesseract_cmd = configured_path
    else:
        detected = shutil.which('tesseract') or shutil.which('tesseract.exe')
        candidates = [WINDOWS_TESSERACT_PATH,
                      Path(os.environ.get('ProgramFiles(x86)', 'C:/Program Files (x86)')) / 'Tesseract-OCR/tesseract.exe',
                      Path(os.environ.get('LOCALAPPDATA', '')) / 'Programs/Tesseract-OCR/tesseract.exe']
        if not detected and os.name == 'nt':
            detected = next((str(path) for path in candidates if path.is_file()), None)
        if detected:
            pytesseract.pytesseract.tesseract_cmd = detected
        else:
            raise DocumentProcessingError(
                'Tesseract OCR is not installed or is not on PATH. Install Tesseract and its English '
                'language data, restart the application, then retry processing.'
            )

    try:
        pytesseract.get_tesseract_version()
    except pytesseract.TesseractNotFoundError as exc:
        raise DocumentProcessingError(
            'Tesseract OCR is not installed or is not on PATH. Install Tesseract and its English '
            'language data, restart the application, then retry processing.'
        ) from exc
    except Exception as exc:
        raise DocumentProcessingError(
            'Tesseract OCR could not be started. Check the installation and PATH, then retry.'
        ) from exc

    try:
        languages = pytesseract.get_languages(config=_tessdata_config())
    except Exception as exc:
        raise DocumentProcessingError(
            'Tesseract is installed, but its language data could not be read. Check the active '
            'tessdata directory and the selected traineddata files, then retry.'
        ) from exc
    directory = _tessdata_directory()
    missing = [code for code in selected if code not in languages or
               (directory is not None and not (directory / f'{code}.traineddata').is_file())]
    if missing:
        names = ', '.join(f'{PRINTED_OCR_LANGUAGES[code]} ({code})' for code in missing)
        raise DocumentProcessingError(
            f'{names} language data is missing. Install the matching .traineddata '
            'files in the active tessdata directory and retry. See docs/P6_MULTILINGUAL_OCR_REPORT.md.'
        )
    return pytesseract


def _preprocess_image(image):
    from PIL import ImageEnhance, ImageOps

    if image.width * image.height > MAX_IMAGE_PIXELS:
        raise DocumentProcessingError('Image dimensions exceed the 40-megapixel processing limit.')
    image = ImageOps.grayscale(image)
    image = ImageOps.autocontrast(image, cutoff=1)
    image = ImageEnhance.Contrast(image).enhance(1.25)
    image = ImageEnhance.Sharpness(image).enhance(1.15)
    return image


def _correct_orientation(image, pytesseract):
    """Use Tesseract OSD as a best-effort rotation hint; OCR continues if OSD is unavailable."""
    try:
        osd = pytesseract.image_to_osd(image, config=' '.join(filter(None, ['--psm 0', _tessdata_config()])), timeout=8)
        rotation = int(re.search(r'Rotate:\s*(\d+)', osd).group(1)) % 360
        if rotation in {90, 180, 270}:
            return image.rotate(rotation, expand=True)
    except Exception:
        # OSD can fail on short text or installations without osd.traineddata.
        pass
    return image


def _ocr_image(image, languages=None):
    """OCR a PIL image, with explicit binary/language checks and a hard timeout."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise DocumentProcessingError('Image OCR requires Pillow. Install the project requirements and retry.') from exc

    try:
        if image.width * image.height > MAX_IMAGE_PIXELS:
            raise DocumentProcessingError('Image dimensions exceed the 40-megapixel processing limit.')
        selected = validate_ocr_languages(languages)
        pytesseract = _tesseract() if languages is None else _tesseract(selected)
        image = _preprocess_image(image)
        image = _correct_orientation(image, pytesseract)
        text = pytesseract.image_to_string(
            image, lang='+'.join(selected),
            config=' '.join(filter(None, ['--psm 3', _tessdata_config()])), timeout=OCR_TIMEOUT_SECONDS,
        ).strip()
        return text
    except DocumentProcessingError:
        raise
    except Exception as exc:
        error_type = type(exc).__name__
        if error_type == 'TesseractNotFoundError':
            raise DocumentProcessingError(
                'Tesseract OCR is not installed or is not on PATH. Install Tesseract and English (eng) data, then retry.'
            ) from exc
        if error_type == 'TimeoutExpired' or 'timeout' in str(exc).lower():
            raise DocumentProcessingError(
                f'OCR exceeded the {OCR_TIMEOUT_SECONDS}-second page limit. Try a clearer or smaller image.'
            ) from exc
        if error_type == 'TesseractError':
            detail = str(exc).lower()
            if 'eng' in detail or 'traineddata' in detail or 'language' in detail:
                raise DocumentProcessingError(
                    'Tesseract could not load the selected language data (' + '+'.join(selected) + '). Check the traineddata files and retry.'
                ) from exc
            raise DocumentProcessingError('Tesseract failed while reading this page. Check the OCR installation and retry.') from exc
        raise DocumentProcessingError('OCR could not process this image. Confirm it is readable and retry.') from exc


def _has_reliable_digital_text(text: str) -> bool:
    words = re.findall(r"[^\W_][\w+#./'-]*", text)
    alphanumeric_count = sum(char.isalnum() for char in text)
    return alphanumeric_count >= 24 and len(words) >= 3


# PDFium requires serialization across documents as well as across pages.
_PDFIUM_LOCK = Lock()


@contextmanager
def _open_pdf(path, *, protected_message='Password-protected PDFs are not supported.'):
    with _PDFIUM_LOCK:
        try:
            document = pdfium.PdfDocument(path)
        except pdfium.PdfiumError as exc:
            if exc.err_code == pdfium.raw.FPDF_ERR_PASSWORD:
                raise DocumentProcessingError(protected_message) from None
            raise
    try:
        with _PDFIUM_LOCK:
            if pdfium.raw.FPDF_GetSecurityHandlerRevision(document) != -1:
                raise DocumentProcessingError(protected_message)
            count = len(document)
            repaired = not pdfium.raw.FPDF_DocumentHasValidCrossReferenceTable(document)
        yield document, count, repaired
    finally:
        with _PDFIUM_LOCK:
            document.close()


@contextmanager
def _pdf_page(document, index):
    with _PDFIUM_LOCK:
        page = document[index]
    try:
        yield page
    finally:
        with _PDFIUM_LOCK:
            page.close()


def _pdf_page_text(page):
    with _PDFIUM_LOCK:
        text_page = page.get_textpage()
        try:
            return text_page.get_text_bounded().replace('\r\n', '\n').strip()
        finally:
            text_page.close()


def _render_pdf_page(page):
    with _PDFIUM_LOCK:
        width, height = page.get_size()
        if not all(math.isfinite(edge) and edge > 0 for edge in (width, height)):
            raise DocumentProcessingError('This PDF page has invalid dimensions.')
        # Leave one pixel of headroom for PDFium's rounding.
        scale = min(
            300 / 72,
            (MAX_RENDER_EDGE - 1) / max(width, height),
            (math.sqrt(MAX_RENDER_PIXELS) - 1) / math.sqrt(width * height),
        )
        bitmap = page.render(scale=scale, draw_annots=True)
        try:
            # Convert while the native bitmap is still alive. RGB conversion owns its buffer.
            return bitmap.to_pil().convert('RGB')
        finally:
            bitmap.close()


def _extract_pdf(path: Path, languages=None) -> ExtractionResult:
    page_chunks = []
    warnings = []
    digital_pages = 0
    ocr_pages = 0
    try:
        with _open_pdf(path) as (doc, page_count, repaired):
            if page_count == 0:
                raise DocumentProcessingError('The PDF contains no pages.')
            if page_count > MAX_DOCUMENT_PAGES:
                raise DocumentProcessingError(f'PDF exceeds the {MAX_DOCUMENT_PAGES}-page processing limit.')
            if repaired:
                warnings.append('The PDF required repair while opening; check the extracted text against the original.')

            for page_num in range(1, page_count + 1):
                with _pdf_page(doc, page_num - 1) as page:
                    try:
                        digital_text = _pdf_page_text(page)
                    except Exception as exc:
                        warnings.append(f'Page {page_num}: digital text extraction failed.')
                        digital_text = ''

                    if _has_reliable_digital_text(digital_text):
                        page_chunks.append(f'[Page {page_num}]\n{digital_text}')
                        digital_pages += 1
                        continue

                    # OCR only pages without enough selectable text. Keep any short digital text
                    # if OCR is unavailable, and clearly mark the unread page for human review.
                    try:
                        with _render_pdf_page(page) as image:
                            ocr_text = _ocr_image(image) if languages is None else _ocr_image(image, languages)
                        if ocr_text:
                            combined = '\n'.join(part for part in (digital_text, ocr_text) if part)
                            page_chunks.append(f'[Page {page_num}]\n{combined}')
                            ocr_pages += 1
                        elif digital_text:
                            page_chunks.append(f'[Page {page_num}]\n{digital_text}')
                            digital_pages += 1
                            warnings.append(f'Page {page_num}: OCR returned no text; retained the limited selectable text.')
                        else:
                            page_chunks.append(f'[Page {page_num}: no readable text extracted]')
                            warnings.append(f'Page {page_num}: {NO_READABLE_TEXT}')
                    except DocumentProcessingError as exc:
                        if digital_text:
                            page_chunks.append(f'[Page {page_num}]\n{digital_text}')
                            digital_pages += 1
                            warnings.append(f'Page {page_num}: OCR was not completed; retained limited selectable text. {exc}')
                        else:
                            page_chunks.append(f'[Page {page_num}: OCR unavailable]')
                            warnings.append(f'Page {page_num}: {exc}')

    except DocumentProcessingError:
        raise
    except Exception as exc:
        raise DocumentProcessingError('The PDF is corrupted or could not be opened.') from exc

    text = '\n\n'.join(page_chunks).strip()
    if not any(char.isalnum() for char in re.sub(r'\[Page[^\]]*\]', '', text)):
        if warnings:
            raise DocumentProcessingError(warnings[0])
        raise DocumentProcessingError(NO_READABLE_TEXT)
    if ocr_pages and digital_pages:
        method = 'pdf_text_ocr'
    elif ocr_pages:
        method = 'pdf_ocr'
    else:
        method = 'pdf_text'
    return ExtractionResult(text, method, warnings, page_count)


def _validate_docx_archive(path: Path):
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            if len(members) > 2_000:
                raise DocumentProcessingError('The DOCX archive contains too many embedded items.')
            uncompressed_size = sum(member.file_size for member in members)
            if uncompressed_size > 50 * 1024 * 1024:
                raise DocumentProcessingError('The DOCX expands beyond the 50 MB processing limit.')
            if any(member.file_size > 1_000_000 and member.compress_size == 0 for member in members):
                raise DocumentProcessingError('The DOCX archive contains an invalid compressed item.')
            if any(member.file_size > max(1_000_000, member.compress_size * 200) for member in members):
                raise DocumentProcessingError('The DOCX archive has an unsafe compression ratio.')
            if 'word/document.xml' not in archive.namelist():
                raise DocumentProcessingError('The DOCX file is not a valid Word document.')
            corrupt_member = archive.testzip()
            if corrupt_member:
                raise DocumentProcessingError('The DOCX archive is corrupted.')
    except DocumentProcessingError:
        raise
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        raise DocumentProcessingError('The DOCX file is corrupted or could not be opened.') from exc


def _extract_docx(path: Path) -> ExtractionResult:
    _validate_docx_archive(path)
    try:
        document = Document(path)
        chunks = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]
        for table in document.tables:
            for row in table.rows:
                values = [cell.text.strip() for cell in row.cells]
                if any(values):
                    chunks.append(' | '.join(values))
        text = '\n'.join(chunks).strip()
    except Exception as exc:
        raise DocumentProcessingError('The DOCX file is corrupted or could not be opened.') from exc
    if not text:
        raise DocumentProcessingError('No readable text was found in the DOCX document.')
    return ExtractionResult(text, 'docx_text', [], None)


def _extract_image(path: Path, suffix: str, languages=None) -> ExtractionResult:
    try:
        from PIL import Image, ImageOps, UnidentifiedImageError
        with Image.open(path) as source:
            expected_format = 'PNG' if suffix == '.png' else 'JPEG'
            if source.format != expected_format:
                raise DocumentProcessingError('The image file content does not match its extension.')
            if source.width * source.height > MAX_IMAGE_PIXELS:
                raise DocumentProcessingError('Image dimensions exceed the 40-megapixel processing limit.')
            source.verify()
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert('RGB')
        text = _ocr_image(image) if languages is None else _ocr_image(image, languages)
    except DocumentProcessingError:
        raise
    except ImportError as exc:
        raise DocumentProcessingError('Image processing requires Pillow. Install the project requirements and retry.') from exc
    except Exception as exc:
        if type(exc).__name__ == 'DecompressionBombError':
            raise DocumentProcessingError('Image dimensions exceed the safe processing limit.') from exc
        raise DocumentProcessingError('The image is invalid or could not be processed.') from exc
    if not text:
        raise DocumentProcessingError(NO_READABLE_TEXT)
    return ExtractionResult(f'[Page 1]\n{text}', 'tesseract_ocr', [], 1)


def extract_document(file_path, languages=None) -> ExtractionResult:
    """Extract text with page boundaries, extraction method, warnings, and page count."""
    path = Path(str(file_path))
    suffix = _validate_file(path)
    if languages is not None:
        languages = validate_ocr_languages(languages)
        # Fail before PDF fallback can disguise a missing selected language pack.
        if suffix != '.docx':
            _tesseract(languages)
    if suffix == '.pdf':
        return _extract_pdf(path) if languages is None else _extract_pdf(path, languages)
    if suffix == '.docx':
        return _extract_docx(path)
    return _extract_image(path, suffix) if languages is None else _extract_image(path, suffix, languages)


def extract_handwritten_document(file_path) -> ExtractionResult:
    """Recognize handwritten English in image files or rendered PDF pages."""
    path = Path(str(file_path))
    suffix = _validate_file(path)
    if suffix == '.docx':
        raise DocumentProcessingError('Handwriting recognition supports PDF, JPG, JPEG and PNG files. DOCX is not supported in handwriting mode.')

    from .handwriting import recognize_handwriting_image

    page_text = []
    if suffix == '.pdf':
        try:
            with _open_pdf(path) as (doc, page_count, _):
                if not 1 <= page_count <= MAX_DOCUMENT_PAGES:
                    raise DocumentProcessingError(f'PDF must contain between 1 and {MAX_DOCUMENT_PAGES} pages.')
                for page_num in range(1, page_count + 1):
                    with _pdf_page(doc, page_num - 1) as page:
                        try:
                            with _render_pdf_page(page) as image:
                                text = recognize_handwriting_image(image)
                        except DocumentProcessingError as exc:
                            raise DocumentProcessingError(f'Handwriting recognition failed on page {page_num}: {exc}') from exc
                        if not text.strip():
                            raise DocumentProcessingError(f'No handwritten text was recognized on page {page_num}. Review the original or use standard extraction.')
                        page_text.append(f'[Page {page_num}]\n{text.strip()}')
        except DocumentProcessingError:
            raise
        except Exception as exc:
            raise DocumentProcessingError('The PDF is corrupted or could not be opened.') from exc
    else:
        from PIL import Image, ImageOps
        try:
            with Image.open(path) as source:
                expected_format = 'PNG' if suffix == '.png' else 'JPEG'
                if source.format != expected_format:
                    raise DocumentProcessingError('The image file content does not match its extension.')
                if source.width * source.height > MAX_IMAGE_PIXELS:
                    raise DocumentProcessingError('Image dimensions exceed the 40-megapixel processing limit.')
                source.verify()
            with Image.open(path) as source:
                image = ImageOps.exif_transpose(source).convert('RGBA' if source.mode in {'RGBA', 'LA', 'P'} else 'RGB')
            with image:
                text = recognize_handwriting_image(image)
            if not text.strip():
                raise DocumentProcessingError('No handwritten text was recognized on this image. Review the original or use standard extraction.')
            page_text.append(f'[Page 1]\n{text.strip()}')
        except DocumentProcessingError:
            raise
        except Exception as exc:
            raise DocumentProcessingError('The image is invalid or could not be processed.') from exc

    return ExtractionResult('\n\n'.join(page_text), 'handwriting_trocr', ['Handwriting recognition may contain errors. Please verify the extracted text against the original document.'], len(page_text))


def extract_text_from_pdf(file_path):
    return extract_document(file_path).text


def extract_text_from_docx(file_path):
    return extract_document(file_path).text


def extract_text_from_resume(file_path):
    return extract_document(file_path).text


def extract_skills_from_text(text):
    text_lower = (text or '').casefold()
    found_skills = []
    for skill in COMMON_SKILLS:
        # Treat +/# as part of a skill token: C must not match C++ or C#.
        # Unlike \b this also handles a skill ending in a non-word character.
        pattern = r'(?<![\w+#])' + re.escape(skill) + r'(?![\w+#])'
        if re.search(pattern, text_lower):
            found_skills.append(skill)
    return sorted(set(found_skills))


def compute_match(candidate_skills, job_skills, precision=1):
    """Explainable required-skill coverage baseline; no arbitrary bonus."""
    normalize = lambda items: {str(item).strip().casefold() for item in items if str(item).strip()}
    candidate_set = normalize(candidate_skills)
    job_set = normalize(job_skills)
    if not job_set:
        return 0.0, [], []
    matching = sorted(candidate_set & job_set)
    missing = sorted(job_set - candidate_set)
    score = (len(matching) / len(job_set)) * 100
    return round(score, precision), matching, missing


def get_recommendation(match_percentage):
    if match_percentage >= 70:
        return 'highly_recommended'
    elif match_percentage >= 40:
        return 'recommended'
    return 'not_suitable'
