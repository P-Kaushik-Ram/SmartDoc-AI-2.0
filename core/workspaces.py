"""Independent, transient workspaces using the existing extraction services."""
import logging
import tempfile
from pathlib import Path

from django import forms
from django.core import signing
from django.views.decorators.cache import never_cache

from .resume_dashboard import analyze, build_report
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_http_methods

from .utils import (
    DocumentProcessingError, extract_document, PRINTED_OCR_LANGUAGES, validate_ocr_languages,
)
from .hybrid_ocr import extract_handwritten_document
from .gemini_ocr import extract_gemini_document, configuration_message

logger = logging.getLogger(__name__)


class UploadForm(forms.Form):
    document = forms.FileField(label='Document', widget=forms.ClearableFileInput(attrs={
        'accept': '.pdf,.docx,.png,.jpg,.jpeg', 'data-upload': '',
    }))

    def clean_document(self):
        upload = self.cleaned_data['document']
        if not 0 < upload.size <= 10 * 1024 * 1024:
            raise forms.ValidationError('Choose a nonempty file of 10 MB or smaller.')
        if Path(upload.name).suffix.lower() not in {'.pdf', '.docx', '.png', '.jpg', '.jpeg'}:
            raise forms.ValidationError('Upload a PDF, DOCX, PNG, JPG, or JPEG file.')
        return upload


class ResumeForm(UploadForm):
    job_description = forms.CharField(required=False, max_length=20000, label='Job description',
        widget=forms.Textarea(attrs={'rows': 8, 'placeholder': 'Paste the job description, including the skills required for the role…'}))

    def clean_document(self):
        if self.cleaned_data.get('document') is None:
            return None
        return super().clean_document()


class DocumentForm(UploadForm):
    engine = forms.ChoiceField(required=False, initial='hybrid', choices=[
        ('hybrid', 'TrOCR + Tesseract — English comparison (Offline)'),
        ('gemini', 'Gemini AI — Multilingual (Online)'),
        ('trocr', 'TrOCR Base — English (Offline)'),
        ('tesseract', 'Tesseract — Local Alternative'),
    ], widget=forms.Select(attrs={'data-ocr-engine': ''}))
    cloud_consent = forms.BooleanField(required=False, label='I choose cloud processing with Gemini.')
    languages = forms.MultipleChoiceField(
        choices=list(PRINTED_OCR_LANGUAGES.items()), required=False, initial=['eng'],
        widget=forms.CheckboxSelectMultiple, label='Document languages',
    )

    def clean_languages(self):
        return validate_ocr_languages(self.cleaned_data.get('languages'))

    mode = forms.ChoiceField(choices=[('printed', 'Printed Document OCR'), ('handwriting', 'Handwriting to Text')])

    def clean(self):
        data = super().clean()
        if data.get('mode') == 'handwriting':
            engine = data.get('engine') or 'hybrid'
            data['engine'] = engine
            if engine == 'gemini' and not data.get('cloud_consent'):
                self.add_error('cloud_consent', 'Explicit cloud-processing consent is required for Gemini.')
            if engine in {'trocr', 'hybrid'} and data.get('languages') != ['eng']:
                self.add_error('languages', 'TrOCR supports English only. Select English or choose Gemini/Tesseract.')
        upload = data.get('document')
        if data.get('mode') == 'handwriting' and upload and Path(upload.name).suffix.lower() == '.docx':
            self.add_error('document', 'Handwriting mode accepts PDF, PNG, JPG, and JPEG files.')
        return data


def process_upload(upload, handwriting=False, languages=None, engine='hybrid', cloud_consent=False):
    # Temporary files are removed on success and failure; no recruitment records
    # or persistent uploads are created by either workspace.
    with tempfile.TemporaryDirectory(prefix='smartdoc-') as directory:
        path = Path(directory) / ('document' + Path(upload.name).suffix.lower())
        with path.open('wb') as destination:
            for chunk in upload.chunks():
                destination.write(chunk)
        if handwriting and engine == 'gemini':
            result = extract_gemini_document(path, languages=languages, cloud_consent=cloud_consent)
        elif handwriting and engine != 'hybrid':
            result = extract_handwritten_document(path, engine=engine, languages=languages)
        else:
            processor = extract_handwritten_document if handwriting else extract_document
            result = processor(path, languages=languages) if not handwriting and languages else processor(path)
        if not result.text.strip() and not any(value.get('text', '').strip() for value in result.engine_results.values()):
            raise DocumentProcessingError('No readable text was extracted. Try a clearer document.')
        return result


def home(request):
    return render(request, 'studio/home.html', {'active': 'home'})


def about(request):
    return render(request, 'studio/about.html', {'active': 'about'})


@never_cache
@require_http_methods(['GET', 'POST'])
def resume_analyzer(request):
    form = ResumeForm(request.POST if request.method == 'POST' else None, request.FILES if request.method == 'POST' else None)
    token = request.POST.get('analysis_token', '')
    if token and not request.FILES:
        form.fields['document'].required = False
    context = {'form': form, 'active': 'resume'}
    if request.method == 'POST' and form.is_valid():
        try:
            if form.cleaned_data.get('document'):
                extraction = process_upload(form.cleaned_data['document'])
                if len(extraction.text) > 200000:
                    raise DocumentProcessingError('Resume text is too long. Upload a resume containing at most 200,000 characters.')
                payload = {'text': extraction.text, 'filename': form.cleaned_data['document'].name,
                           'method': extraction.method, 'warnings': extraction.warnings}
                token = signing.dumps(payload, salt='resume-dashboard-v1', compress=True)
            else:
                if len(token) > 1000000:
                    raise DocumentProcessingError('Analysis data is too large. Upload the resume again.')
                try:
                    payload = signing.loads(token, salt='resume-dashboard-v1', max_age=3600)
                except signing.BadSignature:
                    raise DocumentProcessingError('This analysis expired or is invalid. Upload the resume again.')
            result = analyze(**payload, job_description=form.cleaned_data['job_description'])
            context.update(result=result, report=build_report(result), analysis_token=token)
        except DocumentProcessingError as exc:
            form.add_error(None, str(exc))
        except Exception:
            # Exception messages from document parsers can contain private text.
            logger.error('Resume workspace processing failed')
            form.add_error(None, 'Processing failed unexpectedly. Try again with a smaller or clearer file.')
    return render(request, 'studio/resume.html', context)


@never_cache
@require_http_methods(['GET', 'POST'])
def document_studio(request):
    initial_mode = request.GET.get('mode', 'printed')
    if initial_mode not in {'printed', 'handwriting'}:
        initial_mode = 'printed'
    form = DocumentForm(request.POST if request.method == 'POST' else None, request.FILES if request.method == 'POST' else None, initial={'mode': initial_mode})
    context = {'form': form, 'active': 'documents', 'mode': initial_mode, 'gemini_message': configuration_message()}
    if request.method == 'POST':
        if form.is_valid():
            try:
                selected = form.cleaned_data['languages']
                result = process_upload(form.cleaned_data['document'],
                    handwriting=form.cleaned_data['mode'] == 'handwriting', languages=selected,
                    engine=form.cleaned_data['engine'] or 'hybrid', cloud_consent=form.cleaned_data['cloud_consent'])
                context.update(filename=form.cleaned_data['document'].name,
                    languages=selected, language_names=[PRINTED_OCR_LANGUAGES[code] for code in selected])
                engine_name = {'handwriting_gemini': 'Gemini AI', 'handwriting_trocr': 'TrOCR Base',
                    'handwriting_tesseract': 'Tesseract', 'handwriting_hybrid': 'TrOCR + Tesseract'}.get(result.method, result.method)
                context.update(text=result.text, warnings=result.warnings, method=result.method,
                    engine_name=engine_name, engine_results=result.engine_results)
            except DocumentProcessingError as exc:
                form.add_error('document', str(exc))
            except Exception:
                logger.error('Document workspace processing failed')
                form.add_error(None, 'Processing failed unexpectedly. Try a smaller or clearer file.')
        context['mode'] = request.POST.get('mode') if request.POST.get('mode') in {'printed', 'handwriting'} else 'printed'
        if request.headers.get('Accept') == 'application/json':
            if form.errors:
                return JsonResponse({'errors': form.errors.get_json_data()}, status=400)
            return JsonResponse({'text': context['text'], 'warnings': context['warnings'], 'method': context['method'], 'engine_results': context['engine_results'],
                'engine_name': context['engine_name'],
                'filename': context['filename'], 'languages': context['languages'], 'language_names': context['language_names']})
    if context['mode'] == 'handwriting':
        form.fields['document'].widget.attrs['accept'] = '.pdf,.png,.jpg,.jpeg'
    return render(request, 'studio/documents.html', context)
