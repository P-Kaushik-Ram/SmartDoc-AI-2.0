import os
from datetime import datetime, time, timedelta
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.db.models import Q, Count
from django.utils import timezone
from django.utils.dateparse import parse_date
from django.http import JsonResponse, HttpResponse, FileResponse, Http404
from django.views.decorators.http import require_POST
from pathlib import Path
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment

from .models import JobDescription, Candidate, MatchResult
from .forms import LoginForm, JobDescriptionForm, CandidateForm, MatchForm, StructuredResumeProfileForm
from .utils import (
    DocumentProcessingError, extract_document, extract_handwritten_document,
    extract_skills_from_text, compute_match, get_recommendation,
)
from .structured_extraction import (
    FIELD_LABELS, SCALAR_FIELDS, SECTION_FIELDS, apply_profile_corrections,
    extract_structured_profile, mark_profile_from_reviewed_text, profile_to_form_initial,
)
from .semantic_matching import (
    SemanticMatchingError, SemanticModelUnavailable, compute_hybrid_match,
    validate_candidate_for_matching,
)
from .analytics import (
    document_method_counts, paired_match_averages, score_distribution, skill_frequencies,
)
from .document_assistant import (
    DocumentNotReady, MAX_QUESTION_CHARS, build_evidence_messages,
    is_comparison_question, retrieve_evidence, validate_generated_answer,
)
from .assistant_providers import AssistantProviderError, get_answer_provider
import hashlib
import re


def process_candidate_document(candidate):
    """Process one stored upload and persist its result without affecting other candidates."""
    candidate.extraction_status = 'pending'
    candidate.extraction_error = ''
    candidate.save(update_fields=['extraction_status', 'extraction_error'])
    try:
        if candidate.processing_mode == Candidate.PROCESSING_HANDWRITING:
            result = extract_handwritten_document(candidate.resume.path)
        else:
            result = extract_document(candidate.resume.path)
        if not result.text.strip():
            raise DocumentProcessingError('No readable text was extracted from this document.')
    except Exception as exc:
        candidate.extraction_status = 'failed'
        candidate.extraction_method = ''
        # Known processing messages are written for recruiters; unexpected
        # library/OS exceptions may contain server paths or other internals.
        candidate.extraction_error = (
            str(exc)[:500] if isinstance(exc, DocumentProcessingError)
            else 'Document processing failed unexpectedly. Retry processing or enter the text manually.'
        )
        candidate.extraction_warnings = ''
        candidate.extraction_page_count = None
        candidate.extraction_processed_at = timezone.now()
        candidate.save(update_fields=[
            'extraction_status', 'extraction_method', 'extraction_error',
            'extraction_warnings', 'extraction_page_count', 'extraction_processed_at',
        ])
        return False

    warnings = list(result.warnings)
    readable_chars = sum(character.isalnum() for character in result.text)
    if readable_chars < 25 and not warnings:
        warnings.append('Only a small amount of readable text was extracted; review the original document.')
    if candidate.processing_mode == Candidate.PROCESSING_HANDWRITING:
        structured = {}
    else:
        try:
            structured = extract_structured_profile(result.text)
        except ValueError as exc:
            structured = {'schema_version': 1, 'error': str(exc)}
            warnings.append(f'Structured resume extraction requires review: {exc}')
    candidate.extracted_text = result.text
    # A retry may refresh the original extraction while retaining a recruiter's
    # saved correction. Keep the skill list aligned with the effective text.
    effective_text = candidate.reviewed_text or result.text
    # Model output is always review-required. Do not use uncertain recognition
    # to infer skills until a person saves a reviewed correction.
    candidate.extracted_skills = (
        '' if candidate.processing_mode == Candidate.PROCESSING_HANDWRITING
        else ', '.join(extract_skills_from_text(effective_text))
    )
    candidate.extraction_method = result.method
    candidate.extraction_error = ''
    candidate.extraction_warnings = '\n'.join(warnings)[:4000]
    candidate.extraction_page_count = result.page_count
    candidate.extraction_processed_at = timezone.now()
    candidate.structured_extraction = structured
    candidate.extraction_status = 'needs_review' if warnings or candidate.processing_mode == Candidate.PROCESSING_HANDWRITING else 'completed'
    candidate.save(update_fields=[
        'extracted_text', 'extracted_skills', 'extraction_method', 'extraction_error',
        'extraction_warnings', 'extraction_page_count', 'extraction_processed_at', 'extraction_status',
        'structured_extraction',
    ])
    return True


def login_view(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
    form = LoginForm(request, data=request.POST or None)
    if request.method == 'POST' and form.is_valid():
        user = form.get_user()
        login(request, user)
        return redirect('dashboard')
    return render(request, 'registration/login.html', {'form': form})


@require_POST
def logout_view(request):
    logout(request)
    return redirect('login')


@login_required
def dashboard(request):
    status_choices = dict(Candidate._meta.get_field('extraction_status').choices)
    job_id = request.GET.get('job', '').strip()
    status_filter = request.GET.get('status', '').strip()
    method_filter = request.GET.get('method', '').strip()
    from_raw = request.GET.get('date_from', '').strip()
    to_raw = request.GET.get('date_to', '').strip()
    selected_job = JobDescription.objects.filter(pk=job_id).first() if job_id.isdigit() else None
    if job_id and not selected_job:
        messages.warning(request, 'That job filter is unavailable; showing all jobs.')
        job_id = ''
    if status_filter not in status_choices:
        status_filter = ''
    if method_filter not in {'', 'baseline', 'hybrid'}:
        method_filter = ''

    date_from = parse_date(from_raw) if from_raw else None
    date_to = parse_date(to_raw) if to_raw else None
    if from_raw and not date_from:
        messages.warning(request, 'Enter a valid start date.')
        from_raw = ''
    if to_raw and not date_to:
        messages.warning(request, 'Enter a valid end date.')
        to_raw = ''
    if date_from and date_to and date_from > date_to:
        messages.warning(request, 'Start date must be on or before end date; date filters were cleared.')
        date_from = date_to = None
        from_raw = to_raw = ''

    def date_window(field, queryset):
        start = datetime.combine(date_from, time.min) if date_from else None
        end = datetime.combine(date_to + timedelta(days=1), time.min) if date_to else None
        if start:
            queryset = queryset.filter(**{f'{field}__gte': timezone.make_aware(start)})
        if end:
            queryset = queryset.filter(**{f'{field}__lt': timezone.make_aware(end)})
        return queryset

    match_scope = MatchResult.objects.all()
    if job_id:
        match_scope = match_scope.filter(job_id=job_id)
    if method_filter == 'hybrid':
        match_scope = match_scope.filter(explanation__method='hybrid')
    elif method_filter == 'baseline':
        # Empty explanations are historic skill-coverage results from before Stage 6.
        match_scope = match_scope.filter(
            Q(explanation__method='baseline') | Q(explanation={}) | Q(explanation__method__isnull=True)
        )
    match_scope = date_window('matched_at', match_scope)
    if status_filter:
        match_scope = match_scope.filter(candidate__extraction_status=status_filter)

    candidate_scope = Candidate.objects.all()
    if date_from:
        candidate_scope = candidate_scope.filter(uploaded_at__gte=timezone.make_aware(datetime.combine(date_from, time.min)))
    if date_to:
        candidate_scope = candidate_scope.filter(uploaded_at__lt=timezone.make_aware(datetime.combine(date_to + timedelta(days=1), time.min)))
    if status_filter:
        candidate_scope = candidate_scope.filter(extraction_status=status_filter)
    if job_id or method_filter:
        candidate_scope = candidate_scope.filter(pk__in=match_scope.values('candidate_id'))

    status_counts = candidate_scope.aggregate(
        processed=Count('pk', filter=Q(extraction_status__in=['completed', 'failed', 'needs_review'])),
        successful=Count('pk', filter=Q(extraction_status='completed')),
        failed=Count('pk', filter=Q(extraction_status='failed')),
        review=Count('pk', filter=Q(extraction_status='needs_review')),
        pending=Count('pk', filter=Q(extraction_status='pending')),
    )

    valid_matches = match_scope.filter(
        candidate__extraction_status='completed',
        match_percentage__gte=0, match_percentage__lte=100,
    ).filter(Q(candidate__reviewed_text__gt='') | Q(candidate__extracted_text__gt=''))
    valid_matches = valid_matches.exclude(
        Q(candidate__processing_mode=Candidate.PROCESSING_HANDWRITING) & Q(candidate__reviewed_text='')
    ).select_related('candidate', 'job')
    match_count = valid_matches.count()
    valid_candidate_count = valid_matches.values('candidate_id').distinct().count()
    top_matches = valid_matches.order_by('-match_percentage')[:5]
    distribution = score_distribution(valid_matches.values_list('match_percentage', flat=True))

    # Load only fields used by skill aggregation; resume text and file blobs stay out of memory.
    skill_candidates = candidate_scope.only(
        'id', 'processing_mode', 'reviewed_text', 'reviewed_profile', 'extracted_skills', 'extraction_status',
    )
    skill_stats = skill_frequencies(skill_candidates)
    method_rows = candidate_scope.filter(
        extraction_status__in=['completed', 'needs_review', 'failed'],
    ).values('extraction_method').annotate(count=Count('pk'))
    method_stats = document_method_counts(
        (row['extraction_method'], row['count']) for row in method_rows
    )

    comparable_matches = valid_matches.filter(explanation__method='hybrid')
    comparison = paired_match_averages(comparable_matches.values_list(
        'explanation__baseline_score', 'match_percentage',
    )) if method_filter != 'baseline' else {'count': 0, 'baseline_average': None, 'hybrid_average': None}

    attention_candidates = candidate_scope.filter(extraction_status__in=['needs_review', 'pending']).only(
        'id', 'name', 'email', 'extraction_status', 'processing_mode', 'uploaded_at',
    ).order_by('-uploaded_at')[:8]
    jobs = JobDescription.objects.only('id', 'title').order_by('title')
    recent_candidates = candidate_scope.only('id', 'name', 'email', 'experience', 'extraction_status', 'uploaded_at').order_by('-uploaded_at')[:5]
    recent_jobs = JobDescription.objects.only('id', 'title', 'experience', 'created_date', 'required_skills').order_by('-created_date')[:5]

    context = {
        'total_candidates': candidate_scope.count(),
        'total_jobs': JobDescription.objects.count(),
        'total_matches': match_count,
        'valid_candidate_count': valid_candidate_count,
        'documents_processed': status_counts['processed'],
        'successful_extractions': status_counts['successful'],
        'failed_extractions': status_counts['failed'],
        'documents_needing_review': status_counts['review'],
        'pending_documents': status_counts['pending'],
        'top_matches': top_matches,
        'recent_candidates': recent_candidates,
        'recent_jobs': recent_jobs,
        'attention_candidates': attention_candidates,
        'jobs': jobs,
        'selected_job': selected_job,
        'job_id': job_id,
        'status_filter': status_filter,
        'status_choices': status_choices.items(),
        'method_filter': method_filter,
        'date_from': from_raw,
        'date_to': to_raw,
        'score_distribution': distribution,
        'score_distribution_data': {'labels': [row['label'] for row in distribution], 'values': [row['count'] for row in distribution]},
        'skill_stats': skill_stats,
        'skill_chart_data': {'labels': [row['label'] for row in skill_stats['skills']], 'values': [row['count'] for row in skill_stats['skills']]},
        'document_methods': method_stats,
        'document_method_data': {'labels': [row['label'] for row in method_stats], 'values': [row['count'] for row in method_stats]},
        'method_comparison': comparison,
        'comparison_data': {'labels': ['Skill baseline', 'Hybrid score'], 'values': [comparison['baseline_average'] or 0, comparison['hybrid_average'] or 0]},
        'status_distribution': [
            {'label': status_choices[key], 'count': status_counts[count_key], 'status': key}
            for key, count_key in (
                ('completed', 'successful'), ('needs_review', 'review'),
                ('failed', 'failed'), ('pending', 'pending'),
            )
        ],
        'status_chart_data': {
            'labels': [status_choices[key] for key in ('completed', 'needs_review', 'failed', 'pending')],
            'values': [status_counts['successful'], status_counts['review'], status_counts['failed'], status_counts['pending']],
        },
        'scope_label': selected_job.title if selected_job else 'All candidates',
    }
    return render(request, 'core/dashboard.html', context)


# ── Job Description Views ──────────────────────────────────────────────────────

@login_required
def job_list(request):
    jobs = JobDescription.objects.all()
    query = request.GET.get('q', '')
    if query:
        jobs = jobs.filter(Q(title__icontains=query) | Q(required_skills__icontains=query))
    return render(request, 'core/job_list.html', {'jobs': jobs, 'query': query})


@login_required
def job_create(request):
    form = JobDescriptionForm(request.POST or None)
    if request.method == 'POST' and form.is_valid():
        job = form.save(commit=False)
        job.created_by = request.user
        job.save()
        messages.success(request, f'Job "{job.title}" created successfully.')
        return redirect('job_list')
    return render(request, 'core/job_form.html', {'form': form, 'title': 'Add Job Description'})


@login_required
def job_edit(request, pk):
    job = get_object_or_404(JobDescription, pk=pk)
    form = JobDescriptionForm(request.POST or None, instance=job)
    if request.method == 'POST' and form.is_valid():
        form.save()
        messages.success(request, 'Job description updated.')
        return redirect('job_list')
    return render(request, 'core/job_form.html', {'form': form, 'title': 'Edit Job Description', 'job': job})


@login_required
def job_delete(request, pk):
    job = get_object_or_404(JobDescription, pk=pk)
    if request.method == 'POST':
        job.delete()
        messages.success(request, 'Job description deleted.')
        return redirect('job_list')
    return render(request, 'core/confirm_delete.html', {'obj': job, 'type': 'Job Description'})


@login_required
def job_detail(request, pk):
    job = get_object_or_404(JobDescription, pk=pk)
    results = MatchResult.objects.filter(job=job).select_related('candidate').order_by('-match_percentage')
    return render(request, 'core/job_detail.html', {'job': job, 'results': results})




@login_required
def candidate_resume_file(request, pk):
    candidate = get_object_or_404(Candidate, pk=pk)
    try:
        path = Path(candidate.resume.path).resolve()
        media_root = Path(candidate.resume.storage.location).resolve()
        if media_root not in path.parents or not path.is_file():
            raise Http404('Document not found.')
        return FileResponse(open(path, 'rb'), as_attachment=False, filename=path.name)
    except (ValueError, OSError, AttributeError):
        raise Http404('Document not found.')

# ── Candidate Views ────────────────────────────────────────────────────────────

@login_required
def candidate_list(request):
    candidates = Candidate.objects.all()
    query = request.GET.get('q', '')
    skill_filter = request.GET.get('skill', '')
    job_filter = request.GET.get('job', '')

    if query:
        candidates = candidates.filter(Q(name__icontains=query) | Q(email__icontains=query))
    if skill_filter:
        candidates = candidates.filter(extracted_skills__icontains=skill_filter)

    jobs = JobDescription.objects.all()
    if job_filter:
        matched_ids = MatchResult.objects.filter(job_id=job_filter).values_list('candidate_id', flat=True)
        candidates = candidates.filter(id__in=matched_ids)

    return render(request, 'core/candidate_list.html', {
        'candidates': candidates,
        'query': query,
        'skill_filter': skill_filter,
        'job_filter': job_filter,
        'jobs': jobs,
    })


@login_required
def candidate_create(request):
    form = CandidateForm(request.POST or None, request.FILES or None)
    if request.method == 'POST' and form.is_valid():
        candidate = form.save(commit=False)
        candidate.uploaded_by = request.user
        candidate.save()

        if process_candidate_document(candidate):
            if candidate.extraction_status == 'needs_review':
                messages.warning(request, f'Resume for {candidate.name} was processed with review notes. Check the extracted text.')
            else:
                messages.success(request, f'Resume for {candidate.name} uploaded and processed.')
        else:
            messages.error(request, f'Resume saved, but processing failed: {candidate.extraction_error}')
        return redirect('candidate_detail', pk=candidate.pk)
    return render(request, 'core/candidate_form.html', {'form': form, 'title': 'Upload Resume'})


@login_required
def candidate_edit(request, pk):
    candidate = get_object_or_404(Candidate, pk=pk)
    previous_mode = candidate.processing_mode
    form = CandidateForm(request.POST or None, request.FILES or None, instance=candidate)
    if request.method == 'POST' and form.is_valid():
        if 'resume' in request.FILES or form.cleaned_data['processing_mode'] != previous_mode:
            candidate = form.save(commit=False)
            # A new file or extraction mode invalidates the previous result/review.
            candidate.extracted_text = ''
            candidate.extracted_skills = ''
            candidate.reviewed_text = ''
            candidate.structured_extraction = {}
            candidate.reviewed_profile = {}
            candidate.extraction_method = ''
            candidate.extraction_error = ''
            candidate.extraction_warnings = ''
            candidate.extraction_page_count = None
            candidate.extraction_processed_at = None
            candidate.extraction_status = 'pending'
            candidate.save()
            if process_candidate_document(candidate):
                if candidate.extraction_status == 'needs_review':
                    messages.warning(request, 'Candidate updated. The replacement resume needs text review.')
                else:
                    messages.success(request, 'Candidate and resume updated.')
            else:
                messages.error(request, f'Candidate updated, but resume processing failed: {candidate.extraction_error}')
        else:
            form.save()
            messages.success(request, 'Candidate updated.')
        return redirect('candidate_detail', pk=candidate.pk)
    return render(request, 'core/candidate_form.html', {'form': form, 'title': 'Edit Candidate', 'candidate': candidate})


@login_required
@require_POST
def retry_candidate_extraction(request, pk):
    candidate = get_object_or_404(Candidate, pk=pk)
    if process_candidate_document(candidate):
        if candidate.extraction_status == 'needs_review':
            messages.warning(request, 'Processing finished with review notes. Check the extracted text and original document.')
        else:
            messages.success(request, 'Document processing completed.')
    else:
        messages.error(request, f'Document processing failed: {candidate.extraction_error}')
    return redirect('candidate_detail', pk=candidate.pk)


@login_required
def candidate_delete(request, pk):
    candidate = get_object_or_404(Candidate, pk=pk)
    if request.method == 'POST':
        candidate.delete()
        messages.success(request, 'Candidate deleted.')
        return redirect('candidate_list')
    return render(request, 'core/confirm_delete.html', {'obj': candidate, 'type': 'Candidate'})


@login_required
def candidate_detail(request, pk):
    candidate = get_object_or_404(Candidate, pk=pk)
    results = MatchResult.objects.filter(candidate=candidate).select_related('job').order_by('-match_percentage')
    match_form = MatchForm()
    profile = candidate.reviewed_profile or candidate.structured_extraction or {}
    profile_form = StructuredResumeProfileForm(initial=profile_to_form_initial(profile))
    sections = [
        {'key': field, 'title': FIELD_LABELS[field], 'items': profile.get(field, []),
         'status': profile.get('field_status', {}).get(field, 'missing')}
        for field in SECTION_FIELDS
    ]
    return render(request, 'core/candidate_detail.html', {
        'candidate': candidate,
        'results': results,
        'match_form': match_form,
        'structured_profile': profile,
        'profile_form': profile_form,
        'profile_sections': sections,
        'profile_scalars': [
            {'key': field, 'title': FIELD_LABELS[field], 'item': profile.get(field, {})}
            for field in SCALAR_FIELDS
        ],
        'profile_is_reviewed': bool(candidate.reviewed_profile),
    })


@login_required
def ai_document_assistant(request):
    """Current-session assistant over authenticated workspace resume records."""
    candidates = Candidate.objects.all().only(
        'id', 'name', 'resume', 'processing_mode', 'extraction_status', 'uploaded_at',
    ).order_by('name', 'id')
    jobs = JobDescription.objects.only('id', 'title').order_by('title')
    selected_candidate = None
    compared_candidate = None
    selected_job = None
    history = []
    sources = []
    answer = ''
    error = ''
    question = ''

    if request.method == 'POST':
        candidate_id = request.POST.get('candidate', '').strip()
        compare_id = request.POST.get('compare_candidate', '').strip()
        job_id = request.POST.get('job', '').strip()
        try:
            selected_candidate = get_object_or_404(Candidate, pk=int(candidate_id))
            if compare_id:
                compared_candidate = get_object_or_404(Candidate, pk=int(compare_id))
            if job_id:
                selected_job = get_object_or_404(JobDescription, pk=int(job_id))
        except (ValueError, TypeError):
            error = 'Choose a valid resume and, when applicable, a valid comparison job.'
        action = request.POST.get('action', 'ask')
        scope = '|'.join(str(value or '') for value in (
            selected_candidate.pk if selected_candidate else candidate_id,
            compared_candidate.pk if compared_candidate else compare_id,
            selected_job.pk if selected_job else job_id,
        ))
        session_key = 'smartdoc_assistant_' + hashlib.sha256(scope.encode()).hexdigest()[:24]
        history = request.session.get(session_key, []) if selected_candidate else []
        if action == 'clear' and selected_candidate:
            request.session.pop(session_key, None)
            request.session.modified = True
            history = []
        elif not error:
            question = request.POST.get('question', '').strip()
            if not question or len(question) > MAX_QUESTION_CHARS:
                error = f'Enter a question of 1–{MAX_QUESTION_CHARS} characters.'
            elif is_comparison_question(question) and (not compared_candidate or not selected_job):
                error = 'Select a second candidate and a job to compare candidates.'
            elif compared_candidate and compared_candidate.pk == selected_candidate.pk:
                error = 'Choose two different candidates for a comparison.'
            else:
                try:
                    retrieval_query = question
                    if selected_job:
                        retrieval_query += ' ' + selected_job.required_skills[:2000] + ' ' + selected_job.description[:2000]
                    evidence = retrieve_evidence(selected_candidate, retrieval_query)
                    evidence_groups = [evidence]
                    if is_comparison_question(question):
                        second = retrieve_evidence(compared_candidate, retrieval_query, start=len(evidence) + 1)
                        evidence_groups.append(second)
                    all_sources = [source for group in evidence_groups for source in group]
                    provider = get_answer_provider()
                    raw_answer = provider.answer(build_evidence_messages(
                        question, evidence_groups, job=selected_job, history=history,
                    ))
                    answer, cited_source_ids = validate_generated_answer(
                        raw_answer, all_sources, question=question,
                    )
                    # Only passages whose IDs survived grounding validation are
                    # shown as cited evidence for the generated answer.
                    sources = [source for source in all_sources if source['id'] in cited_source_ids]
                    history = (history + [
                        {'role': 'user', 'content': question[:MAX_QUESTION_CHARS]},
                        {'role': 'assistant', 'content': answer[:1800]},
                    ])[-12:]
                    request.session[session_key] = history
                    request.session.modified = True
                except DocumentNotReady as exc:
                    error = str(exc)
                except AssistantProviderError as exc:
                    error = str(exc)
                except Exception:
                    # Avoid leaking local paths, candidate text, or provider internals.
                    error = 'The assistant could not complete this request. Check the local model service and retry.'

    # Rehydrate current-session turns after an ordinary GET only when the user
    # explicitly selected a candidate. Conversation keys include the full scope.
    if request.method == 'GET' and request.GET.get('candidate', '').isdigit():
        selected_candidate = get_object_or_404(Candidate, pk=request.GET['candidate'])
        compared_candidate = None
        compare_query = request.GET.get('compare_candidate', request.GET.get('compare', ''))
        if compare_query.isdigit():
            compared_candidate = get_object_or_404(Candidate, pk=compare_query)
        selected_job = JobDescription.objects.filter(pk=request.GET.get('job')).first()
        scope = '|'.join(str(value or '') for value in (
            selected_candidate.pk, compared_candidate.pk if compared_candidate else '',
            selected_job.pk if selected_job else '',
        ))
        session_key = 'smartdoc_assistant_' + hashlib.sha256(scope.encode()).hexdigest()[:24]
        history = request.session.get(session_key, [])

    return render(request, 'core/ai_assistant.html', {
        'candidates': candidates, 'jobs': jobs, 'selected_candidate': selected_candidate,
        'compared_candidate': compared_candidate, 'selected_job': selected_job,
        'history': history, 'sources': sources, 'answer': answer, 'error': error,
        'question': question,
    })




@login_required
@require_POST
def save_document_review(request, pk):
    candidate = get_object_or_404(Candidate, pk=pk)
    corrected = request.POST.get('reviewed_text', '')
    if len(corrected) > 200000:
        messages.error(request, 'Reviewed text is too long to save.')
    elif not corrected.strip():
        messages.error(request, 'Reviewed text cannot be empty.')
    else:
        candidate.reviewed_text = corrected
        candidate.extracted_skills = ', '.join(extract_skills_from_text(corrected))
        candidate.reviewed_profile = mark_profile_from_reviewed_text(extract_structured_profile(corrected))
        candidate.extraction_status = 'completed'
        candidate.extraction_error = ''
        candidate.save(update_fields=[
            'reviewed_text', 'extracted_skills', 'reviewed_profile', 'extraction_status', 'extraction_error',
        ])
        messages.success(request, 'Reviewed text saved. Original extraction has been preserved.')
    return redirect('candidate_detail', pk=pk)


@login_required
@require_POST
def save_structured_profile(request, pk):
    candidate = get_object_or_404(Candidate, pk=pk)
    if candidate.processing_mode == Candidate.PROCESSING_HANDWRITING and not candidate.reviewed_text.strip():
        messages.error(request, 'Review and save the handwriting transcription before editing structured fields or skills.')
        return redirect('candidate_detail', pk=pk)
    form = StructuredResumeProfileForm(request.POST)
    if not form.is_valid():
        messages.error(request, 'Check the structured profile fields and correct the highlighted values.')
        return redirect('candidate_detail', pk=pk)

    automatic = candidate.structured_extraction or {}
    previous = candidate.reviewed_profile or {}
    candidate.reviewed_profile = apply_profile_corrections(form.cleaned_data, automatic, previous)
    candidate.extracted_skills = ', '.join(
        item['value'] for item in candidate.reviewed_profile.get('skills', [])
    )
    candidate.save(update_fields=['reviewed_profile', 'extracted_skills'])
    messages.success(request, 'Structured resume profile corrections saved. Original extraction and source evidence are preserved.')
    return redirect('candidate_detail', pk=pk)

# ── Matching Views ─────────────────────────────────────────────────────────────

@login_required
def run_match(request, candidate_pk):
    candidate = get_object_or_404(Candidate, pk=candidate_pk)
    if request.method == 'POST':
        form = MatchForm(request.POST)
        if form.is_valid():
            job = form.cleaned_data['job']
            method = request.POST.get('method', 'baseline')
            try:
                if method == 'hybrid':
                    comparison = compute_hybrid_match(candidate, job)
                    match_pct = comparison['score']
                    matching = comparison['matching_skills']
                    missing = comparison['missing_skills']
                    explanation = {'method': 'hybrid', **comparison}
                elif method == 'baseline':
                    validate_candidate_for_matching(candidate)
                    match_pct, matching, missing = compute_match(candidate.get_skills_list(), job.get_skills_list())
                    explanation = {
                        'method': 'baseline', 'baseline_score': match_pct,
                        'semantic_score': None,
                        'interpretation': 'Required-skill keyword coverage only; not proof of qualification.',
                        'text_source': 'reviewed_text' if candidate.reviewed_text else 'completed_extraction',
                        'limitations': [candidate.extraction_warnings] if candidate.extraction_warnings else [],
                    }
                else:
                    raise SemanticMatchingError('Choose either skill coverage or hybrid semantic matching.')
            except SemanticMatchingError as exc:
                messages.error(request, str(exc))
                return redirect('candidate_detail', pk=candidate_pk)
            recommendation = get_recommendation(match_pct)

            result, created = MatchResult.objects.update_or_create(
                candidate=candidate, job=job,
                defaults={
                    'match_percentage': match_pct,
                    'matching_skills': ', '.join(matching),
                    'missing_skills': ', '.join(missing),
                    'recommendation': recommendation,
                    'explanation': explanation,
                }
            )
            messages.success(request, f'{"Hybrid" if method == "hybrid" else "Skill coverage"} match computed: {match_pct}% — {result.get_recommendation_display()}')
    return redirect('candidate_detail', pk=candidate_pk)


@login_required
def match_all_candidates(request, job_pk):
    if request.method != 'POST':
        return redirect('job_detail', pk=job_pk)
    job = get_object_or_404(JobDescription, pk=job_pk)
    method = request.POST.get('method', 'baseline')
    candidates = Candidate.objects.all()
    count = 0
    skipped = 0
    first_processing_error = ''
    for candidate in candidates:
        try:
            if method == 'hybrid':
                comparison = compute_hybrid_match(candidate, job)
                match_pct = comparison['score']
                matching = comparison['matching_skills']
                missing = comparison['missing_skills']
                explanation = {'method': 'hybrid', **comparison}
            elif method == 'baseline':
                validate_candidate_for_matching(candidate)
                match_pct, matching, missing = compute_match(candidate.get_skills_list(), job.get_skills_list())
                explanation = {
                    'method': 'baseline', 'baseline_score': match_pct, 'semantic_score': None,
                    'interpretation': 'Required-skill keyword coverage only; not proof of qualification.',
                    'text_source': 'reviewed_text' if candidate.reviewed_text else 'completed_extraction',
                    'limitations': [candidate.extraction_warnings] if candidate.extraction_warnings else [],
                }
            else:
                messages.error(request, 'Choose either skill coverage or hybrid semantic matching.')
                return redirect('job_detail', pk=job_pk)
        except SemanticModelUnavailable as exc:
            messages.error(request, str(exc))
            return redirect('job_detail', pk=job_pk)
        except SemanticMatchingError as exc:
            # A failed candidate does not prevent the other candidate rows from
            # being processed; each failed row keeps its earlier result intact.
            skipped += 1
            first_processing_error = first_processing_error or str(exc)
            continue
        recommendation = get_recommendation(match_pct)
        MatchResult.objects.update_or_create(
            candidate=candidate, job=job,
            defaults={
                'match_percentage': match_pct,
                'matching_skills': ', '.join(matching),
                'missing_skills': ', '.join(missing),
                'recommendation': recommendation,
                'explanation': explanation,
            }
        )
        count += 1
    messages.success(request, f'Matched {count} candidates against "{job.title}"; skipped {skipped} with failed, incomplete, or unreviewed evidence.')
    if first_processing_error:
        messages.warning(request, first_processing_error)
    return redirect('job_detail', pk=job_pk)


@login_required
def ranking(request):
    job_id = request.GET.get('job', '')
    min_pct = request.GET.get('min_pct', '')
    rec_filter = request.GET.get('recommendation', '')

    results = MatchResult.objects.select_related('candidate', 'job').order_by('-match_percentage')

    jobs = JobDescription.objects.all()

    if job_id:
        results = results.filter(job_id=job_id)
    if min_pct:
        try:
            threshold = float(min_pct)
            if 0 <= threshold <= 100:
                results = results.filter(match_percentage__gte=threshold)
            else:
                min_pct = ''
                messages.warning(request, 'Minimum match score must be between 0 and 100.')
        except (TypeError, ValueError):
            min_pct = ''
            messages.warning(request, 'Enter a valid minimum match score.')
    if rec_filter:
        results = results.filter(recommendation=rec_filter)

    return render(request, 'core/ranking.html', {
        'results': results,
        'jobs': jobs,
        'job_id': job_id,
        'min_pct': min_pct,
        'rec_filter': rec_filter,
    })


@login_required
def export_excel(request):
    job_id = request.GET.get('job', '')
    results = MatchResult.objects.select_related('candidate', 'job').order_by('-match_percentage')
    if job_id:
        results = results.filter(job_id=job_id)

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Candidate Rankings"

    headers = ['Rank', 'Candidate Name', 'Email', 'Mobile', 'Experience',
               'Job Title', 'Match %', 'Method', 'Skill Coverage %', 'Semantic Relevance %',
               'Matching Skills', 'Skills Not Found', 'Recommendation']

    header_fill = PatternFill(start_color="1a237e", end_color="1a237e", fill_type="solid")
    header_font = Font(bold=True, color="FFFFFF")

    for col, header in enumerate(headers, 1):
        cell = ws.cell(row=1, column=col, value=header)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal='center')

    rec_colors = {
        'highly_recommended': 'C8E6C9',
        'recommended': 'FFF9C4',
        'not_suitable': 'FFCDD2',
    }

    def safe_cell(value):
        if isinstance(value, str) and value[:1] in ('=', '+', '-', '@'):
            return "'" + value
        return value

    for rank, result in enumerate(results, 1):
        row_data = [
            rank, result.candidate.name, result.candidate.email,
            result.candidate.mobile, result.candidate.experience,
            result.job.title, result.match_percentage,
            (result.explanation or {}).get('method', 'baseline'),
            (result.explanation or {}).get('baseline_score'),
            (result.explanation or {}).get('semantic_score'),
            result.matching_skills, result.missing_skills,
            result.get_recommendation_display()
        ]
        fill = PatternFill(start_color=rec_colors.get(result.recommendation, 'FFFFFF'),
                           end_color=rec_colors.get(result.recommendation, 'FFFFFF'),
                           fill_type="solid")
        for col, value in enumerate(row_data, 1):
            cell = ws.cell(row=rank + 1, column=col, value=safe_cell(value))
            cell.fill = fill

    for col in ws.columns:
        max_len = max((len(str(cell.value or '')) for cell in col), default=10)
        ws.column_dimensions[col[0].column_letter].width = min(max_len + 4, 40)

    response = HttpResponse(content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    response['Content-Disposition'] = 'attachment; filename="candidate_rankings.xlsx"'
    wb.save(response)
    return response
