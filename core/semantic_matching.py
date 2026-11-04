"""Optional, local-only semantic resume/job comparison.

Semantic scores are ranking signals, not probabilities or hiring decisions.
The existing exact skill coverage calculation remains the stable baseline.
"""
from __future__ import annotations

import math
import os
import re
from threading import Lock

from .utils import compute_match

MODEL_ID = 'sentence-transformers/all-MiniLM-L6-v2'
MAX_RESUME_CHARS = 60_000
MAX_JOB_CHARS = 20_000
MAX_PASSAGES = 240
MAX_REQUIREMENTS = 80

_MODEL = None
_MODEL_LOCK = Lock()


class SemanticMatchingError(RuntimeError):
    """A local model/configuration/evidence problem that can be shown to users."""


class SemanticModelUnavailable(SemanticMatchingError):
    pass


def _model_name():
    return os.environ.get('SMARTDOC_SEMANTIC_MODEL', MODEL_ID).strip() or MODEL_ID


def _get_model():
    """Load once per process, CPU-first, and never access the network."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise SemanticModelUnavailable(
                'Semantic matching is unavailable: install the optional requirements-semantic.txt '
                'dependencies and configure the local model first. Skill coverage is still available.'
            ) from exc
        model_path = os.environ.get('SMARTDOC_SEMANTIC_MODEL_PATH', '').strip() or _model_name()
        device = os.environ.get('SMARTDOC_SEMANTIC_DEVICE', 'cpu').strip() or 'cpu'
        try:
            _MODEL = SentenceTransformer(model_path, device=device, local_files_only=True)
        except Exception as exc:
            raise SemanticModelUnavailable(
                'Semantic matching model is not available locally. Follow the optional model setup '
                'instructions, then retry. No model download was attempted.'
            ) from exc
    return _MODEL


def _profile_text(candidate):
    """Combine corrected profile values with the text used for review/matching."""
    pieces = []
    profile = candidate.reviewed_profile or {}
    for field in ('full_name', 'email', 'phone'):
        item = profile.get(field, {})
        if isinstance(item, dict) and item.get('value'):
            pieces.append(str(item['value']))
    for field in ('education', 'skills', 'projects', 'work_experience', 'internships', 'certifications', 'achievements', 'dates'):
        for item in profile.get(field, []) if isinstance(profile.get(field, []), list) else []:
            if isinstance(item, dict) and item.get('value'):
                pieces.append(str(item['value']))
    # Reviewed text is authoritative; original text supplies preserved evidence
    # if only individual structured fields were corrected.
    pieces.extend(part for part in (candidate.reviewed_text, candidate.get_effective_text()) if part)
    unique = list(dict.fromkeys(piece.strip() for piece in pieces if piece and piece.strip()))
    return '\n'.join(unique)[:MAX_RESUME_CHARS]


def _passages(text):
    page = None
    result = []
    for raw in text.splitlines():
        marker = re.fullmatch(r'\s*\[Page\s+(\d+)(?:[^\]]*)\]\s*', raw, re.I)
        if marker:
            page = int(marker.group(1))
            continue
        line = raw.strip().lstrip('•*-\u2022\t ').strip()
        for sentence in re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])|\s*[|;]\s*', line):
            sentence = sentence.strip()
            if len(sentence) >= 5:
                result.append({'text': sentence[:1200], 'page': page})
            if len(result) >= MAX_PASSAGES:
                return result
    return result


def _requirements(job):
    result = []
    seen = set()
    for value in job.get_skills_list():
        value = value.strip()
        if value and value.casefold() not in seen:
            result.append(value)
            seen.add(value.casefold())
    for sentence in re.split(r'(?<=[.!?])\s+|[\r\n]+', (job.description or '')[:MAX_JOB_CHARS]):
        sentence = sentence.strip(' \t-*•')
        if 12 <= len(sentence) <= 500 and sentence.casefold() not in seen:
            result.append(sentence)
            seen.add(sentence.casefold())
        if len(result) >= MAX_REQUIREMENTS:
            break
    return result[:MAX_REQUIREMENTS]


def get_weights():
    try:
        skill = float(os.environ.get('SMARTDOC_MATCH_SKILL_WEIGHT', '0.60'))
        semantic = float(os.environ.get('SMARTDOC_MATCH_SEMANTIC_WEIGHT', '0.40'))
    except ValueError as exc:
        raise SemanticMatchingError('Matching weights must be numeric values.') from exc
    total = skill + semantic
    if not all(math.isfinite(value) and value >= 0 for value in (skill, semantic)) or not math.isfinite(total) or total <= 0:
        raise SemanticMatchingError('Matching weights must be finite, non-negative, and sum to more than zero.')
    return {'skill_coverage': round(skill / total, 4), 'semantic_relevance': round(semantic / total, 4)}


def validate_candidate_for_matching(candidate):
    if candidate.extraction_status != 'completed' or not candidate.get_effective_text().strip():
        raise SemanticMatchingError(
            'Matching was not run because this resume has no completed, readable extraction. '
            'Review or reprocess the document first; incomplete extraction is not evidence of missing qualifications.'
        )
    if candidate.processing_mode == 'handwriting' and not candidate.reviewed_text.strip():
        raise SemanticMatchingError(
            'Handwriting text must be manually reviewed and saved before it can be used as matching evidence.'
        )


def compute_hybrid_match(candidate, job):
    """Return a transparent weighted score, existing baseline, and source snippets."""
    validate_candidate_for_matching(candidate)
    candidate_text = _profile_text(candidate)
    resume_passages = _passages(candidate_text)
    requirements = _requirements(job)
    if not resume_passages:
        raise SemanticMatchingError('No usable resume passages were available for semantic comparison.')
    if not requirements:
        raise SemanticMatchingError('Add required skills or a job description before semantic comparison.')

    baseline, matched, missing = compute_match(candidate.get_skills_list(), job.get_skills_list())
    model = _get_model()
    inputs = requirements + [item['text'] for item in resume_passages]
    try:
        vectors = model.encode(inputs, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        vectors = vectors.tolist() if hasattr(vectors, 'tolist') else vectors
        req_vectors = vectors[:len(requirements)]
        passage_vectors = vectors[len(requirements):]
        evidence = []
        best_scores = []
        for requirement, req_vector in zip(requirements, req_vectors):
            similarities = [sum(float(a) * float(b) for a, b in zip(req_vector, vector)) for vector in passage_vectors]
            best_idx = max(range(len(similarities)), key=similarities.__getitem__)
            cosine = max(-1.0, min(1.0, similarities[best_idx]))
            # Cosine is retained for transparency. Clipping at zero makes negative
            # similarity contribute no positive relevance; it is not a probability.
            best_scores.append(max(0.0, cosine))
            passage = resume_passages[best_idx]
            evidence.append({
                'requirement': requirement,
                'passage': passage['text'],
                'page': passage['page'],
                'cosine_similarity': round(cosine, 4),
            })
    except SemanticMatchingError:
        raise
    except Exception as exc:
        raise SemanticMatchingError('The local semantic model failed during inference. Existing match results were not changed.') from exc

    semantic = round(100 * sum(best_scores) / len(best_scores), 1)
    weights = get_weights()
    combined = round(baseline * weights['skill_coverage'] + semantic * weights['semantic_relevance'], 1)
    return {
        'score': combined,
        'baseline_score': baseline,
        'semantic_score': semantic,
        'matching_skills': matched,
        'missing_skills': missing,
        'weights': weights,
        'evidence': evidence,
        'model': _model_name(),
        'text_source': 'reviewed_text_and_corrected_profile' if candidate.reviewed_text or candidate.reviewed_profile else 'completed_extraction',
        'limitations': [candidate.extraction_warnings] if candidate.extraction_warnings else [],
        'interpretation': 'Weighted ranking signal only; semantic cosine is not confidence, probability, or proof of qualification.',
    }


def clear_model_cache():
    """Test hook and operational reset for a changed model/device configuration."""
    global _MODEL
    with _MODEL_LOCK:
        _MODEL = None
