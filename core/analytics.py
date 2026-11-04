"""Small, database-backed analytics calculations used by the dashboard."""
from collections import Counter


DOCUMENT_METHOD_LABELS = {
    'pdf_text': 'Digital PDF text',
    'pdf_ocr': 'Scanned PDF · OCR',
    'pdf_text_ocr': 'Mixed PDF · text + OCR',
    'docx_text': 'DOCX text',
    'tesseract_ocr': 'Image · printed OCR',
    'handwriting_trocr': 'Handwriting recognition',
}


def score_distribution(scores):
    """Count stored match scores in fixed, readable 20-point bands."""
    bins = [
        ('0–19', 0, 20), ('20–39', 20, 40), ('40–59', 40, 60),
        ('60–79', 60, 80), ('80–100', 80, 101),
    ]
    counts = [0] * len(bins)
    for score in scores:
        try:
            numeric = float(score)
        except (TypeError, ValueError):
            continue
        if 0 <= numeric <= 100:
            for index, (_, lower, upper) in enumerate(bins):
                if lower <= numeric < upper:
                    counts[index] += 1
                    break
    return [{'label': label, 'count': count} for (label, _, _), count in zip(bins, counts)]


def skill_frequencies(candidates, limit=12):
    """Count unique skills per successfully extracted candidate.

    Corrected profile skills take precedence. Otherwise corrected resume text's
    skill list is used when present. Unreviewed handwriting and incomplete or
    review-required documents are excluded from this verified-skill chart.
    """
    counts = Counter()
    reviewed_candidates = 0
    automatic_candidates = 0
    excluded_handwriting = 0
    eligible_candidates = 0

    for candidate in candidates:
        if candidate.processing_mode == 'handwriting' and not candidate.reviewed_text.strip():
            excluded_handwriting += 1
            continue
        if candidate.extraction_status != 'completed':
            continue

        profile = candidate.reviewed_profile or {}
        profile_skills = profile.get('skills', []) if isinstance(profile, dict) else []
        values = [
            str(item.get('value', '')).strip()
            for item in profile_skills
            if isinstance(item, dict) and item.get('value')
        ] if isinstance(profile_skills, list) else []
        if values:
            reviewed_candidates += 1
        elif candidate.reviewed_text.strip():
            values = candidate.get_skills_list()
            reviewed_candidates += 1
        elif candidate.processing_mode != 'handwriting':
            values = candidate.get_skills_list()
            automatic_candidates += 1
        else:
            # A handwriting profile must have reviewed text before it is eligible.
            continue

        normalized = {value.casefold(): value for value in values if value}
        counts.update(normalized.keys())
        eligible_candidates += 1

    skills = [
        {'label': label.title(), 'count': count}
        for label, count in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[:limit]
    ]
    return {
        'skills': skills,
        'reviewed_candidates': reviewed_candidates,
        'automatic_candidates': automatic_candidates,
        'excluded_unreviewed_handwriting': excluded_handwriting,
        'eligible_candidates': eligible_candidates,
    }


def document_method_counts(rows):
    """Label recorded extraction methods without inferring missing history."""
    counts = Counter()
    for method, count in rows:
        key = method or 'unrecorded'
        label = DOCUMENT_METHOD_LABELS.get(key, key.replace('_', ' ').title() if key != 'unrecorded' else 'Method not recorded')
        counts[label] += int(count)
    return [{'label': label, 'count': count} for label, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]


def paired_match_averages(rows):
    """Average same-result skill baseline and hybrid total for comparable rows."""
    baselines = []
    hybrid_scores = []
    for baseline, hybrid in rows:
        try:
            baseline = float(baseline)
            hybrid = float(hybrid)
        except (TypeError, ValueError):
            continue
        if 0 <= baseline <= 100 and 0 <= hybrid <= 100:
            baselines.append(baseline)
            hybrid_scores.append(hybrid)
    if not baselines:
        return {'count': 0, 'baseline_average': None, 'hybrid_average': None}
    return {
        'count': len(baselines),
        'baseline_average': round(sum(baselines) / len(baselines), 1),
        'hybrid_average': round(sum(hybrid_scores) / len(hybrid_scores), 1),
    }
