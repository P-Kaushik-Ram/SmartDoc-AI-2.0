"""Deterministic, source-linked extraction of common resume sections."""
import re
import unicodedata

from .utils import COMMON_SKILLS

MAX_STRUCTURED_TEXT_CHARS = 200_000

SCALAR_FIELDS = ('full_name', 'email', 'phone')
SECTION_FIELDS = (
    'education', 'skills', 'projects', 'work_experience', 'internships',
    'certifications', 'achievements', 'dates',
)
FIELD_LABELS = {
    'full_name': 'Full name', 'email': 'Email', 'phone': 'Phone',
    'education': 'Education', 'skills': 'Skills', 'projects': 'Projects',
    'work_experience': 'Work experience', 'internships': 'Internships',
    'certifications': 'Certifications', 'achievements': 'Achievements',
    'dates': 'Dates and durations',
}

HEADER_ALIASES = {
    'education': {'education', 'educational background', 'academic background', 'academic qualifications'},
    'skills': {'skills', 'technical skills', 'core skills', 'core competencies', 'skills & tools', 'technologies'},
    'projects': {'projects', 'personal projects', 'academic projects', 'selected projects'},
    'work_experience': {'experience', 'work experience', 'professional experience', 'employment history', 'work history'},
    'internships': {'internship', 'internships', 'internship experience'},
    'certifications': {'certifications', 'certificates', 'licenses & certifications', 'licenses and certifications'},
    'achievements': {'achievements', 'awards', 'honors', 'honours', 'accomplishments'},
}
GENERIC_HEADERS = {
    'summary', 'professional summary', 'profile', 'objective', 'career objective',
    'contact', 'contact information', 'personal details', 'languages', 'interests',
    'volunteer experience', 'references',
}
HEADER_LOOKUP = {
    re.sub(r'[^a-z0-9]+', ' ', name.lower()).strip(): key
    for key, aliases in HEADER_ALIASES.items() for name in aliases
}
INLINE_HEADER_RE = re.compile(
    r'^(?P<header>' + '|'.join(
        re.escape(name) for name in sorted(HEADER_LOOKUP, key=len, reverse=True)
    ) + r')\s*[:\-–—]\s*(?P<value>.+)$', re.I,
)

EMAIL_RE = re.compile(r'(?<![\w.+-])[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}(?![\w.-])', re.I)
PHONE_RE = re.compile(r'(?<!\w)(?:\+?\d[\d\s()./-]{6,}\d)(?!\w)')
MONTH_RE = r'(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)'
DATE_RE = re.compile(
    rf'(?i)\b(?:{MONTH_RE}\.?\s+)?(?:19|20)\d{{2}}'
    rf'(?:\s*(?:-|–|—|to)\s*(?:{MONTH_RE}\s+)?(?:19|20)\d{{2}}|'
    rf'\s*(?:-|–|—|to)\s*(?:present|current))?\b'
)
DURATION_RE = re.compile(r'(?i)\b\d+(?:\.\d+)?\s*(?:\+\s*)?(?:years?|months?|weeks?)\b')
NAME_LABEL_RE = re.compile(
    r'^(?:full\s+name|candidate\s+name|name)\s*[:|\-–—]\s*(?P<value>.+)$', re.I,
)
OCR_DECORATION_RE = re.compile(r'^[\s\ufeff\u200b-\u200f\u202a-\u202e\u2060•●▪◦‣⁃·∙■□◆◇★☆✦✧❖*]+')
OCR_TRAILING_DECORATION_RE = re.compile(r'[\s\ufeff\u200b-\u200f\u202a-\u202e\u2060•●▪◦‣⁃·∙■□◆◇★☆✦✧❖*]+$')


def _temporal_matches(text):
    found = list(DATE_RE.finditer(text)) + list(DURATION_RE.finditer(text))
    return list(dict.fromkeys(match.group(0).strip() for match in sorted(found, key=lambda match: match.start())))


class StructuredExtractionError(ValueError):
    """Invalid input to the deterministic resume extraction service."""


def _new_scalar(value='', status='missing', sources=None, review_status='pending'):
    return {
        'value': value,
        'status': status,
        'review_status': review_status,
        'sources': sources or [],
    }


def _page_lines(text):
    page = None
    for raw_line in text.splitlines():
        page_marker = re.fullmatch(r'\s*\[Page\s+(\d+)(?:[^\]]*)\]\s*', raw_line, re.I)
        if page_marker:
            page = int(page_marker.group(1))
            continue
        # OCR often emits ornamental glyphs, zero-width marks, or repeated
        # bullet characters around otherwise useful text. Keep the original
        # extraction untouched on the Candidate; normalize only parser input.
        line = unicodedata.normalize('NFKC', raw_line).strip()
        line = OCR_DECORATION_RE.sub('', line)
        line = OCR_TRAILING_DECORATION_RE.sub('', line).strip()
        if line:
            yield {'text': line, 'page': page}


def _source(line):
    return {'page': line['page'], 'text': line['text']}


def _clean_header(line):
    return re.sub(r'[^a-z0-9]+', ' ', line.lower()).strip().rstrip(':').strip()


def _looks_like_name(line):
    candidate = line.strip().strip('|,;:-')
    words = candidate.split()
    if not 2 <= len(words) <= 5 or len(candidate) > 80:
        return False
    if re.search(r'[@\d]|https?://|\b(resume|curriculum vitae|profile|engineer|developer|manager)\b', candidate, re.I):
        return False
    if any(word.lower() in {'email', 'phone', 'address', 'linkedin', 'github'} for word in words):
        return False
    return all(
        word[0].isalpha() and all(character.isalpha() or character in "'.’-" for character in word)
        for word in words
    )


def _name_value(line):
    """Return a plausible header name fragment, without contact/title text."""
    text = line['text'].strip()
    labeled = NAME_LABEL_RE.match(text)
    if labeled:
        text = labeled.group('value').strip()
    else:
        # Common resume headers put the name and contact links on one row.
        # Keep only the leading fragment before a detectable contact value.
        positions = [match.start() for pattern in (EMAIL_RE, PHONE_RE) if (match := pattern.search(text))]
        if positions:
            text = text[:min(positions)]
    text = text.strip(' \t|,;:-–—•·✦✧❖')
    # Some templates place a role after a vertical bar; inspect only the
    # leading segment, so a title cannot turn into a name candidate.
    text = re.split(r'\s*[|]\s*', text, maxsplit=1)[0].strip(' \t,;:-–—')
    return text


def _field_items(lines):
    items = []
    seen = set()
    for line in lines:
        value = line['text'].strip()
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        dates = _temporal_matches(value)
        items.append({
            'value': value,
            'dates': dates,
            'status': 'extracted',
            'review_status': 'pending',
            'sources': [_source(line)],
        })
    return items


def extract_structured_profile(text):
    """Extract conservative resume fields with exact source-line/page references.

    This service uses email/phone/date patterns, explicit section headings, and
    the application's existing skills lexicon. It does not infer qualifications.
    """
    if not isinstance(text, str):
        raise StructuredExtractionError('Resume text must be a string.')
    if len(text) > MAX_STRUCTURED_TEXT_CHARS:
        raise StructuredExtractionError(f'Resume text exceeds {MAX_STRUCTURED_TEXT_CHARS} characters.')

    lines = list(_page_lines(text))
    profile = {key: _new_scalar() for key in SCALAR_FIELDS}
    profile.update({key: [] for key in SECTION_FIELDS})
    profile['schema_version'] = 1

    email_occurrences = []
    phone_occurrences = []
    for line in lines:
        for value in EMAIL_RE.findall(line['text']):
            email_occurrences.append((value, line))
        for value in PHONE_RE.findall(line['text']):
            value = re.sub(r'\s+', ' ', value).strip(' .-/')
            digits = re.sub(r'\D', '', value)
            if 7 <= len(digits) <= 15 and not DATE_RE.fullmatch(value):
                phone_occurrences.append((value, line))

    for field, occurrences in (('email', email_occurrences), ('phone', phone_occurrences)):
        unique = {}
        for value, line in occurrences:
            unique.setdefault(value.casefold(), (value, line))
        values = list(unique.values())
        if len(values) == 1:
            value, line = values[0]
            profile[field] = _new_scalar(value, 'extracted', [_source(line)])
        elif values:
            profile[field] = _new_scalar(
                ' / '.join(value for value, _ in values), 'ambiguous',
                [_source(line) for _, line in values],
            )

    current_section = None
    section_lines = {field: [] for field in HEADER_ALIASES}
    unsectioned = []
    for line in lines:
        normalized = _clean_header(line['text'])
        header = HEADER_LOOKUP.get(normalized)
        if header:
            current_section = header
            continue
        if normalized in GENERIC_HEADERS:
            current_section = None
            continue
        inline = INLINE_HEADER_RE.match(line['text'])
        if inline:
            current_section = HEADER_LOOKUP.get(_clean_header(inline.group('header')))
            value = inline.group('value').strip()
            if value:
                section_lines[current_section].append({'text': value, 'page': line['page']})
            continue
        (section_lines[current_section] if current_section else unsectioned).append(line)

    # A name is emitted only when the pre-section header contains one plausible
    # name line. Multiple plausible lines remain ambiguous for human review.
    candidates = {}
    for line in unsectioned[:12]:
        value = _name_value(line)
        if _looks_like_name(value):
            candidates.setdefault(value.casefold(), (value, line))
    if len(candidates) == 1:
        value, line = next(iter(candidates.values()))
        profile['full_name'] = _new_scalar(value, 'extracted', [_source(line)])
    elif candidates:
        profile['full_name'] = _new_scalar(
            ' / '.join(value for value, _ in candidates.values()), 'ambiguous',
            [_source(line) for _, line in candidates.values()],
        )

    for field in ('education', 'projects', 'work_experience', 'internships', 'certifications', 'achievements'):
        profile[field] = _field_items(section_lines[field])

    skill_lines = section_lines['skills']
    skills = []
    seen_skills = set()
    for line in skill_lines:
        for skill in COMMON_SKILLS:
            if re.search(r'(?<!\w)' + re.escape(skill) + r'(?!\w)', line['text'], re.I):
                key = skill.casefold()
                if key not in seen_skills:
                    seen_skills.add(key)
                    skills.append({
                        'value': skill,
                        'dates': [],
                        'status': 'extracted',
                        'review_status': 'pending',
                        'sources': [_source(line)],
                    })
    profile['skills'] = skills

    date_items = []
    for line in lines:
        for date in _temporal_matches(line['text']):
            date_items.append((date, line))
    unique_dates = {}
    for date, line in date_items:
        unique_dates.setdefault((date.casefold(), line['text'].casefold()), (date, line))
    profile['dates'] = [
        {'value': date, 'dates': [date], 'status': 'extracted', 'review_status': 'pending', 'sources': [_source(line)]}
        for date, line in unique_dates.values()
    ]

    for field in SCALAR_FIELDS:
        if profile[field]['status'] == 'missing':
            profile[field]['review_status'] = 'not_found'
    profile['field_status'] = {
        field: profile[field]['status'] for field in SCALAR_FIELDS
    } | {
        field: ('extracted' if profile[field] else 'missing') for field in SECTION_FIELDS
    }
    return profile


def profile_to_form_initial(profile):
    """Flatten extraction output into safe editable form values."""
    initial = {}
    for field in SCALAR_FIELDS:
        initial[field] = profile.get(field, {}).get('value', '')
    for field in SECTION_FIELDS:
        if field == 'dates':
            values = [item.get('value', '') for item in profile.get(field, [])]
        else:
            values = [item.get('value', '') for item in profile.get(field, [])]
        initial[field] = '\n'.join(value for value in values if value)
    return initial


def apply_profile_corrections(values, automatic_profile, previous_profile=None):
    """Create a reviewed profile, retaining evidence for unchanged extracted values."""
    reviewed = {'schema_version': 1, 'source_type': 'manual_profile_review', 'field_status': {}}
    for field in SCALAR_FIELDS:
        value = values.get(field, '').strip()
        original = automatic_profile.get(field, {})
        previous = (previous_profile or {}).get(field, original)
        unchanged = value.casefold() == previous.get('value', '').strip().casefold()
        reviewed[field] = {
            'value': value,
            'status': 'missing' if not value else 'reviewed',
            'review_status': 'reviewed',
            'sources': previous.get('sources', []) if unchanged else [],
            'origin': (previous.get('origin') or 'automatic_value_reviewed') if unchanged else 'manual_correction',
        }
        reviewed['field_status'][field] = reviewed[field]['status']

    for field in SECTION_FIELDS:
        original_items = automatic_profile.get(field, [])
        previous_items = (previous_profile or {}).get(field, original_items)
        by_value = {}
        for item in [*original_items, *previous_items]:
            by_value[item.get('value', '').casefold()] = item
        items = []
        for raw in values.get(field, '').splitlines():
            parts = re.split(r'\s*[,;|]\s*', raw) if field == 'skills' else [raw]
            for part in parts:
                value = part.strip().lstrip('•*- ').strip()
                if not value:
                    continue
                original = by_value.get(value.casefold(), {})
                dates = _temporal_matches(value)
                items.append({
                    'value': value,
                    'dates': dates or original.get('dates', []),
                    'status': 'reviewed',
                    'review_status': 'reviewed',
                    'sources': original.get('sources', []),
                    'origin': (original.get('origin') or 'automatic_value_reviewed') if original else 'manual_correction',
                })
        reviewed[field] = items
        reviewed['field_status'][field] = 'reviewed' if items else 'missing'
    return reviewed


def mark_profile_from_reviewed_text(profile):
    """Label deterministic extraction performed over a user's corrected text."""
    profile['source_type'] = 'reviewed_text'
    for field in SCALAR_FIELDS:
        profile[field]['origin'] = 'reviewed_text'
        profile[field]['review_status'] = 'derived_from_reviewed_text'
    for field in SECTION_FIELDS:
        for item in profile.get(field, []):
            item['origin'] = 'reviewed_text'
            item['review_status'] = 'derived_from_reviewed_text'
    return profile
