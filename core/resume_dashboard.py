"""Transient, deterministic analytics. Every parsed field is unverified.

Entries require explicit section evidence; ambiguous sections retain source text.
Completeness gives six equal points for detected evidence, never hiring fitness.
"""
import re

from .structured_extraction import (
    HEADER_LOOKUP, GENERIC_HEADERS, EMAIL_RE, PHONE_RE, DATE_RE,
    _clean_header, _name_value, _looks_like_name, _page_lines,
)
from .utils import extract_skills_from_text, compute_match

CATEGORIES = {
    'Programming languages': 'python java javascript typescript c c++ c# ruby php swift kotlin go rust scala r matlab perl bash shell'.split(),
    'Web development': ['html', 'css', 'sass', 'rest api', 'graphql', 'microservices'],
    'Databases': 'sql mysql postgresql mongodb redis sqlite oracle elasticsearch cassandra firebase dynamodb'.split(),
    'Frameworks and libraries': 'react angular vue node nodejs express django flask fastapi spring laravel rails bootstrap jquery selenium junit pytest'.split(),
    'Cloud and DevOps': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'git', 'github', 'gitlab', 'ci/cd', 'terraform', 'ansible', 'linux', 'nginx', 'apache'],
    'Data science and AI/ML': ['machine learning', 'deep learning', 'nlp', 'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'pandas', 'numpy', 'matplotlib', 'data analysis', 'data science', 'power bi', 'tableau', 'spark', 'hadoop'],
}
CANONICAL = {'javascript': 'JavaScript', 'typescript': 'TypeScript', 'postgresql': 'PostgreSQL', 'mysql': 'MySQL', 'mongodb': 'MongoDB', 'nodejs': 'Node.js', 'github': 'GitHub', 'gitlab': 'GitLab', 'pytorch': 'PyTorch', 'tensorflow': 'TensorFlow', 'scikit-learn': 'scikit-learn'}


def display(skill):
    return CANONICAL.get(skill, skill.upper() if skill in {'c', 'c++', 'c#', 'html', 'css', 'sql', 'aws', 'gcp', 'nlp', 'ci/cd', 'rest api'} else skill.title())


def categorize(skills):
    buckets = {name: [] for name in [*CATEGORIES, 'Other technical tools']}
    for skill in sorted(set(skills)):
        category = next((name for name, values in CATEGORIES.items() if skill in values), 'Other technical tools')
        buckets[category].append(display(skill))
    return [{'label': name, 'skills': values, 'count': len(values)} for name, values in buckets.items() if values]


def sections_from(text):
    sections = {}
    current = 'header'
    for line in _page_lines(text):
        value = line['text']
        # Recognize inline headings as well as standalone headings.
        parts = value.split(':', 1)
        normalized = _clean_header(parts[0])
        key = {'project experience': 'projects', 'employment': 'work_experience',
               'academic projects': 'projects', 'education and qualifications': 'education',
               'professional profile': 'summary', 'career summary': 'summary'}.get(normalized) or HEADER_LOOKUP.get(normalized)
        if normalized in GENERIC_HEADERS:
            key = 'summary' if normalized in {'summary', 'professional summary', 'profile', 'objective', 'career objective'} else normalized
        if key:
            current = key
            sections.setdefault(current, [])
            if len(parts) == 2 and parts[1].strip():
                sections[current].append(parts[1].strip())
        else:
            sections.setdefault(current, []).append(value)
    return sections


def entries(lines, kind):
    """Only split on explicit title/degree/role signals; retain all source lines."""
    starts = []
    for i, line in enumerate(lines):
        if kind == 'education':
            signal = re.search(r'\b(?:bachelor|master|doctor|ph\.?d|b\.?tech|m\.?tech|b\.?sc|m\.?sc|b\.?e\.?|mba|diploma)\b', line, re.I)
        elif kind == 'experience':
            signal = re.search(r'\b(?:role|position)\s*:|\b(?:engineer|developer|analyst|intern|manager|designer|consultant)\b.*(?:\s[|@]\s|\sat\s)', line, re.I)
        else:
            signal = re.match(r'(?:project(?:\s+\d+)?\s*:|\d+[.)]\s+\S)', line, re.I) or (len(line) < 110 and ' | ' in line and not re.match(r'(?:built|developed|implemented|technologies)', line, re.I))
            if not signal and i + 1 < len(lines) and len(line.split()) <= 10 and not line.endswith('.'):
                signal = re.match(r'(?:[-•*]\s*|built\b|developed\b|implemented\b|technologies\s*:|description\s*:)', lines[i + 1], re.I)
        if signal:
            starts.append(i)
    if not starts or starts[0] != 0:
        return []
    result = []
    for n, start in enumerate(starts):
        block = lines[start:starts[n + 1] if n + 1 < len(starts) else len(lines)]
        source = '\n'.join(block)
        item = {'title': block[0], 'source': source, 'description': '\n'.join(block[1:]), 'technologies': []}
        if kind == 'education':
            item['degree'] = block[0]
            item['institution'] = next((line for line in block if re.search(r'\b(university|college|institute|school)\b', line, re.I)), '')
            specialization = re.search(r'(?:speciali[sz]ation|major)\s*:\s*([^\n|]+)', source, re.I)
            item['specialization'] = specialization.group(1) if specialization else ''
            year = re.search(r'(?:graduat(?:ed|ion|ing)|class of|expected)\s*[:\-]?\s*((?:19|20)\d{2})\b', source, re.I)
            item['year'] = year.group(1) if year else ''
        elif kind == 'experience':
            role_org = re.split(r'\s+(?:[|@]|at)\s+', block[0], maxsplit=1)
            item['role'] = role_org[0]
            item['organization'] = role_org[1] if len(role_org) > 1 else ''
            item['dates'] = ', '.join(match.group() for match in DATE_RE.finditer('\n'.join(block[:3])))
        else:
            item['technologies'] = [display(s) for s in extract_skills_from_text(source)]
        result.append(item)
    return result


def analyze(text, filename='', method='', warnings=None, job_description=''):
    sections = sections_from(text)
    header = sections.get('header', [])
    personal = {key: '' for key in ('name', 'email', 'phone', 'location', 'linkedin', 'github', 'portfolio')}
    # Names only come from the first header line, never from later sections.
    if header:
        name_line = next((line for line in header[:5] if re.match(r'^(?:full name|candidate name|name)\s*:', line, re.I)), None)
        if name_line is None:
            name_line = next((line for line in header[:3] if _clean_header(line) not in {'resume', 'cv', 'curriculum vitae'}), '')
        first = _name_value({'text': name_line})
        if _looks_like_name(first) and not re.search(r'\b(project|system|application|education|technical|skills|university|college)\b', first, re.I):
            personal['name'] = first
    emails = list(dict.fromkeys(EMAIL_RE.findall(text)))
    personal['email'] = ' / '.join(emails)
    for line in header + sections.get('contact', []) + sections.get('contact information', []) + sections.get('personal details', []):
        for value in PHONE_RE.findall(line):
            if 7 <= len(re.sub(r'\D', '', value)) <= 15 and not DATE_RE.fullmatch(value.strip()):
                personal['phone'] = value.strip()
        location = re.match(r'(?:location|address|based in)\s*:\s*(.+)', line, re.I)
        if location:
            personal['location'] = location.group(1)
    for service in ('linkedin', 'github'):
        match = re.search(r'(?:https?://)?(?:www\.)?' + service + r'\.com/[^\s<>|,]+', text, re.I)
        if match:
            personal[service] = match.group().rstrip('.;)')
    portfolio = re.search(r'(?:portfolio|website)\s*:\s*(https?://[^\s<>|]+)', text, re.I)
    if portfolio:
        personal['portfolio'] = portfolio.group(1).rstrip('.;)')
    else:
        websites = list(dict.fromkeys(url.rstrip('.;)') for url in re.findall(r'https?://[^\s<>|,]+', '\n'.join(header), re.I)
                                     if not re.search(r'(?:linkedin|github)\.com/', url, re.I)))
        if len(websites) == 1:
            personal['portfolio'] = websites[0]
    skills = extract_skills_from_text(text)
    groups = categorize(skills)
    education = entries(sections.get('education', []), 'education')
    work = entries(sections.get('work_experience', []), 'experience')
    internships = entries(sections.get('internships', []), 'experience')
    experience = work + internships
    projects = entries(sections.get('projects', []), 'projects')
    evidence = [
        ('Contact information', bool(personal['email'] or personal['phone']), 'Contact details were not detected; check the extracted text.'),
        ('Technical skills', bool(skills), 'No supported technical skills detected; unsupported terms may be present.'),
        ('Education', bool(sections.get('education')), 'An education section was not detected.'),
        ('Practical experience', bool(sections.get('work_experience') or sections.get('internships') or sections.get('projects')), 'Add relevant projects, internships, or employment evidence if available.'),
        ('Projects or work examples', bool(sections.get('projects') or sections.get('work_experience') or sections.get('internships')), 'No section containing practical work examples was detected.'),
        ('Professional summary', bool(sections.get('summary')), 'A professional summary was not detected.'),
    ]
    rubric = [{'label': label, 'detected': found, 'detail': 'Evidence detected in extracted text.' if found else reason} for label, found, reason in evidence]
    tips = [row['detail'] + ' Review the source before making changes.' for row in rubric if not row['detected']]
    if not personal['email']:
        tips.append('No email address was detected. Check that a readable contact email is included.')
    if sections.get('projects') and not projects:
        tips.append('Project boundaries could not be determined. Use explicit Project: titles and separate descriptions.')
    work_text = '\n'.join(sections.get('projects', []) + sections.get('work_experience', []) + sections.get('internships', []))
    if work_text and not re.search(r'\d+\s*(?:%|percent|users|customers|hours|seconds|revenue)', work_text, re.I):
        tips.append('No measurable outcomes were detected in work examples. Add verifiable measurements where you have them.')
    job = None
    if job_description.strip():
        required = extract_skills_from_text(job_description)
        score, matched, missing = compute_match(skills, required, precision=2)
        job = {'score': score if required else None, 'requirements': required, 'matched': matched, 'missing': missing,
               'recommendations': [f'{display(s)}: if you have this skill, support it with an example; otherwise develop it before listing it.' for s in missing]}
    return {'filename': filename, 'method': method, 'warnings': warnings or [], 'text': text,
            'personal': [{'label': {'linkedin': 'LinkedIn', 'github': 'GitHub'}.get(key, key.title()), 'value': value} for key, value in personal.items()],
            'skills': skills, 'categories': groups, 'education': education, 'experience': experience, 'projects': projects,
            'education_source': '\n'.join(sections.get('education', [])),
            'experience_source': '\n'.join(sections.get('work_experience', []) + sections.get('internships', [])),
            'projects_source': '\n'.join(sections.get('projects', [])),
            'project_count': len(projects) if projects else None,
            'experience_count': (len(experience) if experience and
                                 (not sections.get('work_experience') or work) and
                                 (not sections.get('internships') or internships) else None),
            'completeness': round(sum(row['detected'] for row in rubric) / 6 * 100),
            'rubric': rubric, 'recommendations': tips, 'job': job,
            # Preserve existing integration consumers when a JD is supplied.
            **({key: job[key] for key in ('score', 'matched', 'missing', 'requirements')} if job else {})}


def build_report(result):
    lines = ['SmartDoc AI 2.0 — Resume Analytics Report', result['filename'],
             'Extracted information is unverified. Not detected does not prove absence.',
             'Candidate overview:']
    lines += [f"{row['label']}: {row['value'] or 'Not detected'}" for row in result['personal']]
    lines += ['Detected technical skills: ' + ', '.join(display(s) for s in result['skills'])]
    lines += [f"{row['label']} ({row['count']}): {', '.join(row['skills'])}" for row in result['categories']]
    for key in ('education', 'experience', 'projects'):
        lines += [key.title() + ' — unverified source', result[key + '_source'] or 'Not detected']
    lines += [f"Resume Completeness: {result['completeness']}%", 'An indicator of detected resume information, not an ATS or hiring score.',
              'Rubric: six equal criteria, one point each. Projects or internships satisfy practical experience and work examples; employment is not required.']
    lines += [f"{'Detected' if row['detected'] else 'Not detected'}: {row['label']} — {row['detail']}" for row in result['rubric']]
    lines += ['Resume-quality recommendations:', *result['recommendations']]
    job = result['job']
    if job:
        lines += ['Job Skill Coverage: ' + (f"{job['score']}%" if job['score'] is not None else 'Not determined — no supported job skills'),
                  f"{len(job['matched'])} of {len(job['requirements'])} detected job-description skills matched.",
                  'Required: ' + ', '.join(display(s) for s in job['requirements']),
                  'Matched: ' + ', '.join(display(s) for s in job['matched']),
                  'Missing from extracted text: ' + ', '.join(display(s) for s in job['missing']),
                  'Job-matching recommendations:', *job['recommendations']]
    lines += ['Extraction method: ' + result['method'], 'Extraction warnings:', *result['warnings'], 'Extracted resume text:', result['text']]
    return '\n'.join(lines)
