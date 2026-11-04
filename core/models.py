from django.db import models
from django.contrib.auth.models import User


class JobDescription(models.Model):
    title = models.CharField(max_length=200)
    required_skills = models.TextField(help_text="Comma-separated skills")
    experience = models.CharField(max_length=100)
    description = models.TextField()
    created_date = models.DateTimeField(auto_now_add=True)
    created_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)

    def __str__(self):
        return self.title

    def get_skills_list(self):
        return [s.strip().lower() for s in self.required_skills.split(',') if s.strip()]

    class Meta:
        ordering = ['-created_date']


class Candidate(models.Model):
    PROCESSING_STANDARD = 'standard'
    PROCESSING_HANDWRITING = 'handwriting'
    PROCESSING_MODES = [
        (PROCESSING_STANDARD, 'Standard extraction (digital text / printed OCR)'),
        (PROCESSING_HANDWRITING, 'English handwriting recognition (TrOCR)'),
    ]

    name = models.CharField(max_length=200)
    email = models.EmailField(unique=True)
    mobile = models.CharField(max_length=20)
    experience = models.CharField(max_length=100)
    resume = models.FileField(upload_to='resumes/')
    processing_mode = models.CharField(max_length=20, choices=PROCESSING_MODES, default=PROCESSING_STANDARD)
    extracted_text = models.TextField(blank=True)
    extracted_skills = models.TextField(blank=True, help_text="Comma-separated extracted skills")
    extraction_status = models.CharField(max_length=20, default='pending', choices=[
        ('pending', 'Pending'), ('completed', 'Completed'),
        ('needs_review', 'Needs review'), ('failed', 'Failed'),
    ])
    extraction_method = models.CharField(max_length=40, blank=True)
    extraction_error = models.CharField(max_length=500, blank=True)
    extraction_warnings = models.TextField(blank=True)
    extraction_page_count = models.PositiveSmallIntegerField(null=True, blank=True)
    extraction_processed_at = models.DateTimeField(null=True, blank=True)
    reviewed_text = models.TextField(blank=True)
    structured_extraction = models.JSONField(default=dict, blank=True)
    reviewed_profile = models.JSONField(default=dict, blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    uploaded_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)

    def __str__(self):
        return self.name

    def get_skills_list(self):
        return [s.strip().lower() for s in self.extracted_skills.split(',') if s.strip()]

    def get_effective_text(self):
        """Reviewed text is authoritative when present; retain the original separately."""
        return self.reviewed_text.strip() or self.extracted_text

    class Meta:
        ordering = ['-uploaded_at']


class MatchResult(models.Model):
    RECOMMENDATION_CHOICES = [
        ('highly_recommended', 'Highly Recommended'),
        ('recommended', 'Recommended'),
        ('not_suitable', 'Not Suitable'),
    ]

    candidate = models.ForeignKey(Candidate, on_delete=models.CASCADE, related_name='match_results')
    job = models.ForeignKey(JobDescription, on_delete=models.CASCADE, related_name='match_results')
    match_percentage = models.FloatField(default=0.0)
    matching_skills = models.TextField(blank=True)
    missing_skills = models.TextField(blank=True)
    recommendation = models.CharField(max_length=30, choices=RECOMMENDATION_CHOICES, default='not_suitable')
    # Additive and optional so historic results keep their existing scores and
    # can be identified as baseline results when this object is empty.
    explanation = models.JSONField(default=dict, blank=True)
    matched_at = models.DateTimeField(auto_now_add=True)

    def get_matching_skills_list(self):
        return [s.strip() for s in self.matching_skills.split(',') if s.strip()]

    def get_missing_skills_list(self):
        return [s.strip() for s in self.missing_skills.split(',') if s.strip()]

    class Meta:
        ordering = ['-match_percentage']
        unique_together = ['candidate', 'job']
