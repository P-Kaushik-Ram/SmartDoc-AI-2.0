from django.contrib import admin
from .models import JobDescription, Candidate, MatchResult


@admin.register(JobDescription)
class JobAdmin(admin.ModelAdmin):
    list_display = ['title', 'experience', 'created_date', 'created_by']
    search_fields = ['title', 'required_skills']
    list_filter = ['created_date']


@admin.register(Candidate)
class CandidateAdmin(admin.ModelAdmin):
    list_display = ['name', 'email', 'mobile', 'experience', 'uploaded_at']
    search_fields = ['name', 'email']


@admin.register(MatchResult)
class MatchResultAdmin(admin.ModelAdmin):
    list_display = ['candidate', 'job', 'match_percentage', 'recommendation', 'matched_at']
    list_filter = ['recommendation']
    ordering = ['-match_percentage']
