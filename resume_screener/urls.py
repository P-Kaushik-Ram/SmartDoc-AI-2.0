from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from core import views
from core import workspaces

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', workspaces.home, name='home'),
    path('resume-analyzer/', workspaces.resume_analyzer, name='resume_analyzer'),
    path('document-studio/', workspaces.document_studio, name='document_studio'),
    path('about/', workspaces.about, name='about'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('assistant/', views.ai_document_assistant, name='ai_document_assistant'),

    # Jobs
    path('jobs/', views.job_list, name='job_list'),
    path('jobs/create/', views.job_create, name='job_create'),
    path('jobs/<int:pk>/', views.job_detail, name='job_detail'),
    path('jobs/<int:pk>/edit/', views.job_edit, name='job_edit'),
    path('jobs/<int:pk>/delete/', views.job_delete, name='job_delete'),
    path('jobs/<int:job_pk>/match-all/', views.match_all_candidates, name='match_all_candidates'),

    # Candidates
    path('candidates/', views.candidate_list, name='candidate_list'),
    path('candidates/upload/', views.candidate_create, name='candidate_create'),
    path('candidates/<int:pk>/', views.candidate_detail, name='candidate_detail'),
    path('candidates/<int:pk>/resume-file/', views.candidate_resume_file, name='candidate_resume_file'),
    path('candidates/<int:pk>/edit/', views.candidate_edit, name='candidate_edit'),
    path('candidates/<int:pk>/delete/', views.candidate_delete, name='candidate_delete'),
    path('candidates/<int:pk>/review/', views.save_document_review, name='save_document_review'),
    path('candidates/<int:pk>/structured-profile/', views.save_structured_profile, name='save_structured_profile'),
    path('candidates/<int:pk>/retry-extraction/', views.retry_candidate_extraction, name='retry_candidate_extraction'),
    path('candidates/<int:candidate_pk>/match/', views.run_match, name='run_match'),

    # Rankings & Export
    path('rankings/', views.ranking, name='ranking'),
    path('export/excel/', views.export_excel, name='export_excel'),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
