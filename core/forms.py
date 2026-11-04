from django import forms
from pathlib import Path
from django.contrib.auth.forms import AuthenticationForm
from .models import JobDescription, Candidate


class LoginForm(AuthenticationForm):
    username = forms.CharField(widget=forms.TextInput(attrs={
        'class': 'form-control', 'placeholder': 'Username'
    }))
    password = forms.CharField(widget=forms.PasswordInput(attrs={
        'class': 'form-control', 'placeholder': 'Password'
    }))


class JobDescriptionForm(forms.ModelForm):
    class Meta:
        model = JobDescription
        fields = ['title', 'required_skills', 'experience', 'description']
        widgets = {
            'title': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'e.g. Senior Python Developer'}),
            'required_skills': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'e.g. python, django, postgresql, docker, rest api'
            }),
            'experience': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'e.g. 3-5 years'}),
            'description': forms.Textarea(attrs={'class': 'form-control', 'rows': 5}),
        }
        labels = {
            'required_skills': 'Required Skills (comma-separated)',
        }


class CandidateForm(forms.ModelForm):
    MAX_UPLOAD_BYTES = 10 * 1024 * 1024

    def clean_resume(self):
        uploaded = self.cleaned_data.get('resume')
        if not uploaded or (self.instance.pk and 'resume' not in self.files):
            return uploaded
        suffix = Path(uploaded.name).suffix.lower()
        if suffix not in {'.pdf', '.docx', '.jpg', '.jpeg', '.png'}:
            raise forms.ValidationError('Upload a PDF, DOCX, JPG, JPEG, or PNG document.')
        if uploaded.size <= 0:
            raise forms.ValidationError('The selected file is empty.')
        if uploaded.size > self.MAX_UPLOAD_BYTES:
            raise forms.ValidationError('Files must be 10 MB or smaller.')
        # Check common file signatures rather than trusting the extension alone.
        try:
            head = uploaded.read(16)
            uploaded.seek(0)
        except Exception:
            raise forms.ValidationError('The selected file could not be read.')
        valid = (
            (suffix == '.pdf' and head.startswith(b'%PDF-')) or
            (suffix == '.docx' and head.startswith(b'PK\x03\x04')) or
            (suffix in {'.jpg', '.jpeg'} and head.startswith(b'\xff\xd8\xff')) or
            (suffix == '.png' and head.startswith(b'\x89PNG\r\n\x1a\n'))
        )
        if not valid:
            raise forms.ValidationError('The file content does not match its extension.')
        return uploaded

    class Meta:
        model = Candidate
        fields = ['name', 'email', 'mobile', 'experience', 'resume', 'processing_mode']
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Full Name'}),
            'email': forms.EmailInput(attrs={'class': 'form-control', 'placeholder': 'candidate@example.com'}),
            'mobile': forms.TextInput(attrs={'class': 'form-control', 'placeholder': '+1 202 555 0147'}),
            'experience': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'e.g. 2 years'}),
            'resume': forms.FileInput(attrs={'class': 'form-control', 'accept': '.pdf,.docx,.jpg,.jpeg,.png'}),
            'processing_mode': forms.Select(attrs={'class': 'form-select'}),
        }
        help_texts = {
            'processing_mode': 'Choose TrOCR only for English handwriting. The optional Base checkpoint must be installed locally (about 1.33 GB); review every transcription.',
        }


class StructuredResumeProfileForm(forms.Form):
    full_name = forms.CharField(required=False, max_length=200, label='Full name', widget=forms.TextInput(attrs={'class': 'form-control'}))
    email = forms.EmailField(required=False, max_length=254, label='Email', widget=forms.EmailInput(attrs={'class': 'form-control'}))
    phone = forms.CharField(required=False, max_length=60, label='Phone', widget=forms.TextInput(attrs={'class': 'form-control'}))
    education = forms.CharField(required=False, max_length=12000, label='Education', widget=forms.Textarea(attrs={'class': 'form-control', 'rows': 3}))
    skills = forms.CharField(required=False, max_length=12000, label='Skills', widget=forms.Textarea(attrs={'class': 'form-control', 'rows': 3}))
    projects = forms.CharField(required=False, max_length=12000, label='Projects', widget=forms.Textarea(attrs={'class': 'form-control', 'rows': 3}))
    work_experience = forms.CharField(required=False, max_length=12000, label='Work experience', widget=forms.Textarea(attrs={'class': 'form-control', 'rows': 4}))
    internships = forms.CharField(required=False, max_length=12000, label='Internships', widget=forms.Textarea(attrs={'class': 'form-control', 'rows': 3}))
    certifications = forms.CharField(required=False, max_length=12000, label='Certifications', widget=forms.Textarea(attrs={'class': 'form-control', 'rows': 3}))
    achievements = forms.CharField(required=False, max_length=12000, label='Achievements', widget=forms.Textarea(attrs={'class': 'form-control', 'rows': 3}))
    dates = forms.CharField(required=False, max_length=12000, label='Dates and durations', widget=forms.Textarea(attrs={'class': 'form-control', 'rows': 3}))
class MatchForm(forms.Form):
    job = forms.ModelChoiceField(
        queryset=JobDescription.objects.all(),
        empty_label="Select a Job Description",
        widget=forms.Select(attrs={'class': 'form-select'})
    )
