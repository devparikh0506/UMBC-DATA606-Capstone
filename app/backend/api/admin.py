from django.contrib import admin
from .models import Subject, EvaluationRun, TrialData


@admin.register(Subject)
class SubjectAdmin(admin.ModelAdmin):
    list_display = ['subject_id', 'created_at']
    search_fields = ['subject_id']


@admin.register(EvaluationRun)
class EvaluationRunAdmin(admin.ModelAdmin):
    list_display = ['subject', 'run_id', 'session_id', 'n_trials', 'created_at']
    list_filter = ['subject', 'created_at']
    search_fields = ['subject__subject_id', 'run_id']


@admin.register(TrialData)
class TrialDataAdmin(admin.ModelAdmin):
    list_display = ['run', 'trial_index', 'ground_truth']
    list_filter = ['run__subject', 'run', 'ground_truth']
    search_fields = ['run__subject__subject_id', 'run__run_id']

