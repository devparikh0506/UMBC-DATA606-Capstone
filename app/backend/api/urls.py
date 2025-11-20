from django.urls import path
from . import views

urlpatterns = [
    path('subjects/', views.list_subjects, name='list_subjects'),
    path('subjects/<str:subject_id>/runs/', views.list_runs, name='list_runs'),
    path('subjects/<str:subject_id>/runs/<str:run_id>/info/', views.run_info, name='run_info'),
]

