from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
import os
from django.conf import settings
import re
from .models import Subject, EvaluationRun


@api_view(['GET'])
def list_subjects(request):
    """List all available subjects (B01-B09)."""
    # First try to get from database
    db_subjects = list(Subject.objects.all().values_list('subject_id', flat=True))
    
    # Also check file system for models
    models_dir = settings.MODELS_FOLDER
    file_subjects = []
    
    if models_dir and os.path.exists(models_dir):
        try:
            for item in os.listdir(models_dir):
                subject_path = os.path.join(models_dir, item)
                if os.path.isdir(subject_path) and re.match(r'^B\d{2}$', item):
                    model_path = os.path.join(subject_path, 'best_model.pth')
                    if os.path.exists(model_path):
                        file_subjects.append(item)
        except (OSError, PermissionError) as e:
            # Log error but continue
            pass
    
    # Combine and deduplicate
    subjects = sorted(set(db_subjects + file_subjects))
    return Response({'subjects': subjects}, status=status.HTTP_200_OK)


@api_view(['GET'])
def list_runs(request, subject_id):
    """
    List all available training runs for a subject.
    
    Returns:
    - Training runs: 01T, 02T, 03T
    """
    # First try to get from database
    runs = []
    try:
        subject = Subject.objects.get(subject_id=subject_id)
        db_runs = EvaluationRun.objects.filter(subject=subject, session_type='T').values(
            'run_id', 'session_id', 'session_type', 'filename', 'n_trials'
        )
        for run in db_runs:
            runs.append({
                'run_id': run['run_id'],
                'session_id': run['session_id'],
                'session_type': run['session_type'],
                'filename': run['filename'],
                'n_trials': run['n_trials'],
                'type': 'training',
                'source': 'database'
            })
    except Subject.DoesNotExist:
        pass
    
    # Also check file system for training runs (01T, 02T, 03T)
    data_folder = str(settings.DATA_FOLDER)
    
    if os.path.exists(data_folder):
        # Pattern for training runs only
        run_pattern = re.compile(rf'^{re.escape(subject_id)}(\d{{2}})T\.gdf$')
        
        for filename in os.listdir(data_folder):
            match = run_pattern.match(filename)
            if match:
                session_id = match.group(1)
                file_path = os.path.join(data_folder, filename)
                
                if os.path.exists(file_path):
                    run_id = f"{session_id}T"
                    # Check if already in runs list
                    if not any(r.get('run_id') == run_id for r in runs):
                        runs.append({
                            'run_id': run_id,
                            'session_id': session_id,
                            'session_type': 'T',
                            'filename': filename,
                            'type': 'training',
                            'source': 'filesystem'
                        })
    
    # Sort by session_id (01, 02, 03, 04, 05)
    runs.sort(key=lambda x: x['session_id'])
    return Response({'runs': runs}, status=status.HTTP_200_OK)


@api_view(['GET'])
def run_info(request, subject_id, run_id):
    """
    Get metadata for a specific training run.
    
    Supports:
    - Training runs: 01T, 02T, 03T
    """
    # Only allow training runs
    if not run_id.endswith('T'):
        return Response({'error': 'Only training runs (T) are supported'}, 
                       status=status.HTTP_400_BAD_REQUEST)
    
    # Try to get from database first
    try:
        subject = Subject.objects.get(subject_id=subject_id)
        run = EvaluationRun.objects.get(subject=subject, run_id=run_id)
        return Response({
            'run_id': run.run_id,
            'session_id': run.session_id,
            'session_type': run.session_type,
            'type': 'training',
            'n_trials': run.n_trials,
            'n_channels': run.n_channels,
            'n_times': run.n_times,
            'sampling_rate': run.sampling_rate,
            'duration_per_trial': run.duration_per_trial,
            'source': 'database'
        }, status=status.HTTP_200_OK)
    except (Subject.DoesNotExist, EvaluationRun.DoesNotExist):
        pass
    
    # Fallback to loading from file
    from predictions.utils import load_evaluation_data
    try:
        data_info = load_evaluation_data(subject_id, run_id, metadata_only=True, use_db=False)
        data_info.update({
            'run_id': run_id,
            'session_type': 'T',
            'type': 'training',
            'source': 'filesystem'
        })
        return Response(data_info, status=status.HTTP_200_OK)
    except Exception as e:
        return Response(
            {'error': str(e)}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

