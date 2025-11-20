"""
Django management command to load training run data from GDF files into the database.
"""
from django.core.management.base import BaseCommand
from api.models import Subject, EvaluationRun, TrialData
from predictions.utils import load_evaluation_data, EEGDataLoader
from django.conf import settings
import numpy as np
import json


class Command(BaseCommand):
    help = 'Load training run data from GDF files into the database'

    def add_arguments(self, parser):
        parser.add_argument(
            '--subject',
            type=str,
            help='Load data for a specific subject (e.g., B01). If not specified, loads all subjects.',
        )
        parser.add_argument(
            '--overwrite',
            action='store_true',
            help='Overwrite existing data if it already exists',
        )

    def handle(self, *args, **options):
        subject_filter = options.get('subject')
        overwrite = options.get('overwrite', False)
        
        # Get all files
        data_loader = EEGDataLoader(
            data_folder=str(settings.DATA_FOLDER),
            raw_channels=['EEG:Cz', 'EEG:C3', 'EEG:C4'],
            channel_fun=lambda x: x.replace('EEG:', ''),
            filtered=True,
            apply_standardization=False,
            filter_range=(4, 40),
            apply_notch=True,
            notch_freq=50
        )
        all_files = data_loader._scan_data_dir_()
        
        # Only load training runs (T)
        files_to_process = [f for f in all_files if f['session_type'] == 'T']
        
        if subject_filter:
            files_to_process = [f for f in files_to_process if f['subject_id'] == subject_filter]
        
        self.stdout.write(f'Found {len(files_to_process)} training runs to process (01T, 02T, 03T)')
        
        processed = 0
        skipped = 0
        errors = 0
        
        for file_info in files_to_process:
            subject_id = file_info['subject_id']
            session_id = file_info['session_id']
            session_type = file_info['session_type']  # 'T' or 'E'
            run_id = f"{session_id}{session_type}"
            
            try:
                # Check if already exists
                subject, _ = Subject.objects.get_or_create(subject_id=subject_id)
                run_exists = EvaluationRun.objects.filter(subject=subject, run_id=run_id).exists()
                
                if run_exists and not overwrite:
                    self.stdout.write(
                        self.style.WARNING(f'Skipping {subject_id}/{run_id} (already exists, use --overwrite to replace)')
                    )
                    skipped += 1
                    continue
                
                # Load data
                self.stdout.write(f'Loading training run: {subject_id}/{run_id}...')
                
                try:
                    eeg_data, labels = load_evaluation_data(subject_id, run_id, metadata_only=False, use_db=False)
                    metadata = load_evaluation_data(subject_id, run_id, metadata_only=True, use_db=False)
                except Exception as e:
                    self.stdout.write(
                        self.style.WARNING(f'  Warning: Could not load from file, trying direct load: {str(e)}')
                    )
                    # Try loading directly using data_loader
                    data = data_loader.get_data(subject_id, session_id, session_type)
                    if data['epochs'] is None or len(data['epochs']) == 0:
                        raise ValueError(f"No epochs found for {subject_id}/{run_id}")
                    
                    epochs = data['epochs']
                    eeg_data = epochs.get_data(copy=False)  # (n_trials, n_channels, n_times)
                    labels = epochs.events[:, -1] - 1  # Convert to 0-indexed
                    
                    metadata = {
                        'n_trials': len(eeg_data),
                        'n_channels': eeg_data.shape[1],
                        'n_times': eeg_data.shape[2],
                        'sampling_rate': epochs.info['sfreq'],
                        'duration_per_trial': eeg_data.shape[2] / epochs.info['sfreq'],
                    }
                
                # Create or update run
                run, created = EvaluationRun.objects.update_or_create(
                    subject=subject,
                    run_id=run_id,
                    defaults={
                        'session_id': session_id,
                        'session_type': session_type,
                        'filename': file_info['filename'],
                        'n_trials': metadata['n_trials'],
                        'n_channels': metadata['n_channels'],
                        'n_times': metadata['n_times'],
                        'sampling_rate': metadata['sampling_rate'],
                        'duration_per_trial': metadata['duration_per_trial'],
                    }
                )
                
                # Delete existing trials if overwriting
                if overwrite and run_exists:
                    TrialData.objects.filter(run=run).delete()
                
                # Store trial data
                self.stdout.write(f'  Storing {len(eeg_data)} trials...')
                trial_objects = []
                for trial_idx, (trial_data, label) in enumerate(zip(eeg_data, labels)):
                    trial = TrialData(
                        run=run,
                        trial_index=trial_idx,
                        ground_truth=int(label),
                    )
                    trial.set_data(trial_data)  # Convert numpy array to JSON
                    trial_objects.append(trial)
                
                # Bulk create trials (batch size for performance)
                batch_size = 100
                for i in range(0, len(trial_objects), batch_size):
                    batch = trial_objects[i:i + batch_size]
                    TrialData.objects.bulk_create(batch, ignore_conflicts=True)
                
                self.stdout.write(
                    self.style.SUCCESS(f'✓ Successfully loaded {subject_id}/{run_id}: {len(eeg_data)} trials')
                )
                processed += 1
                
            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(f'✗ Error loading {subject_id}/{run_id}: {str(e)}')
                )
                import traceback
                self.stdout.write(self.style.ERROR(traceback.format_exc()))
                errors += 1
        
        # Summary
        self.stdout.write('\n' + '='*60)
        self.stdout.write(self.style.SUCCESS('Summary:'))
        self.stdout.write(f'  Total training runs found: {len(files_to_process)}')
        self.stdout.write(f'  Processed: {processed}')
        self.stdout.write(f'  Skipped: {skipped}')
        self.stdout.write(f'  Errors: {errors}')
        self.stdout.write('='*60)
        self.stdout.write('\nNote: Training runs are now available for testing in the frontend!')

