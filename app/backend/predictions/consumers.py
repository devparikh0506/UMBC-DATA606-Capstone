"""
WebSocket consumer for streaming BCI predictions.
"""
import json
import asyncio
from channels.generic.websocket import AsyncWebsocketConsumer
from .utils import load_model, load_evaluation_data, create_sliding_windows, predict_window
import torch


class PredictionConsumer(AsyncWebsocketConsumer):
    """WebSocket consumer for real-time prediction streaming."""
    
    async def connect(self):
        self.subject_id = self.scope['url_route']['kwargs']['subject_id']
        self.run_id = self.scope['url_route']['kwargs']['run_id']
        self.room_group_name = f'predict_{self.subject_id}_{self.run_id}'
        
        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name
        )
        
        await self.accept()
        
        # Send connection confirmation
        await self.send(text_data=json.dumps({
            'type': 'connection',
            'status': 'connected',
            'subject_id': self.subject_id,
            'run_id': self.run_id
        }))
        
        # Load model and data in background
        await self.send(text_data=json.dumps({
            'type': 'status',
            'message': 'Loading model and data...'
        }))
        
        try:
            # Run blocking operations in thread pool
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._load_resources)
            
            await self.send(text_data=json.dumps({
                'type': 'status',
                'message': 'Ready to start predictions',
                'n_trials': self.n_trials,
                'n_windows': len(self.window_times)
            }))
            
        except Exception as e:
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': f'Failed to load resources: {str(e)}'
            }))
            await self.close()
    
    def _load_resources(self):
        """Load model and data (blocking operation)."""
        from predictions.utils import get_standardization_stats
        
        # Determine device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load model
        self.model = load_model(self.subject_id, device=self.device)
        
        # Load training session data (try database first, fallback to file)
        eeg_data, labels = load_evaluation_data(self.subject_id, self.run_id, use_db=True)
        
        # Get standardization statistics from training data (if available)
        # This matches training preprocessing: standardize using training stats
        self.stats = get_standardization_stats(self.subject_id)
        
        # Store raw data (will standardize during prediction if stats available)
        self.eeg_data = eeg_data
        self.labels = labels
        self.n_trials = len(eeg_data)
        
        # Create sliding windows (non-overlapping windows matching model input: 1125 time points)
        # Model expects 1125 time points (4.5 seconds at 250 Hz) from training
        windows, window_times = create_sliding_windows(eeg_data, window_size=1125, step=1125)
        self.windows = windows
        self.window_times = window_times
    
    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(
            self.room_group_name,
            self.channel_name
        )
    
    async def receive(self, text_data):
        """Handle messages from client."""
        data = json.loads(text_data)
        message_type = data.get('type')
        
        if message_type == 'start':
            # Get speed multiplier (default 1x = 4s delay, 2x = 2s, 4x = 1s, etc.)
            speed = float(data.get('speed', 1.0))
            await self.start_predictions(speed=speed)
        elif message_type == 'stop':
            await self.stop_predictions()
    
    async def start_predictions(self, speed=1.0):
        """Start streaming predictions.
        
        Args:
            speed: Speed multiplier (1.0 = real-time 4s, 2.0 = 2x faster, 4.0 = 4x faster)
        """
        if not hasattr(self, 'model'):
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': 'Model not loaded'
            }))
            return
        
        self.is_running = True
        self.correct_count = 0
        self.total_count = 0
        self.speed = speed
        
        # Calculate delay: 4 seconds (real-time) divided by speed
        # 1x speed = 4s delay, 2x speed = 2s delay, 4x speed = 1s delay
        self.prediction_delay = 4.0 / speed
        
        await self.send(text_data=json.dumps({
            'type': 'prediction_start',
            'total_windows': len(self.window_times),
            'speed': speed
        }))
        
        # Stream predictions
        await self._stream_predictions()
    
    async def _stream_predictions(self):
        """Stream predictions asynchronously."""
        self.correct_count = 0
        self.total_count = 0
        print(len(self.window_times))
        for i in range(len(self.window_times)):
            if not self.is_running:
                break
            
            window_data = self.windows[i]
            trial_idx, start, center, end = self.window_times[i]
            
            # Get ground truth for this trial (ensure Python int)
            ground_truth = int(self.labels[trial_idx])
            
            # Run prediction in executor (blocking operation)
            # Pass standardization stats to match training preprocessing
            loop = asyncio.get_event_loop()
            prediction, confidence, probabilities = await loop.run_in_executor(
                None,
                predict_window,
                self.model,
                window_data,
                self.device,
                getattr(self, 'stats', None)  # Pass standardization stats if available
            )
            
            # Check if correct (ensure Python bool, not numpy bool)
            correct = bool(int(prediction) == ground_truth)
            if correct:
                self.correct_count += 1
            self.total_count += 1
            
            # Calculate timestamp (center of window in seconds)
            timestamp = float(center) / 250.0  # 250 Hz sampling rate
            
            # Send prediction (ensure all values are JSON-serializable Python types)
            message = {
                'type': 'prediction',
                'timestamp': round(timestamp, 2),
                'prediction': int(prediction),
                'confidence': round(float(confidence), 3),
                'ground_truth': int(ground_truth),
                'correct': bool(correct),  # Explicit bool conversion
                'trial_idx': int(trial_idx),
                'window_idx': int(i),
                'progress': {
                    'current': int(i + 1),
                    'total': int(len(self.window_times))
                }
            }
            
            await self.send(text_data=json.dumps(message))
            
            # Delay based on speed setting (4s / speed)
            # 1x speed = 4s delay, 2x speed = 2s delay, 4x speed = 1s delay
            delay = getattr(self, 'prediction_delay', 4.0)
            await asyncio.sleep(delay)
        
        # Send final results
        if self.is_running:
            accuracy = self.correct_count / self.total_count if self.total_count > 0 else 0
            final_message = {
                'type': 'prediction_complete',
                'accuracy': round(accuracy, 3),
                'correct': self.correct_count,
                'total': self.total_count
            }
            await self.send(text_data=json.dumps(final_message))
    
    async def stop_predictions(self):
        """Stop streaming predictions."""
        self.is_running = False
        await self.send(text_data=json.dumps({
            'type': 'prediction_stopped'
        }))
    
    async def prediction_message(self, event):
        """Handle prediction messages from channel layer."""
        message = event['message']
        await self.send(text_data=json.dumps(message))

