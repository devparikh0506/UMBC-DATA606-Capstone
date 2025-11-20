# BCI Real-time Prediction Application

A full-stack web application for real-time BCI (Brain-Computer Interface) motor imagery prediction using EEGNet models.

## Architecture

- **Backend**: Django REST API + Django Channels (WebSocket) for real-time prediction streaming
- **Frontend**: React + Vite with TypeScript for interactive visualization
- **Models**: Subject-specific EEGNet models trained on BCI Competition IV 2b dataset

## Project Structure

```
app/
├── backend/          # Django backend application
│   ├── bci_app/     # Django project settings
│   ├── api/         # REST API endpoints
│   ├── predictions/ # WebSocket consumers and model utilities
│   └── requirements.txt
├── frontend/        # React + Vite frontend
│   ├── src/
│   │   ├── components/  # UI components
│   │   ├── pages/      # Page components
│   │   └── services/   # API and WebSocket clients
│   └── package.json
└── resources/
    └── models/      # Trained EEGNet models (B01-B09)
```

## Quick Start

### Backend Setup

1. Navigate to backend directory:
```bash
cd app/backend
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run migrations:
```bash
python manage.py migrate
```

4. Start the server:
```bash
python manage.py runserver
```

Backend will run on `http://localhost:8000`

### Frontend Setup

1. Navigate to frontend directory:
```bash
cd app/frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

Frontend will run on `http://localhost:5173`

## Usage

1. Open `http://localhost:5173` in your browser
2. Select a subject (B01-B09)
3. Choose an evaluation run (04E or 05E)
4. Click "Start Predictions" to begin real-time streaming
5. Watch the animated smiley move based on predictions
6. View final results with accuracy metrics

## Features

- **Subject Selection**: Browse available subjects with trained models
- **Run Selection**: Choose from evaluation sessions
- **Real-time Predictions**: Stream predictions using sliding window approach
- **Visual Feedback**: Animated smiley that moves left/right based on predictions
- **Accuracy Tracking**: Real-time correctness indicators and final statistics
- **Responsive Design**: Works on desktop and mobile devices

## Technical Details

- **Sliding Window**: 4-second windows, sliding by 1 second
- **Prediction Frequency**: One prediction per second
- **Model Input**: 3 channels (Cz, C3, C4) × 1000 time points
- **Output**: Binary classification (Left hand vs Right hand motor imagery)

## API Documentation

### REST Endpoints

- `GET /api/subjects/` - List all subjects
- `GET /api/subjects/{subject_id}/runs/` - List runs for a subject
- `GET /api/subjects/{subject_id}/runs/{run_id}/info/` - Get run metadata

### WebSocket

- `ws://localhost:8000/ws/predict/{subject_id}/{run_id}/` - Real-time prediction stream

## Requirements

- Python 3.8+
- Node.js 18+
- PyTorch (for model inference)
- MNE (for EEG data processing)
