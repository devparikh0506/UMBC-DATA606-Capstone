# BCI Backend (Django)

Django backend for BCI real-time prediction application.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run migrations:
```bash
python manage.py migrate
```

3. Load all run data into database (optional but recommended):
```bash
# Load all runs (training + evaluation) for all subjects
python manage.py load_evaluation_data

# Load all runs for specific subject
python manage.py load_evaluation_data --subject B01

# Load only training runs (01T, 02T, 03T)
python manage.py load_evaluation_data --training-only

# Load only evaluation runs (04E, 05E)
python manage.py load_evaluation_data --evaluation-only

# Overwrite existing data
python manage.py load_evaluation_data --overwrite
```

4. Start the ASGI server (required for WebSocket support):
```bash
# Using Daphne (recommended for Channels)
daphne -b 0.0.0.0 -p 8000 bci_app.asgi:application

# OR using uvicorn (alternative)
# uvicorn bci_app.asgi:application --host 0.0.0.0 --port 8000

# Note: Do NOT use 'python manage.py runserver' as it uses WSGI and doesn't support WebSockets
```

The server will run on `http://localhost:8000` with WebSocket support

## API Endpoints

- `GET /api/subjects/` - List all available subjects
- `GET /api/subjects/{subject_id}/runs/` - List all runs (training + evaluation) for a subject
  - Returns: Training runs (01T, 02T, 03T) and Evaluation runs (04E, 05E)
- `GET /api/subjects/{subject_id}/runs/{run_id}/info/` - Get run metadata
  - Supports: All run types (01T, 02T, 03T, 04E, 05E)

## WebSocket

- `ws://localhost:8000/ws/predict/{subject_id}/{run_id}/` - Real-time prediction streaming

## Database Models

- **Subject**: Stores subject information
- **EvaluationRun**: Stores run metadata (both training and evaluation runs)
  - Fields: `run_id`, `session_id`, `session_type` ('T' or 'E'), metadata
  - Supports: Training runs (01T, 02T, 03T) and Evaluation runs (04E, 05E)
- **TrialData**: Stores preprocessed trial data for simulation

## Configuration

- Models are loaded from `app/resources/models/{subject_id}/best_model.pth`
- Data files are loaded from `data/BCICIV_2b_gdf/` (or from database if loaded)
- CORS is configured to allow requests from `http://localhost:5173` (Vite dev server)

## Data Loading

The `load_evaluation_data` management command:
- Loads **all run types**: Training (01T, 02T, 03T) and Evaluation (04E, 05E) GDF files
- Preprocesses the data using the same pipeline as training (filtering 4-40 Hz, notch 50 Hz, epochs [2.5, 7.0] seconds)
- Stores trial data in the database for faster access
- Enables testing on all runs in the frontend without re-reading GDF files each time

**Note:** All runs are now available for testing in the frontend!
