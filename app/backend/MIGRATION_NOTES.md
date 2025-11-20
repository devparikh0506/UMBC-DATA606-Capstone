# Migration Notes: Load All Runs for Frontend Testing

## Summary

The backend has been updated to load and serve **all runs** (both training and evaluation) for frontend testing.

## Changes Made

### 1. Database Model Updates (`api/models.py`)

**Added `session_type` field to `EvaluationRun` model:**
- Field: `session_type = models.CharField(max_length=1)` 
- Values: 'T' for training runs, 'E' for evaluation runs
- Added properties: `is_training()` and `is_evaluation()` for convenience

**Note:** You need to create and run a migration:
```bash
cd app/backend
python manage.py makemigrations api --name add_session_type_to_runs
python manage.py migrate
```

### 2. Management Command Updates (`api/management/commands/load_evaluation_data.py`)

**Updated to load all run types:**
- Previously: Only loaded evaluation runs (04E, 05E)
- Now: Loads both training (01T, 02T, 03T) and evaluation (04E, 05E) runs

**New command options:**
- `--training-only`: Load only training runs
- `--evaluation-only`: Load only evaluation runs (default behavior before)
- Default: Loads all runs (training + evaluation)

**Usage:**
```bash
# Load all runs for all subjects
python manage.py load_evaluation_data

# Load all runs for specific subject
python manage.py load_evaluation_data --subject B01

# Load only training runs
python manage.py load_evaluation_data --training-only

# Load only evaluation runs
python manage.py load_evaluation_data --evaluation-only

# Overwrite existing data
python manage.py load_evaluation_data --overwrite
```

### 3. API View Updates (`api/views.py`)

**Updated `list_runs()` endpoint:**
- Returns all runs (training and evaluation) for a subject
- Added `session_type`, `type`, and `n_trials` fields to response
- Checks both database and filesystem for runs

**Response format:**
```json
{
  "runs": [
    {
      "run_id": "01T",
      "session_id": "01",
      "session_type": "T",
      "filename": "B0101T.gdf",
      "n_trials": 108,
      "type": "training",
      "source": "database"
    },
    {
      "run_id": "02T",
      "session_id": "02",
      "session_type": "T",
      "filename": "B0102T.gdf",
      "n_trials": 108,
      "type": "training",
      "source": "database"
    },
    {
      "run_id": "03T",
      "session_id": "03",
      "session_type": "T",
      "filename": "B0103T.gdf",
      "n_trials": 108,
      "type": "training",
      "source": "database"
    },
    {
      "run_id": "04E",
      "session_id": "04",
      "session_type": "E",
      "filename": "B0104E.gdf",
      "n_trials": 160,
      "type": "evaluation",
      "source": "database"
    },
    {
      "run_id": "05E",
      "session_id": "05",
      "session_type": "E",
      "filename": "B0105E.gdf",
      "n_trials": 164,
      "type": "evaluation",
      "source": "database"
    }
  ]
}
```

**Updated `run_info()` endpoint:**
- Works for both training and evaluation runs
- Returns session_type and type information

### 4. Utility Function Updates (`predictions/utils.py`)

**Updated `load_evaluation_data()` function:**
- Now supports all run types: 01T, 02T, 03T, 04E, 05E
- Database loading works for all run types (previously only evaluation)
- Better error handling and fallback to file loading

## API Endpoints

### List All Runs for a Subject
```
GET /api/subjects/{subject_id}/runs/
```

**Response:** List of all runs (training and evaluation)

### Get Run Information
```
GET /api/subjects/{subject_id}/runs/{run_id}/info/
```

**Supports:**
- Training runs: 01T, 02T, 03T
- Evaluation runs: 04E, 05E

## Frontend Access

The frontend can now:
1. **List all available runs** for any subject (training + evaluation)
2. **Test on training runs** (01T, 02T, 03T) 
3. **Test on evaluation runs** (04E, 05E)
4. **Differentiate between run types** using the `type` field in the response

## Migration Steps

1. **Create and run the database migration:**
   ```bash
   cd app/backend
   python manage.py makemigrations api --name add_session_type_to_runs
   python manage.py migrate
   ```

2. **Load all runs into database:**
   ```bash
   python manage.py load_evaluation_data --overwrite
   ```

3. **Verify runs are loaded:**
   ```bash
   # Check via Django shell
   python manage.py shell
   >>> from api.models import EvaluationRun
   >>> EvaluationRun.objects.all().values('run_id', 'session_type')
   ```

## Testing

All runs are now available for testing in the frontend:
- **Training runs**: 01T, 02T, 03T (used for model training)
- **Evaluation runs**: 04E, 05E (used for model evaluation)

The frontend can select any run for testing predictions!

