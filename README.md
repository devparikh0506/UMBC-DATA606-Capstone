# Brain2Action: Turning Brain Signals into Actions

**BCI Motor Imagery Classification using Deep Learning**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive Brain-Computer Interface (BCI) system for classifying motor imagery from EEG signals using deep learning. This project implements EEGNet architecture to distinguish between left-hand and right-hand motor imagery, with a complete pipeline from data preprocessing to web-based real-time predictions.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Technologies](#technologies)
- [Author](#author)
- [License](#license)

---

## 🧠 Overview

This project focuses on **classifying brain signals (EEG)** recorded during motor imagery tasks using deep learning models. Motor imagery refers to the mental simulation of movement, such as imagining moving the left or right hand, without any physical action.

### Key Objectives

1. **Classification**: Accurately distinguish between left-hand and right-hand motor imagery from EEG signals
2. **Training Comparison**: Compare subject-dependent vs subject-independent training strategies
3. **Real-World Application**: Develop a web-based interface for real-time predictions
4. **Complete Pipeline**: From raw EEG data to actionable predictions

### Research Questions

- How effectively can deep learning models classify motor imagery from EEG signals?
- Which training strategy works better: subject-dependent or subject-independent?
- How can this be integrated into real-world BCI applications?

---

## ✨ Features

- **Deep Learning Model**: EEGNet architecture (1,186 parameters) for efficient EEG classification
- **Dual Training Strategies**: 
  - Subject-Dependent: 3-fold cross-validation per subject
  - Subject-Independent: Leave-One-Subject-Out (LOSO) cross-validation
- **Complete Preprocessing Pipeline**: Bandpass filtering, notch filtering, normalization, epoch extraction
- **Web Application**: React-based interface for real-time predictions
- **Comprehensive Analysis**: EDA, model evaluation, statistical comparison
- **9 Trained Models**: One model per subject (B01-B09)

---

## 📂 Project Structure

```
UMBC-DATA606-Capstone/
├── app/                          # Main application code
│   ├── backend/                  # Django REST API
│   │   ├── api/                  # API endpoints
│   │   ├── predictions/         # WebSocket for real-time predictions
│   │   └── bci_app/             # Django settings
│   ├── frontend/                 # React + TypeScript frontend
│   │   └── src/                  # React components and pages
│   ├── models/                   # Deep learning models
│   │   └── eegnet.py            # EEGNet architecture
│   ├── datasets/                 # Data loading utilities
│   │   ├── data_loader.py       # EEG data loader
│   │   └── dataset.py           # PyTorch datasets
│   ├── scripts/                  # Training scripts
│   │   ├── train.py             # Main training script
│   │   ├── train_subject_dependent.py
│   │   └── train_subject_independent.py
│   ├── utils/                    # Utility functions
│   └── resources/                # Trained models
│       └── models/                # 9 subject-specific models
│
├── data/                         # Dataset
│   └── BCICIV_2b_gdf/           # BCI Competition IV-2b GDF files
│
├── notebooks/                    # Jupyter notebooks
│   ├── exploratory_data_analysis.ipynb
│   ├── preliminary_data_analysis.ipynb
│   └── training.ipynb            # Model training and evaluation
│
├── docs/                         # Documentation
│   ├── proposal.md              # Project proposal
│   ├── training.md               # Training documentation
│   └── README.md
│
├── requirements.txt              # Python dependencies
├── package.json                  # Node.js dependencies (for frontend)
└── README.md                     # This file
```

---

## 📊 Dataset

### BCI Competition IV Dataset 2b

- **Source**: Graz University of Technology, Institute for Knowledge Discovery
- **Subjects**: 9 right-handed subjects (B01-B09)
- **Format**: GDF (General Data Format for biomedical signals)
- **Channels**: 
  - 3 EEG channels: C3, Cz, C4 (motor cortex)
  - 3 EOG channels: Eye movement (artifact detection)
- **Sampling Rate**: 250 Hz
- **Trial Duration**: 4.5 seconds (2.5-7.0s relative to cue)
- **Sessions**: 5 per subject (3 training, 2 evaluation)
- **Total Trials**: ~300-400 per subject
- **Task**: Binary motor imagery (Left hand vs Right hand)

### Data Preprocessing

1. **Bandpass Filtering**: 4-40 Hz (motor imagery frequency bands)
2. **Notch Filtering**: 50 Hz (power line interference)
3. **Channel Selection**: C3, Cz, C4
4. **Epoch Extraction**: [2.5, 7.0] seconds (1,125 samples)
5. **Normalization**: Z-score per channel (from training data)
6. **Trial Validation**: Remove rejected trials

---

## 🚀 Installation

### Prerequisites

- Python 3.8+
- Node.js 16+ (for frontend)
- CUDA-capable GPU (recommended for training)

### Python Environment

```bash
# Clone the repository
git clone https://github.com/wcj365/UMBC-Data-Science-Capstone.git
cd UMBC-DATA606-Capstone

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Frontend Setup

```bash
cd app/frontend
npm install
```

### Backend Setup

```bash
cd app/backend
pip install -r requirements.txt
python manage.py migrate
```

---

## 💻 Usage

### Training Models

#### Subject-Dependent Training

```bash
python app/scripts/train_subject_dependent.py
```

This trains separate models for each subject using 3-fold cross-validation.

#### Subject-Independent Training

```bash
python app/scripts/train_subject_independent.py
```

This trains models using Leave-One-Subject-Out cross-validation.

#### Using Jupyter Notebook

```bash
jupyter notebook notebooks/training.ipynb
```

### Running the Web Application

#### Backend (Django)

```bash
cd app/backend
python manage.py runserver
```

#### Frontend (React)

```bash
cd app/frontend
npm run dev
```

The application will be available at `http://localhost:5173`

### Making Predictions

1. Select a subject (B01-B09)
2. Upload EEG trial data or use sample data
3. Get real-time predictions (Left/Right hand)
4. View confidence scores and detailed results

---

## 📈 Results

### Model Performance

| Metric | Subject-Dependent | Subject-Independent |
|--------|------------------|---------------------|
| **Mean Accuracy** | **63.17%** | 50.42% |
| **Std Deviation** | ± 9.59% | ± 1.30% |
| **Best Subject** | B04 (76.25%) | B08 (53.35%) |
| **Worst Subject** | B02 (52.02%) | B03 (49.32%) |

### Key Findings

- ✅ **Subject-Dependent training shows 25.3% improvement** over Subject-Independent
- ✅ **Statistically significant** (p = 0.0018 < 0.05)
- ✅ **EEGNet architecture** works effectively with only 1,186 parameters
- ✅ **Compact model size**: ~5 KB per model

### Per-Subject Performance

| Subject | Subject-Dependent | Subject-Independent | Improvement |
|---------|------------------|---------------------|-------------|
| B01 | 63.08% | 51.23% | +11.85% |
| B02 | 52.02% | 51.06% | +0.96% |
| B03 | 54.44% | 49.32% | +5.12% |
| B04 | **76.25%** | 49.62% | **+26.63%** |
| B05 | 54.98% | 49.48% | +5.50% |
| B06 | 63.84% | 50.17% | +13.67% |
| B07 | 69.44% | 50.00% | +19.44% |
| B08 | 64.14% | 53.35% | +10.79% |
| B09 | 70.31% | 49.53% | +20.78% |

---

## 🛠 Technologies

### Backend
- **Python 3.8+**
- **PyTorch**: Deep learning framework
- **Django**: REST API backend
- **MNE-Python**: EEG data processing
- **NumPy, Pandas**: Data manipulation
- **Scikit-learn**: Model evaluation

### Frontend
- **React**: UI framework
- **TypeScript**: Type-safe JavaScript
- **Tailwind CSS**: Styling
- **Vite**: Build tool

### Machine Learning
- **EEGNet**: Compact CNN architecture (Lawhern et al., 2018)
- **Adam Optimizer**: Learning rate 0.001
- **Cross-Entropy Loss**: Classification loss
- **Early Stopping**: Patience = 50 epochs

---

## 📚 Documentation

- [Project Proposal](docs/proposal.md)
- [Training Documentation](docs/training.md)
- [Data README](data/README.md)
- [Notebooks README](notebooks/README.md)
- [App README](app/README.md)

---

## 👤 Author

**Dev Parikh**

- GitHub: [@wcj365](https://github.com/wcj365)
- LinkedIn: [devparikh0506](https://www.linkedin.com/in/devparikh0506/)
- Project: [UMBC Data Science Capstone](https://github.com/wcj365/UMBC-Data-Science-Capstone)

**Institution**: UMBC (University of Maryland, Baltimore County)  
**Program**: Data Science Master's Degree  
**Advisor**: Dr. Chaojie (Jay) Wang

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Dataset**: BCI Competition IV Dataset 2b (Graz University of Technology)
- **Architecture**: EEGNet (Lawhern et al., 2018)
- **Institution**: UMBC Data Science Program
- **Advisor**: Dr. Chaojie (Jay) Wang

---

## 📝 Citation

If you use this project in your research, please cite:

```bibtex
@misc{brain2action2024,
  title={Brain2Action: Turning Brain Signals into Actions},
  author={Parikh, Dev},
  year={2024},
  institution={UMBC},
  note={Data Science Capstone Project}
}
```

---

## 🔮 Future Work

- [ ] Expand to more subjects (20+)
- [ ] Multi-class classification (more movement types)
- [ ] Real-time streaming pipeline
- [ ] Integration with robotic systems
- [ ] Mobile EEG device support
- [ ] Transfer learning approaches
- [ ] Attention mechanisms in architecture

---

## 📞 Contact

For questions or collaborations, please open an issue on GitHub or contact via LinkedIn.

---

**Note**: This project is part of the UMBC Data Science Master's Degree Capstone requirement.
