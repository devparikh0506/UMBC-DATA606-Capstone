# UMBC-DATA606-Capstone (Notebooks)

This directory contains the Jupyter notebooks used for the **Brain2Action - Turning Brain Signals into Actions** capstone project.  
The notebooks serve as an interactive workspace for data exploration, preprocessing, model training, and evaluation.

---

## 📂 Contents
- [**preliminary_data_analysis.ipynb**](./preliminary_data_analysis.ipynb)
  - **Purpose:** Initial exploration of the BCI Competition IV 2b dataset.
  - **Key Steps:** Loading GDF files, inspecting signal properties, visualizing raw EEG waveforms, extracting event markers, and analyzing trial distributions per subject.

- [**exploratory_data_analysis.ipynb**](./exploratory_data_analysis.ipynb)
  - **Purpose:** In-depth exploratory data analysis (EDA) of the BCI Competition IV 2b dataset to identify neural signatures of motor imagery.
  - **Key Steps:**
    - **Data Loading & Preprocessing:** Implemented a robust `EEGDataLoader` class for handling GDF files, applying filters, and creating epochs.
    - **Dataset Overview:** Analyzed trial distribution and confirmed class balance between left and right motor imagery tasks.
    - **ERP Analysis:** Visualized and compared Event-Related Potentials (ERPs) for left and right hand imagery, including topographic maps of the differences.
    - **Time-Frequency Analysis:** Investigated Event-Related Desynchronization (ERD) in the alpha and beta bands to observe power changes over the motor cortex.
    - **Model Input Visualization:** Plotted single-trial EEG data to understand the input structure for machine learning models.

- [**training.ipynb**](./training.ipynb)
  - **Purpose:** Train and compare EEGNet models using subject-dependent and subject-independent strategies.
  - **Key Steps:**
    - **Subject-Dependent Training:** Train separate models per subject using 3-fold cross-validation.
    - **Subject-Independent Training:** Leave-One-Subject-Out (LOSO) cross-validation for generalization assessment.
    - **Model Comparison:** Statistical analysis and visualization of performance differences between strategies.
    - **Model Storage:** Best models saved to `app/resources/models/{subject_id}/best_model.pth`.

- [**evaluation.ipynb**](./evaluation.ipynb)
  - **Purpose:** Evaluate trained EEGNet models on evaluation runs (04E, 05E) for all subjects.
  - **Key Steps:**
    - Load trained models for each subject.
    - Evaluate on evaluation runs (04E, 05E) using the same preprocessing as training.
    - Calculate accuracy metrics, confusion matrices, and classification reports.
    - Generate visualizations: accuracy bar charts, confusion matrices, run comparisons, and statistical summaries.
    - Save results to CSV for further analysis.

---

## ⚙️ Requirements
Install dependencies before running any notebook:

```bash
pip install -r ../requirements.txt
