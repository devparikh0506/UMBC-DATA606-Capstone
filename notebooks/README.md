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

- **Future Work:**
  - Feature extraction and engineering (e.g., Common Spatial Patterns).
  - Development and training of classification models (e.g., EEGNet, traditional ML).
  - Model evaluation and performance analysis.

---

## ⚙️ Requirements
Install dependencies before running any notebook:

```bash
pip install -r ../requirements.txt
