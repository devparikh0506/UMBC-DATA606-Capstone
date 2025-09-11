# UMBC DATA 606 Capstone project proposal (Brain2Action - Turning Brain Signals into Actions)
 
## 1. Title and Author

- Project Title: **Brain2Action - Turning Brain Signals into Actions**
- Prepared for UMBC Data Science Master Degree Capstone by Dr Chaojie (Jay) Wang
- Author : **Dev Parikh** 
- GitHub repository: https://github.com/wcj365/UMBC-Data-Science-Capstone/tree/main
- LinkedIn profile: https://www.linkedin.com/in/devparikh0506/
    
## 2. Background

### 🧠 About the Project  
This project focuses on **classifying brain signals (EEG)** recorded during *motor imagery* tasks using deep learning models.  
Motor imagery refers to the mental simulation of a movement, such as imagining moving the **left** or **right hand**, without any physical action.  

The dataset used is the **BCI Competition IV Dataset 2b**, which includes EEG recordings from **9 subjects**.  
- Each trial shows a visual cue (left or right arrow).  
- The subject imagines moving the corresponding hand.  
- EEG is recorded from **three bipolar electrodes (C3, Cz, C4)** at **250 Hz**.  
- Three EOG channels are also included for artifact detection.  

The main goal of this project is to build **deep learning model** that can accurately distinguish between left-hand and right-hand motor imagery. To make the work **interactive and accessible**, classification results will be showcased through a **web-based simulation built with React**.  

---

### 🌍 Project Need  
Motor imagery classification is at the core of **Brain Computer Interfaces (BCIs)**.  
In this project, the focus is on **predicting and classifying brain signals** when a person imagines moving their left or right hand. Accurate prediction of these signals is a crucial step toward translating **thoughts into digital commands**.  

Why does this matter?  
- It shows how **deep learning** can uncover subtle patterns in noisy brain activity.  
- It proves that accurate prediction is possible, even with complex and subject-specific EEG data.  
- Most importantly, the broader vision is **life-changing**:  
  - By integrating this prediction pipeline with **machines or robots**, people with disabilities could perform tasks that normally require physical movement.  
  - Imagine controlling a **robotic arm, a wheelchair, or smart home devices**; not with hands, but with thoughts.  

By combining **deep learning models** with an **interactive React simulation**, this project bridges the gap between research and application. It demonstrates how brain signal prediction can move from theory into **real-world solutions that improve accessibility and independence**.  

---

### ❓ Research Questions  
1. **Prediction Accuracy** – How effectively can deep learning models predict and classify left-hand versus right-hand motor imagery from EEG signals in the BCI Competition IV Dataset 2b?  
2. **Model Robustness** – How well do the models handle challenges such as noise, variability between subjects, and limited electrode inputs?  
3. **Interactive Simulation** – How can the classification output be integrated into a React application to provide a clear and engaging demonstration of brain signal prediction?  
4. **Broader Impact** – How can this classification pipeline be extended and integrated with machines or robotic systems to support people with disabilities in performing everyday tasks through thought?

## 3. Data

### 📌 Data Sources  
The dataset used for this project is the [**BCI Competition IV Dataset 2b**](https://www.bbci.de/competition/iv/), provided by the Institute for Knowledge Discovery, Graz University of Technology.  
- EEG recordings were collected during **motor imagery experiments** where subjects imagined left-hand or right-hand movements.  
- Data is available in **GDF (General Data Format for biomedical signals)** files.  
- Each subject has multiple sessions: some without feedback (training) and some with feedback (evaluation).  

---

### 📦 Data Size  
- Each subject’s recordings are stored in multiple GDF files (approx. **5–10 MB per file**).  
- With 9 subjects and 5 sessions each, the dataset totals about **300–400 MB**.  

---

### 📐 Data Shape  
- Sampling frequency: **250 Hz** (250 samples per second).  
- Each recording contains **6 channels**:  
  - **3 EEG channels**: C3, Cz, C4  
  - **3 EOG channels**: eye movement (for artifact detection, not classification)  
- Each subject completed:  
  - 2 training sessions × 6 runs × 120 trials ≈ 240 trials  
  - 3 feedback sessions × 4 runs × 80 trials ≈ 240 trials  
- **Total per subject** ≈ 480 trials  
- **Across all subjects (9)** ≈ 4,320 trials  
- **Trial duration**: Each trial lasts about **6–8 seconds**  
  - Cue: 1.25 seconds  
  - Motor imagery period: 4 seconds  
  - Pause between trials: 1.5–2.5 seconds  
 

---

### ⏳ Time Period  
The dataset was collected during the **2008 BCI Competition IV** period.  
Each recording session lasted about **10–15 minutes**, and subjects participated in multiple sessions across different days.  

---

### 🧾 What does each row represent?  
Each **row (trial)** represents a **single motor imagery attempt** by a subject:  
- The subject saw a cue (left or right arrow).  
- They imagined moving the corresponding hand for **4 seconds**.  
- Each trial lasted about **6–8 seconds total** including cue, imagery, and pause.  
- The EEG + EOG signals recorded during that trial form one labeled instance.  

---

### 📚 Data Dictionary  

| Column / Variable | Data Type | Definition | Potential Values |
|-------------------|-----------|------------|------------------|
| `EEG_C3`          | Float     | EEG signal recorded at electrode C3 | Continuous values in µV |
| `EEG_Cz`          | Float     | EEG signal recorded at electrode Cz | Continuous values in µV |
| `EEG_C4`          | Float     | EEG signal recorded at electrode C4 | Continuous values in µV |
| `EOG_1`           | Float     | EOG signal (horizontal eye movement) | Continuous values in µV |
| `EOG_2`           | Float     | EOG signal (vertical eye movement) | Continuous values in µV |
| `EOG_3`           | Float     | EOG signal (eye rotation/blinks) | Continuous values in µV |
| `Label`           | String   | Motor imagery class label | Left, Right |
| `Trial_ID`        | Integer   | Unique trial identifier | e.g., 0-119 per subject |
| `Time`            | Float     | Time index of the signal (in seconds) | 0.0 – duration of trial |

---

### 🎯 Target Variable  
- **`Label`** → Binary classification task  
  - `1` = Left-hand motor imagery  
  - `2` = Right-hand motor imagery  

---

### 🔎 Feature Variables  
The following columns will be used as **features/predictors** for the ML model:  
- `EEG_C3`, `EEG_Cz`, `EEG_C4` (primary EEG channels of interest)  
- `Time` dimension (used for temporal modeling in CNNs/RNNs)  
- **Note**: EOG channels (`EOG_1`, `EOG_2`, `EOG_3`) are recorded for artifact removal but will **not** be used directly as features.  

---

✅ This structured dataset will allow us to train deep learning models to **predict the motor imagery class (left vs right hand)** based on EEG patterns.  

