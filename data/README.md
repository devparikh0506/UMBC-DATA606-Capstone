# UMBC-DATA606-Capstone (Data)

This directory contains the EEG dataset used for the **Brain2Action - Turning Brain Signals into Actions** project.  
Specifically, it includes the **BCI Competition IV Dataset 2b** in `.gdf` format.

---

## 📂 Dataset Overview
- **Dataset:** BCI Competition IV Dataset 2b (2008, Graz University of Technology)  
- **Format:** `.gdf` (General Data Format for biomedical signals)  
- **Subjects:** 9 right-handed subjects  
- **Sessions per subject:** 5  
  - First 3 sessions → Training data (no feedback)  
  - Last 2 sessions → Feedback sessions  
- **Trials:** Motor imagery of the **left hand** and **right hand** guided by visual cues  

---

## 📌 File Naming Convention
Each file follows the pattern:  

`B<subject><session><type>.gdf`  

- **Subject ID:** `01–09` (e.g., `B01`)  
- **Session number:** `01–05`  
- **Type:**  
  - `T` → Training session  
  - `E` → Evaluation session  

**Example:**  
- `B0101T.gdf` → Subject 1, Session 1, Training  
- `B0503E.gdf` → Subject 5, Session 3, Evaluation  

---

## 🧪 Experimental Paradigm
- Subjects sat in front of a monitor and performed **motor imagery tasks**.  
- Visual cues indicated whether to imagine moving the **left hand** or **right hand**.  
- Sessions included both trials with **no feedback** and trials with **online feedback**.  

---

## 🔗 Reference
- **Official dataset:** [BCI Competition IV Dataset 2b](http://www.bbci.de/competition/iv/)  
- **Publication:**  
  R. Leeb, C. Brunner, G. Müller-Putz, A. Schlögl, and G. Pfurtscheller.  
  *“BCI Competition 2008 – Graz data set B.”*  

---

