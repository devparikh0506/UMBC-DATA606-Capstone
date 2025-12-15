# Brain2Action: Turning Brain Signals into Actions
## BCI Motor Imagery Classification using Deep Learning

**Dev Parikh**  
**Fall 2024**  
**UMBC DATA 606 Capstone Project**

---

## Links

- **YouTube Video**: [Brain2Action: Turning Brain Signals into Actions](https://www.youtube.com/watch?v=-q0o-iK1lVw)
- **Presentation Slides**: [Brain2Action_Turning_Brain_Signals_into_Actions.pptx](./Brain2Action_Turning_Brain_Signals_into_Actions.pptx)
- **GitHub Repository**: [https://github.com/wcj365/UMBC-Data-Science-Capstone](https://github.com/wcj365/UMBC-Data-Science-Capstone)

---

## Background

Brain-Computer Interfaces (BCIs) represent a revolutionary technology that enables direct communication between the human brain and external devices. Motor imagery, the mental simulation of movement without physical execution, has emerged as a promising paradigm for BCI applications. This project focuses on classifying motor imagery signals from electroencephalography (EEG) data to distinguish between left-hand and right-hand movement intentions.

The ability to accurately decode motor imagery from EEG signals has profound implications for:
- **Assistive Technologies**: Enabling individuals with motor disabilities to control prosthetic limbs, wheelchairs, or computer interfaces
- **Neurorehabilitation**: Supporting stroke recovery and motor function restoration
- **Human-Robot Interaction**: Creating intuitive control systems for robotic devices
- **Gaming and Entertainment**: Developing immersive brain-controlled experiences

This capstone project implements a complete pipeline from raw EEG data preprocessing to real-time prediction, comparing subject-dependent and subject-independent training strategies to determine the most effective approach for motor imagery classification.

---

## Description of Data Sources

### BCI Competition IV Dataset 2b

The dataset used in this project is the **BCI Competition IV Dataset 2b**, provided by the Institute for Knowledge Discovery at Graz University of Technology. This dataset is a well-established benchmark in the BCI research community and has been used extensively for motor imagery classification studies.

**Dataset Characteristics:**
- **Source**: Graz University of Technology, Institute for Knowledge Discovery
- **Collection Period**: 2008 (BCI Competition IV)
- **Subjects**: 9 right-handed subjects (B01-B09)
- **Data Format**: GDF (General Data Format for biomedical signals)
- **Total Dataset Size**: Approximately 300-400 MB

**Recording Setup:**
- **Sampling Rate**: 250 Hz (250 samples per second)
- **Channels**: 6 channels total
  - **3 EEG channels**: C3, Cz, C4 (motor cortex electrodes)
  - **3 EOG channels**: Eye movement channels (for artifact detection)
- **Session Structure**: 5 sessions per subject
  - 3 training sessions (without feedback)
  - 2 evaluation sessions (with feedback)
- **Trial Structure**:
  - Each trial: ~6-8 seconds total duration
  - Cue presentation: 1.25 seconds
  - Motor imagery period: 4 seconds
  - Inter-trial pause: 1.5-2.5 seconds
- **Total Trials**: Approximately 300-400 trials per subject (~4,320 trials across all subjects)

**Task Design:**
- **Binary Classification**: Left-hand vs. Right-hand motor imagery
- **Visual Cue**: Arrow pointing left or right
- **Subject Instructions**: Imagine moving the corresponding hand without physical movement
- **Trial Validation**: Trials marked as rejected (due to artifacts) were excluded from analysis

---

## Data Elements

### Raw Data Structure

Each GDF file contains:
- **Continuous EEG/EOG signals**: Multi-channel time series data
- **Event markers**: Temporal markers indicating trial start, motor imagery type, and trial rejection
- **Metadata**: Subject ID, session information, sampling rate, channel names

### Preprocessed Data Elements

After preprocessing, each trial is represented as:

| Element | Description | Dimensions/Values |
|---------|------------|------------------|
| **EEG Signal (C3)** | Left motor cortex activity | 1,125 samples (4.5 seconds × 250 Hz) |
| **EEG Signal (Cz)** | Central motor cortex activity | 1,125 samples |
| **EEG Signal (C4)** | Right motor cortex activity | 1,125 samples |
| **Label** | Motor imagery class | Binary: 0 (Left) or 1 (Right) |
| **Subject ID** | Subject identifier | B01-B09 |
| **Session ID** | Session number | 01-05 |
| **Trial ID** | Unique trial identifier | Sequential number |

### Feature Variables

The primary features used for classification are:
- **Temporal EEG signals** from C3, Cz, and C4 channels
- **Time window**: [2.5, 7.0] seconds relative to cue onset (1,125 samples at 250 Hz)
- **Frequency content**: 4-40 Hz band (motor imagery relevant frequencies)

### Target Variable

- **Binary Classification**: 
  - `0` = Left-hand motor imagery
  - `1` = Right-hand motor imagery

### Data Preprocessing Pipeline

1. **Bandpass Filtering**: 4-40 Hz (captures alpha and beta bands associated with motor imagery)
2. **Notch Filtering**: 50 Hz (removes power line interference)
3. **Channel Selection**: C3, Cz, C4 (motor cortex channels)
4. **Epoch Extraction**: [2.5, 7.0] seconds relative to cue (1,125 samples)
5. **Normalization**: Z-score normalization per channel (computed from training data)
6. **Trial Validation**: Removal of rejected trials (marked with event code 1023)

---

## Results of Exploratory Data Analysis (EDA)

### Dataset Overview

**Trial Distribution Analysis:**
- The dataset exhibits **balanced class distribution** across all subjects
- Left-hand and right-hand trials are approximately equal in number per subject
- This balance ensures that classification models will not be biased toward one class

**Subject Variability:**
- Significant inter-subject variability in trial counts (ranging from ~240 to ~480 trials per subject)
- All subjects completed the required sessions, ensuring consistent data collection protocol

### Neural Signatures of Motor Imagery

**Event-Related Potential (ERP) Analysis:**
- ERP analysis revealed distinct voltage patterns for left vs. right hand imagery
- Topographic maps showed differential activation patterns over motor cortex regions
- C3 electrode (left motor cortex) showed stronger activation during right-hand imagery
- C4 electrode (right motor cortex) showed stronger activation during left-hand imagery
- This contralateral activation pattern is consistent with known neurophysiology

**Time-Frequency Analysis:**
- **Event-Related Desynchronization (ERD)** was observed in alpha (8-12 Hz) and beta (13-30 Hz) frequency bands
- ERD patterns were most pronounced over the motor cortex during motor imagery
- Power decreases (ERD) in mu (8-12 Hz) and beta (13-30 Hz) bands are characteristic of motor imagery
- The analysis confirmed that discriminative information exists in both temporal and spectral domains

**Spatial Patterns:**
- Topographic analysis revealed clear spatial differences between left and right motor imagery
- The difference maps (Left - Right) showed distinct activation patterns that could be leveraged for classification

### Signal Quality Assessment

- **Channel Reliability**: All three EEG channels (C3, Cz, C4) showed consistent signal quality across subjects
- **Artifact Detection**: EOG channels were used to identify and exclude trials with eye movement artifacts
- **Trial Rejection**: Approximately 5-10% of trials were marked as rejected and excluded from analysis

### Key Insights for Model Development

1. **Discriminative Features**: Both temporal and spectral features contain discriminative information
2. **Optimal Time Window**: The [2.5, 7.0] second window captures the motor imagery period effectively
3. **Channel Importance**: All three motor cortex channels (C3, Cz, C4) contribute to classification
4. **Subject-Specific Patterns**: Significant inter-subject variability suggests subject-dependent training may be beneficial

---

## Results of Machine Learning

### Model Architecture: EEGNet

The project implements **EEGNet**, a compact convolutional neural network specifically designed for EEG-based brain-computer interfaces (Lawhern et al., 2018). EEGNet uses:
- **Depthwise and separable convolutions** to efficiently capture spatial and temporal features
- **Compact architecture**: Only 1,186 parameters per model
- **Small model size**: ~5 KB per trained model
- **Efficient training**: Fast convergence with minimal computational resources

### Training Strategies

Two training strategies were compared:

1. **Subject-Dependent Training**:
   - Train separate models for each subject
   - Use 3-fold cross-validation per subject
   - Models are personalized to individual neural patterns

2. **Subject-Independent Training**:
   - Leave-One-Subject-Out (LOSO) cross-validation
   - Train on data from multiple subjects, test on held-out subject
   - Tests generalization across subjects

### Model Performance Results

#### Overall Performance Comparison

| Metric | Subject-Dependent | Subject-Independent | Improvement |
|--------|------------------|---------------------|-------------|
| **Mean Accuracy** | **63.17%** | 50.42% | **+25.3%** |
| **Standard Deviation** | ± 9.59% | ± 1.30% | - |
| **Statistical Significance** | p = 0.0018 < 0.05 | - | **Significant** |

#### Per-Subject Performance (Subject-Dependent)

| Subject | Accuracy | Notes |
|---------|----------|-------|
| B01 | 63.08% | Above average performance |
| B02 | 52.02% | Lowest performance |
| B03 | 54.44% | Below average |
| **B04** | **76.25%** | **Best performing subject** |
| B05 | 54.98% | Below average |
| B06 | 63.84% | Above average |
| B07 | 69.44% | Strong performance |
| B08 | 64.14% | Above average |
| B09 | 70.31% | Strong performance |

#### Per-Subject Performance (Subject-Independent)

| Subject | Accuracy | Notes |
|---------|----------|-------|
| B01 | 51.23% | Near chance level |
| B02 | 51.06% | Near chance level |
| B03 | 49.32% | Below chance level |
| B04 | 49.62% | Near chance level |
| B05 | 49.48% | Near chance level |
| B06 | 50.17% | Near chance level |
| B07 | 50.00% | At chance level |
| B08 | 53.35% | Best generalization |
| B09 | 49.53% | Near chance level |

### Key Findings

1. **Subject-Dependent Training is Superior**:
   - 25.3% improvement over subject-independent approach
   - Statistically significant difference (p < 0.05)
   - Demonstrates the importance of personalized models in BCI applications

2. **High Inter-Subject Variability**:
   - Subject-dependent accuracy ranges from 52.02% to 76.25%
   - Standard deviation of 9.59% indicates significant individual differences
   - Subject B04 achieved exceptional performance (76.25%)

3. **Subject-Independent Training Challenges**:
   - Mean accuracy (50.42%) is near chance level (50%)
   - Low standard deviation (1.30%) suggests consistent poor performance across subjects
   - Indicates that neural patterns are highly subject-specific

4. **Model Efficiency**:
   - Compact architecture (1,186 parameters) enables real-time inference
   - Small model size (~5 KB) allows deployment on resource-constrained devices
   - Fast training and inference times support practical BCI applications

### Model Evaluation Metrics

- **Accuracy**: Primary metric for binary classification
- **Cohen's Kappa**: Measures agreement beyond chance
- **Confusion Matrix**: Detailed per-class performance analysis
- **Cross-Validation**: Robust performance estimation

### Real-World Application

The trained models have been integrated into a **web-based application** featuring:
- **React-based frontend** for user interaction
- **Django REST API** backend for model inference
- **WebSocket support** for real-time predictions
- **9 trained models** (one per subject) ready for deployment

---

## Conclusion

This capstone project successfully developed and evaluated a complete Brain-Computer Interface system for motor imagery classification. The key achievements include:

1. **Successful Classification**: Achieved 63.17% mean accuracy with subject-dependent training, significantly outperforming chance level (50%) and subject-independent approaches.

2. **Personalized Models are Essential**: The 25.3% improvement of subject-dependent over subject-independent training demonstrates that personalized models are crucial for effective BCI systems, as neural patterns are highly individual-specific.

3. **Practical Implementation**: The development of a complete pipeline from data preprocessing to web-based real-time prediction demonstrates the feasibility of deploying BCI systems in real-world applications.

4. **Comprehensive Analysis**: Through extensive EDA, we identified discriminative neural signatures (ERPs, ERD patterns) that validate the neurophysiological basis of motor imagery classification.

5. **Efficient Architecture**: The compact EEGNet architecture (1,186 parameters) enables real-time inference while maintaining competitive performance, making it suitable for practical BCI applications.

The project contributes to the growing field of BCIs by providing:
- A complete, reproducible pipeline for motor imagery classification
- Evidence-based comparison of training strategies
- An open-source implementation for the research community
- A foundation for future work in assistive technologies and human-robot interaction

---

## Limitations

Several limitations should be acknowledged:

1. **Small Sample Size**: 
   - Only 9 subjects were included in the study
   - Limited generalizability to broader populations
   - Future work should include more diverse subject populations

2. **Subject-Dependent Training Requirement**:
   - The superior performance of subject-dependent models requires individual calibration sessions
   - This may limit practical deployment in scenarios where calibration is difficult
   - Subject-independent approaches need further development

3. **Limited Channel Configuration**:
   - Only 3 EEG channels (C3, Cz, C4) were used
   - High-density EEG systems with more channels might improve performance
   - Spatial resolution is limited by channel count

4. **Binary Classification Only**:
   - The system only distinguishes left vs. right hand imagery
   - Multi-class classification (e.g., left, right, both hands, feet) was not explored
   - Limited to two-class motor imagery tasks

5. **Dataset Constraints**:
   - BCI Competition IV Dataset 2b is from 2008
   - Modern EEG systems may have different characteristics
   - Limited to laboratory-controlled conditions

6. **Performance Variability**:
   - Significant inter-subject variability (52% to 76% accuracy)
   - Some subjects (e.g., B02, B03) showed poor performance
   - Factors contributing to poor performance need investigation

7. **Real-Time Considerations**:
   - While the system supports real-time inference, full real-time pipeline validation was limited
   - Latency and computational requirements in production environments need further study

8. **Artifact Handling**:
   - Artifact rejection was based on dataset annotations
   - Real-time artifact detection and removal were not implemented
   - Eye movements and muscle artifacts could degrade performance in practice

---

## Future Research Directions

Based on the findings and limitations, several promising directions for future research are identified:

1. **Expand Subject Population**:
   - Include 20+ subjects with diverse demographics
   - Investigate factors contributing to performance variability
   - Develop strategies to improve performance for low-performing subjects

2. **Multi-Class Classification**:
   - Extend to multiple motor imagery classes (left hand, right hand, both hands, feet, tongue)
   - Explore hierarchical classification approaches
   - Investigate transfer learning between related motor imagery tasks

3. **Advanced Architectures**:
   - Experiment with attention mechanisms (Transformer-based models)
   - Investigate graph neural networks for spatial relationships
   - Explore ensemble methods combining multiple architectures

4. **Transfer Learning Approaches**:
   - Develop methods to transfer knowledge between subjects
   - Investigate few-shot learning for rapid subject adaptation
   - Explore domain adaptation techniques

5. **Real-Time Artifact Handling**:
   - Implement real-time artifact detection and removal
   - Develop adaptive filtering techniques
   - Investigate artifact-resistant feature extraction

6. **High-Density EEG Systems**:
   - Evaluate performance with 64+ channel EEG systems
   - Investigate optimal channel selection strategies
   - Explore spatial filtering techniques (CSP, xDAWN)

7. **Integration with Robotic Systems**:
   - Real-time control of robotic arms (e.g., Kinova Gen3)
   - Closed-loop feedback systems
   - Human-robot collaboration scenarios

8. **Mobile EEG Device Support**:
   - Adapt models for consumer-grade EEG devices
   - Investigate performance trade-offs with lower-quality signals
   - Develop calibration strategies for mobile devices

9. **Longitudinal Studies**:
   - Investigate performance changes over time
   - Develop adaptive models that improve with use
   - Study learning effects in BCI users

10. **Clinical Applications**:
    - Evaluate system performance with stroke patients
    - Investigate motor imagery for neurorehabilitation
    - Develop assistive technology for individuals with motor disabilities

11. **Hybrid BCI Systems**:
    - Combine motor imagery with other BCI paradigms (P300, SSVEP)
    - Integrate with other physiological signals (EMG, fNIRS)
    - Develop multi-modal BCI interfaces

12. **Explainable AI for BCIs**:
    - Develop interpretable models to understand neural patterns
    - Visualize learned features and decision boundaries
    - Provide feedback to users about their motor imagery quality

---

## Acknowledgments

- **Dataset**: BCI Competition IV Dataset 2b (Graz University of Technology, Institute for Knowledge Discovery)
- **Architecture**: EEGNet (Lawhern et al., 2018)
- **Institution**: UMBC Data Science Program
- **Advisor**: Dr. Chaojie (Jay) Wang
- **Open Source Libraries**: MNE-Python, PyTorch, Django, React

---

## References

1. Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon, S. M., Hung, C. P., & Lance, B. J. (2018). EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces. *Journal of neural engineering*, 15(5), 056013.

2. Brunner, C., Leeb, R., Müller-Putz, G., Schlögl, A., & Pfurtscheller, G. (2008). BCI Competition 2008–Graz data set A. *Institute for Knowledge Discovery (Laboratory of Brain-Computer Interfaces), Graz University of Technology*, 16, 1-6.

3. Pfurtscheller, G., & Neuper, C. (2001). Motor imagery and direct brain-computer communication. *Proceedings of the IEEE*, 89(7), 1123-1134.

4. MNE-Python Contributors. (2023). MNE-Python: A Python package for MEG and EEG data analysis. *Journal of Open Source Software*, 8(89), 5434.

---

*This report was generated as part of the UMBC DATA 606 Capstone Project, Fall 2024.*

