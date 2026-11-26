# PTB-XL ECG Delineation Annotations 

A manually annotated dataset of P/QRS/T wave onset and offset points for 12-lead ECG, comprising 1100 samples. It is constructed based on the PTB-XL dataset and is suitable for 12-lead ECG synchronous delineation tasks. The annotations were performed by senior cardiologists and physiologists and have undergone multiple rounds of review and verification.

## 🔍 Dataset Overview
### 1.1 Data Source
This annotation set is constructed based on the PTB-XL dataset​ (https://physionet.org/content/ptb-xl/1.0.3/). It contains only annotation information, not the original ECG signal data. It must be used by linking with the original dataset via the record_id.

### 1.2 Sample Size
Total number of samples: 1000+
Base samples: 800+, proportionally and randomly selected​ from the 5 arrhythmia classes (MI/AF/STTC/CD/HYP) in PTB-XL.
Supplementary samples: 200+ AF (Atrial Fibrillation) samples extracted from PTB-XL to enhance annotation coverage for low-amplitude P-wave detection tasks.
Annotation coverage: Full 12-lead annotation (I, II, III, aVR, aVL, aVF, V1-V6), synchronously annotating the onset​ and offset​ of P-waves, QRS complexes, and T-waves.

