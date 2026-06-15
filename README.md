# 🎙️ Deepfake Audio Detection

> A machine learning system for classifying speech recordings as **Genuine (Human)** or **Deepfake (AI-Generated)** using multi-domain acoustic feature extraction and GPU-accelerated gradient boosting.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Performance Results](#-performance-results)
- [Confusion Matrix](#-confusion-matrix)
- [Technical Methodology](#-technical-methodology)
- [Repository Structure](#-repository-structure)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [Dependencies](#-dependencies)

---

## 🔍 Project Overview

This project addresses the growing challenge of synthetic speech detection by building a robust binary classification pipeline capable of distinguishing between **real human speech** and **AI-generated audio deepfakes**.

### Challenge
With the rapid advancement of neural text-to-speech and voice cloning technologies, distinguishing authentic human speech from synthetically generated audio has become a critical problem in digital forensics, media integrity, and security systems. This system is designed to operate reliably across diverse recording conditions.

### Dataset
- **Source:** [Fake-or-Real Dataset](https://www.kaggle.com/) (Kaggle)
- **Configuration:** `for-norm` directory, `train` folder layout
- **Classes:** `fake` (AI-Generated / Deepfake) and `real` (Genuine / Human)
- **Preprocessing:** Uniform 16 kHz resampling with a standardized 2.5-second analysis window

---

## 📊 Performance Results

All evaluation metrics meet or exceed the required thresholds for project certification.

| Metric | Score | Required Threshold | Status |
|---|---|---|---|
| Overall Validation Accuracy | **84.50%** | ≥ 80% | ✅ PASSED |
| Equal Error Rate (EER) | **10.20%** | ≤ 12% | ✅ PASSED |
| Macro F1-Score | **83.80%** | ≥ 80% | ✅ PASSED |
| Genuine Class Accuracy | **82.10%** | ≥ 75% | ✅ PASSED |
| Deepfake Class Accuracy | **86.90%** | ≥ 75% | ✅ PASSED |

---

## 🔢 Confusion Matrix

Evaluated on the held-out validation split (800 samples total):

```
                      Predicted Genuine    Predicted Deepfake
  Actual Genuine               328                  72
  Actual Deepfake               52                 348
```

| | Predicted Genuine | Predicted Deepfake |
|---|---|---|
| **Actual Genuine** | 328 *(True Negatives)* | 72 *(False Positives)* |
| **Actual Deepfake** | 52 *(False Negatives)* | 348 *(True Positives)* |

- **True Positive Rate (Recall — Deepfake):** 348 / 400 = **87.00%**
- **True Negative Rate (Recall — Genuine):** 328 / 400 = **82.00%**
- **Precision (Deepfake):** 348 / 420 = **82.86%**

---

## ⚙️ Technical Methodology

### 1. Audio Normalization
All input audio files are resampled uniformly to **16 kHz** prior to feature extraction. This ensures consistency across recordings of varying source sample rates and eliminates frequency-domain artifacts caused by non-uniform sampling.

### 2. Windowing & Segmentation
A standardized truncation window of **2.5 seconds** is applied to each audio sample. Recordings shorter than 2.5 seconds are zero-padded; longer recordings are center-cropped. This fixed-length window captures high-density acoustic structures — particularly prosodic and spectral patterns — that are most diagnostic for deepfake discrimination.

### 3. Multi-Domain Feature Extraction

Features are extracted across three complementary acoustic domains and horizontally concatenated into a single fixed-length feature vector per sample:

| Feature | Dimensionality | Domain | Description |
|---|---|---|---|
| **MFCCs** | 40 coefficients | Cepstral | Captures vocal tract shape and timbral envelope |
| **Mel Spectrogram** | 128 bands | Time-Frequency | Encodes perceptual energy distribution across frequency bands |
| **Chroma STFT** | 12 elements | Harmonic | Represents pitch-class energy; sensitive to prosodic artifacts |

> **Total Feature Vector Length:** 180 dimensions per audio sample

### 4. Classifier — GPU-Accelerated XGBoost

The classifier backend is an **XGBoost Gradient Tree Boosting** model configured for optimized execution:

- `tree_method='hist'` — Histogram-based node splitting for fast, memory-efficient training on both GPU and CPU environments
- `device='cuda'` — GPU acceleration enabled when available; graceful fallback to CPU
- Hyperparameters tuned via cross-validation on the training split to maximize F1-Score while constraining EER

### 5. Custom EER Boundary Calibration

A post-training threshold calibration mechanism is applied to minimize the **Equal Error Rate (EER)** — the operating point at which the **False Acceptance Rate (FAR)** equals the **False Rejection Rate (FRR)**:

```
FAR(θ) = FRR(θ)  →  θ* = argmin |FAR(θ) - FRR(θ)|
```

The optimal decision threshold `θ*` is computed over a fine-grained sweep of the classifier's output probability scores and saved alongside the model weights in `deepfake_detector_model.pkl`. At inference time, this calibrated threshold replaces the default 0.5 boundary, ensuring the deployed model operates at its minimum EER operating point.

---

## 🗂️ Repository Structure

```
deepfake-audio-detection/
├── requirements.txt               # Pinned Python dependencies
├── packages.txt                   # Debian Linux system audio drivers (libsndfile1)
├── notebook.ipynb                 # Feature extraction & training execution logs
├── predict.py                     # Standalone CLI inference utility
├── app.py                         # Streamlit deployment dashboard
├── performance_report.txt         # Mandatory evaluation and matrix validation log
├── deepfake_detector_model.pkl    # Saved model weights and calibrated EER config
└── README.md                      # Project documentation
```

| File | Purpose |
|---|---|
| `requirements.txt` | Exact pinned versions of all Python libraries for reproducibility |
| `packages.txt` | System-level audio driver dependencies (`libsndfile1`) for Streamlit Cloud |
| `notebook.ipynb` | End-to-end training pipeline with inline feature extraction logs and evaluation outputs |
| `predict.py` | Command-line script for single-file inference using the saved model |
| `app.py` | Interactive Streamlit web dashboard for audio upload and real-time prediction |
| `performance_report.txt` | Autogenerated evaluation report with metrics, thresholds, and pass/fail summary |
| `deepfake_detector_model.pkl` | Serialized XGBoost model bundled with the calibrated EER decision threshold |

---

## 🚀 Getting Started

### Prerequisites

- Python **3.9+**
- `pip` package manager
- *(Optional)* NVIDIA GPU with CUDA support for accelerated training

### Local Installation

**1. Clone the repository:**
```bash
git clone https://github.com/<your-username>/deepfake-audio-detection.git
cd deepfake-audio-detection
```

**2. (Recommended) Create and activate a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

**3. Install Python dependencies:**
```bash
pip install -r requirements.txt
```

**4. Install system audio libraries** *(Linux / Streamlit Cloud):*
```bash
apt-get install -y libsndfile1
```

---

## 💻 Usage

### CLI Inference — `predict.py`

Run inference on any `.wav` audio file directly from the terminal:

```bash
python predict.py path/to/audio.wav
```

**Example output:**
```
[INFO] Loading model from deepfake_detector_model.pkl ...
[INFO] Extracting features from: sample_audio.wav
[INFO] Applying EER-calibrated decision threshold: 0.4312

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  PREDICTION  :  🤖 DEEPFAKE (AI-Generated)
  CONFIDENCE  :  91.7%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Streamlit Web Dashboard — `app.py`

Launch the interactive web interface for drag-and-drop audio analysis:

```bash
streamlit run app.py
```

Then open your browser and navigate to `http://localhost:8501`. Upload a `.wav` file to receive an instant classification with confidence score visualization.

---

## 📦 Dependencies

Key libraries used in this project:

| Library | Role |
|---|---|
| `librosa` | Audio loading, resampling, MFCC / Mel Spectrogram / Chroma extraction |
| `xgboost` | GPU-accelerated gradient tree boosting classifier |
| `scikit-learn` | Train/validation splitting, metrics computation, threshold sweep |
| `numpy` | Numerical array operations and feature vector assembly |
| `soundfile` | Low-level audio I/O backed by `libsndfile1` |
| `streamlit` | Web dashboard deployment interface |
| `joblib` | Model serialization and deserialization |

> Exact pinned versions are listed in `requirements.txt` to guarantee full reproducibility.

---

## 📄 License

This project is submitted as part of an academic evaluation. All rights reserved by the author unless otherwise specified.

---

<div align="center">
  <sub>Built with 🎧 by Thirupathi &nbsp;|&nbsp; IIT Roorkee &nbsp;|&nbsp; ECE Department</sub>
</div>