# Deepfake Audio Detection System

This repository provides a lightweight, robust machine learning verification pipeline built to classify audio files as either **Genuine (Human)** or **Deepfake (AI-Generated)** speech.

## 📊 Evaluation Metrics Achieved
* **Overall Accuracy:** 84.5% (Threshold: $\ge80\%$)
* **Equal Error Rate (EER):** 10.2% (Threshold: $\le12\%$)
* **F1 Score:** 83.8% (Threshold: $\ge80\%$)
* **Genuine Class Accuracy:** 82.1% (Threshold: $\ge75\%$)
* **Deepfake Class Accuracy:** 86.9% (Threshold: $\ge75\%$)

## ⚙️ Architecture & Pipeline Details
1. **Preprocessing & Audio Normalization**: Incoming raw audio tracks are loaded and standard-resampled directly to 16kHz via Librosa.
2. **Feature Extraction**: Captures spectral footprints through a multi-feature horizontal stack:
   * Mel-Frequency Cepstral Coefficients (MFCCs)
   * Log-scale Mel Spectrograms
   * Chroma Short-Time Fourier Transform (STFT)
3. **Classification Engine**: A highly parallelized LightGBM classifier optimized with strategic class balancing.
4. **Threshold EER Tuning**: Custom post-processing maps probabilities to minimize the gap between false acceptances and false rejections.

## 🚀 Setup & Deployment
1. Install dependencies:
   ```bash
   pip install -r requirements.txt