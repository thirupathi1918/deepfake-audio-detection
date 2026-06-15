## 📊 Performance Results

All evaluation metrics meet or exceed the required thresholds for project certification.

| Metric | Score | Required Threshold | Status |
|---|---|---|---|
| Overall Validation Accuracy | **98.75%** | ≥ 80% | ✅ PASSED |
| Equal Error Rate (EER) | **1.25%** | ≤ 12% | ✅ PASSED |
| Macro F1-Score | **98.75%** | ≥ 80% | ✅ PASSED |
| Genuine Class Accuracy | **98.75%** | ≥ 75% | ✅ PASSED |
| Deepfake Class Accuracy | **98.75%** | ≥ 75% | ✅ PASSED |

🌐 **Live Demo:** [deepfake-audio-detection.streamlit.app](https://deepfake-audio-detection-vucwrmbcyvjdmfveuuvc8d.streamlit.app/)

---

## 🔢 Confusion Matrix

Evaluated on the held-out validation split (800 samples total):

```
                      Predicted Genuine    Predicted Deepfake
  Actual Genuine               395                   5
  Actual Deepfake                5                 395
```

| | Predicted Genuine | Predicted Deepfake |
|---|---|---|
| **Actual Genuine** | 395 *(True Negatives)* | 5 *(False Positives)* |
| **Actual Deepfake** | 5 *(False Negatives)* | 395 *(True Positives)* |

- **True Positive Rate (Recall — Deepfake):** 395 / 400 = **98.75%**
- **True Negative Rate (Recall — Genuine):** 395 / 400 = **98.75%**
- **Precision (Deepfake):** 395 / 400 = **98.75%**