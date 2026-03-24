# 🤚 Hand Gesture Classification Using MediaPipe Landmarks

A machine learning project for classifying hand gestures using 21-landmark data extracted from the [HaGRID dataset](https://github.com/hukenovs/hagrid) via MediaPipe.

---

## 📋 Project Overview

This project takes a CSV file of hand landmark coordinates (x, y, z) as input and trains multiple machine learning classifiers to recognize hand gestures. The best-performing model is then used to classify gestures in real-time video.

**Key steps covered:**
- Data loading & exploration
- Landmark normalization (re-centering & scaling)
- Stratified train/test splitting
- Training and comparing multiple ML models with Grid Search
- Evaluation using accuracy, precision, recall, and F1-score
- Experiment tracking via MLflow
- Real-time gesture classification using MediaPipe

---

## 📁 Project Structure

```
├── Dataset/
│   └── hand_landmarks_data.csv   # Extracted MediaPipe landmarks + labels
├── mlruns/                        # MLflow experiment tracking logs
├── evaluation.py                  # Model evaluation utilities
├── mlflow_logging.py              # MLflow setup & logging helpers
├── notebook.ipynb                 # Main Colab notebook
└── README.md
```

---

## 🗂️ Dataset

- **Source:** [HaGRID](https://github.com/hukenovs/hagrid) — Hand Gesture Recognition Image Dataset
- **Features:** 21 hand landmarks × 3 coordinates (x, y, z) = **63 features** per sample
- **Classes:** 18 hand gesture classes + `no_gesture` (added manually)
- **Class imbalance:** Ranges from ~945 samples (`fist`) to ~1653 samples (`three2`)
- A `no_gesture` class with 1000 zero-filled samples was added to handle absent hands

---

## ⚙️ Setup & Installation

```bash
pip install pandas numpy matplotlib seaborn scikit-learn mlflow mediapipe
```

---

## 🔄 Preprocessing Pipeline

1. **No missing values** — confirmed via `data.info()`
2. **Added `no_gesture` class** — 1000 rows of zeros appended
3. **Stratified splitting** — `StratifiedShuffleSplit` (80/20) to preserve class ratios
4. **Landmark normalization:**
   - Re-center all (x, y) coordinates using the **wrist point** (landmark 1) as the origin
   - Scale by the **distance to the middle finger base** (landmark 13) to make all hands comparable in size
   - Z coordinates are left as-is (already normalized by MediaPipe)
5. **Label encoding** — `OrdinalEncoder` applied to gesture class labels

---

## 🤖 Models Trained

| Model | Accuracy | Precision | Recall | F1-Score | Tuning |
|---|---|---|---|---|---|
| KNN (K=5) | 97.75% | 97.79% | 97.77% | 97.77% | Manual |
| KNN (K=3) | 97.98% | 98.02% | 98.00% | 98.00% | Manual |
| Decision Tree | 95.76% | 95.84% | 95.82% | 95.82% | GridSearchCV |
| Random Forest | 97.60% | 97.66% | 97.62% | 97.63% | GridSearchCV |
| **SVC (RBF kernel)** ⭐ | **98.59%** | **98.62%** | **98.60%** | **98.60%** | GridSearchCV |

All experiments were tracked using **MLflow**.

**Best model: SVC** — achieved the highest scores across all metrics and was selected for deployment.

---

## 📊 Evaluation

Each model is assessed using:
- **Accuracy**
- **Precision** (macro)
- **Recall** (macro)
- **F1-Score** (macro)
- **Confusion Matrix** (row-normalized %)

---

## 🎥 Output Video

A demo video showing real-time gesture classification using the trained SVC model and MediaPipe hand landmark extraction is available here:

👉 **[Video Demo Link](https://drive.google.com/drive/u/2/folders/1VeAmcvCfZLaxk1P2Ac0qH2m_j5JYeHZV)** 

> Tip: Predictions are stabilized using a sliding window mode over recent frames to reduce flickering.

---

## 🔗 Resources

- 📓 **Colab Notebook:** *(replace with your GitHub link)*
- 📦 **HaGRID Dataset:** https://github.com/hukenovs/hagrid
- 🖐️ **MediaPipe Hands:** https://developers.google.com/mediapipe/solutions/vision/hand_landmarker

---

## 👥 Authors

*(Add your names here)*
