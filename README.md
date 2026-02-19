<div align="center">

# 🛡️ Online Payments Fraud Detection

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![SMOTE](https://img.shields.io/badge/SMOTE-Class%20Balancing-6DB33F?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge)

> **Detecting fraudulent online transactions using Machine Learning — with class balancing via SMOTE and ensemble classification.**

</div>

---

## 📌 Project Overview

Online payment fraud is one of the most critical challenges in digital finance. This project builds a robust machine learning pipeline to **identify fraudulent transactions** from a large, highly imbalanced dataset using supervised learning techniques.

The core challenge — extreme class imbalance — was addressed using **SMOTE (Synthetic Minority Oversampling Technique)**, applied exclusively on training data to prevent data leakage.

---

## 📊 Dataset Information

| Property | Details |
|---|---|
| **Total Samples** | 200,000 |
| **Target Column** | `isFraud` |
| **Class `0`** | ✅ Not Fraud |
| **Class `1`** | 🚨 Fraud |
| **Train Split** | 80% |
| **Test Split** | 20% |
| **Imbalance Handling** | SMOTE on training data only |

---

## ⚙️ Data Preprocessing

- ✔️ Removed duplicate records
- ✔️ Checked and handled missing values
- ✔️ Applied **SMOTE** to balance minority class in training set
- ✔️ Feature scaling applied where necessary
- ✔️ Stratified train-test split to maintain class ratios

---

## 📉 Class Balancing with SMOTE

SMOTE was applied **only to training data** to simulate real-world conditions and avoid leakage. The test set remained imbalanced.

**Before SMOTE:**

```
Class 0 (Not Fraud) : 159,786
Class 1 (Fraud)     :     214
```

**After SMOTE:**

```
Class 0 (Not Fraud) : 159,786
Class 1 (Fraud)     : 159,786  ✅ Balanced!
```

---

## 🤖 Models Trained

Three classification models were trained and compared:

| Model | Notes |
|---|---|
| 📈 Logistic Regression | Baseline linear model |
| 🌿 Decision Tree | Non-linear, interpretable |
| 🌲 Random Forest | Ensemble method — best performer |

---

## 📈 Evaluation Metrics

Each model was evaluated using a comprehensive set of metrics:

- **Accuracy** — Overall correctness
- **Confusion Matrix** — True/False Positives & Negatives
- **Precision** — Correctness of fraud predictions
- **Recall** — Coverage of actual fraud cases
- **F1-Score** — Harmonic mean of precision & recall
- **ROC Curve** — True vs. False Positive Rate
- **Precision-Recall Curve** — Performance under imbalance
- **Learning Curve** — Overfitting/underfitting diagnostics
- **Threshold Tuning** — Optimal classification threshold

---

## 🏆 Final Model — Regularized Random Forest

```
Accuracy : ~99%
Model    : RandomForestClassifier (Regularized)
```

Overfitting was reduced using the following hyperparameters:

```python
RandomForestClassifier(
    max_depth=...,
    min_samples_split=...,
    min_samples_leaf=...
)
```

> ✅ High accuracy with a strong balance between precision and recall — no major overfitting observed after regularization.

---

## 📊 Visualizations Included

| Visualization | Purpose |
|---|---|
| Class Distribution Graph | Visualize imbalance before/after SMOTE |
| Correlation Heatmap | Feature relationships |
| Boxplots | Outlier detection |
| ROC Curve | Model discrimination ability |
| Precision-Recall Curve | Performance under imbalance |
| Model Comparison Chart | Side-by-side evaluation |
| Learning Curve | Bias-variance diagnostics |

---

## 🧠 Key Findings

- 🌲 **Random Forest** outperformed all other models
- 📊 **SMOTE** significantly improved minority class (fraud) recall
- ⚖️ Class 1 (Fraud) recall improved substantially after balancing
- 🔧 Regularization effectively controlled overfitting
- 🚫 No major overfitting observed in the final model

---

## 💾 Model Saving

The final model is saved using `joblib` for easy deployment:

```python
import joblib
joblib.dump(model, "fraud_detection_model.pkl")

# Load later
model = joblib.load("fraud_detection_model.pkl")
```

---

## 🚀 How to Run

**1. Open the notebook in Google Colab**

**2. Install required libraries:**

```bash
pip install imbalanced-learn
```

**3. Run all cells in order** — preprocessing → SMOTE → training → evaluation

---

## 📂 Project Structure

```
Online_Payments_Fraud_Detection/
│
├── 📓 Online_Payments_Fraud_Detection.ipynb   # Main notebook
├── 🤖 fraud_detection_model.pkl               # Saved model
└── 📄 README.md                               # Project documentation
```

---

## 🔮 Future Improvements

- [ ] Hyperparameter tuning with **GridSearchCV** / **RandomizedSearchCV**
- [ ] Explore **XGBoost** and **LightGBM** for better performance
- [ ] Deploy model as an interactive app using **Streamlit**
- [ ] Build a **real-time fraud detection REST API**
- [ ] Experiment with **deep learning** approaches (e.g., autoencoders)

---

## 👨‍💻 Author

<div align="center">

**Your Name**
*Machine Learning Project — 2026*

</div>

---

<div align="center">

⭐ **If you found this project helpful, please consider giving it a star!** ⭐

</div>
