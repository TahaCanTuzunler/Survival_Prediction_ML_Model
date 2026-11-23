# 🩺 Survival Prediction in Acute Pancreatitis — ML Model

A machine learning project that predicts **one-month survival** of acute pancreatitis patients from admission-time clinical and laboratory data. Built for my *Data Science & Business Intelligence* course.

The project covers the full workflow: exploratory analysis, **leakage-free imputation** of missing values, categorical encoding, training two classifiers, and evaluating them with a focus on the clinically critical class (patients who die).

> ⚠️ **Disclaimer:** This is an educational exercise. It is **not** a clinical diagnostic tool and its predictions are not validated for medical decisions.

---

## 📂 Repository Contents

| File | Description |
|------|-------------|
| `model.py` | Full Python pipeline (data loading → imputation → encoding → modeling → evaluation). Cells are separated with `# %%` so it runs in VS Code / Jupyter interactive mode. |
| `Acute_Pancreatitis_Survival_Assignment_Student.xlsx` | Dataset (patient data, assignment notes and data dictionary sheets). |
| `REPORT.docx` | Written project report explaining decisions, results and conclusions. |
| `requirements.txt` | Python dependencies. |

---

## 📊 Dataset

- **Source:** Ince et al., *Early prediction of the severe course, survival, and ICU requirements in acute pancreatitis by artificial intelligence*, Pancreatology (2023). [DOI: 10.1016/j.pan.2022.12.005](https://doi.org/10.1016/j.pan.2022.12.005)
- **Size:** 1,334 patients × 16 features + 1 target
- **Target:** `SURVIVE` → `0 = Alive`, `1 = Died` (within one month)
- **Class imbalance:** 1,202 survived vs. 132 died (~10% positive class)
- **Missing values:** 160 values (12%) deliberately masked in each of `BUN_MG_DL`, `CRP_MG_DL` and `DIABETES`

| Group | Features |
|-------|----------|
| Demographics | `SEX`, `AGE_YEARS`, `HEIGHT_CM`, `WEIGHT_KG`, `BMI_RECORDED`, `BMI_GROUP` |
| History | `SMOKING`, `ALCOHOL_USE`, `DIABETES`, `HYPERTENSION` |
| Cause | `ETIOLOGY` (15 categories) |
| Lab results | `WBC`, `HEMATOCRIT`, `CRP_MG_DL`, `BUN_MG_DL`, `CREATININE_MG_DL` |

---

## ⚙️ Pipeline

```
Raw data ──► Train/Test split (70/30, stratified) ──► Imputation ──► Encoding ──► Model training ──► Evaluation
               ▲
               └── done FIRST, so every imputer/encoder is fit on training data only (no data leakage)
```

### 1. Train/Test Split
The data is split **before** any preprocessing (`test_size=0.30`, `stratify=y`, `random_state=42`). Every imputer and encoder learns only from the training set and is then applied to the test set, which prevents information from the test set leaking into the model.

### 2. Missing Value Imputation
Each imputation method was chosen after a correlation analysis of the variable:

| Variable | Method | Why |
|----------|--------|-----|
| `BUN_MG_DL` | **KNN Imputer** (k=5, on MinMax-scaled data) | BUN is strongly correlated with Creatinine and Age, so values from similar patients give a better estimate than a single median. |
| `CRP_MG_DL` | **Median Imputer** | CRP showed no strong correlation with other features, and as an inflammation marker it has extreme outliers, so the median is a safe, robust fill. |
| `DIABETES` | **Logistic Regression** | It is binary, so it's treated as a classification problem. Filling with the most frequent value would reduce variance; Logistic Regression was chosen over Naive Bayes because the predictor columns are not independent. |

### 3. Categorical Encoding
- **`SEX`** → manual mapping (`Female: 1`, `Male: 0`)
- **`ETIOLOGY`** → the 5 most frequent causes in the training set (Biliary, Idiopathic, Hypertriglyceridemia, Multiple etiologies, Alcohol-related) are kept and the rest grouped as `Other`, then **one-hot encoded** (`drop_first=True`). Train/test columns are aligned to guard against unseen categories.

### 4. Models
| Model | Reason for choice |
|-------|-------------------|
| **Logistic Regression** | Handles correlated features by assigning weights to each. |
| **Gaussian Naive Bayes** | Works well with continuous numerical data (despite assuming feature independence). |

---

## 📈 Results (test set: 401 patients, 40 deaths)

| Metric | Logistic Regression | Gaussian Naive Bayes |
|--------|:------------------:|:-------------------:|
| Accuracy | **0.90** | 0.88 |
| ROC-AUC | **0.787** | 0.783 |
| Precision (Died) | 0.25 | **0.37** |
| **Recall (Died)** | 0.03 | **0.25** |
| F1-score (Died) | 0.05 | **0.30** |
| Deaths caught (TP) | 1 / 40 | **10 / 40** |
| Missed deaths (FN) | 39 | **30** |
| False alarms (FP) | **3** | 17 |

### Confusion Matrices

**Logistic Regression**
| | Predicted Alive | Predicted Died |
|---|:---:|:---:|
| **Actual Alive** | 358 | 3 |
| **Actual Died** | 39 | 1 |

**Gaussian Naive Bayes**
| | Predicted Alive | Predicted Died |
|---|:---:|:---:|
| **Actual Alive** | 344 | 17 |
| **Actual Died** | 30 | 10 |

---

## 🧠 Conclusion

The results show a clear **accuracy paradox** caused by class imbalance:

- **Logistic Regression** reached the highest accuracy (90%) mostly by predicting "Alive" for nearly everyone. It caught only **1 of 40** patients who died.
- **Gaussian Naive Bayes** had slightly lower accuracy (88%) but caught **10 of 40** deaths, with a much better recall and F1-score on the critical class.
- Both models have a similar ROC-AUC (~0.78), so they rank patients similarly, but GNB's default decision threshold suits this problem better.

In medical analytics, a **false negative** (missing a high-risk patient) is far more dangerous than a false alarm. For that reason, **Gaussian Naive Bayes is the preferred model** for this dataset.

### Lessons Learned & Possible Improvements
- Grouping `ETIOLOGY` into the top 5 + "Other" put many potentially meaningful causes into one bucket. Target encoding or a clinically informed grouping could keep more information.
- The class imbalance could be addressed with `class_weight='balanced'`, decision-threshold tuning, or resampling methods such as SMOTE (applied to training data only).
- Feature scaling before Logistic Regression and cross-validation instead of a single split would make the comparison more robust.

---

## 🚀 How to Run

```bash
git clone https://github.com/TahaCanTuzunler/Survival_Prediction_ML_Model.git
cd Survival_Prediction_ML_Model
pip install -r requirements.txt
```

Then open `model.py` in **VS Code** (with the Jupyter extension) or any editor that supports `# %%` cells, and run the cells in order. The dataset file must be in the same folder as the script.

> Code comments and console output are in Turkish.

---

## 🛠️ Tech Stack
Python · pandas · NumPy · scikit-learn · Jupyter / IPython

## 👤 Author
**Taha Can Tüzünler** — Management Information Systems
[GitHub](https://github.com/TahaCanTuzunler)
