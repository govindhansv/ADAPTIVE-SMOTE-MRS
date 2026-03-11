# 🔬 Adaptive SMOTE-MRS — Complete Project Status

> **Project Title:** Adaptive SMOTE-MRS: An Enhanced Multi-Resolution Sampling Technique for Imbalanced Medical Data Classification  
> **Base Paper:** SMOTE-MRS: A Novel SMOTE-Multiresolution Sampling Technique for Imbalanced Distribution to Improve Prediction of Anemia  
> **Domain:** Machine Learning / Medical Data Analytics  
> **Execution Environment:** Kaggle Notebooks ([notebook link](https://www.kaggle.com/code/jimhopper333/notebookcc21ac3543))  
> **Last Updated:** February 12, 2026

---

## 📌 Project Summary

This MSc research project improves upon the SMOTE-MRS algorithm from the base paper by introducing **Adaptive cluster-wise oversampling** based on the Imbalance Ratio (IR) of each cluster. Instead of blindly applying SMOTE to all clusters (as the base paper does), our method checks whether each cluster actually *needs* oversampling — skipping balanced clusters and focusing resources on truly imbalanced regions.

---

## 🧠 Core Concepts

### The Problem — Class Imbalance
Medical datasets have far more healthy patients than sick ones (e.g., 19:1 in stroke data). Standard ML models trained on such data learn to always predict "Healthy" — achieving high accuracy but missing most sick patients.

### Base Paper's Approach — SMOTE-MRS (3 Steps)

```
Step 1: K-Means Clustering → Group minority samples into R clusters
Step 2: SMOTE              → Generate synthetic samples within each cluster
Step 3: ROS                → Random Over-Sampling for final balance
```

### Our Contribution — Adaptive SMOTE-MRS

```
For each cluster:
  → Calculate IR = Majority / Minority
  → If IR ≥ 1.5 (threshold): Apply SMOTE ✅
  → If IR < 1.5 (balanced):  Skip SMOTE ⏭️
```

**Why it's better:** Avoids noise amplification in already-balanced clusters, reduces unnecessary synthetic samples, and focuses on truly imbalanced regions.

---

## ✅ What Has Been Completed

### 1. Base Paper Replication
| Task | Status | File |
|------|--------|------|
| SMOTE-MRS algorithm implementation | ✅ Done | [smote_mrs_base.py](file:///c:/Users/gokul/Desktop/Research/smote_mrs_base.py) |
| Pima Diabetes evaluation (10-Fold CV) | ✅ Done | [base.py](file:///c:/Users/gokul/Desktop/Research/new/base.py) |
| Stroke dataset evaluation (10-Fold CV) | ✅ Done | [stroke.py](file:///c:/Users/gokul/Desktop/Research/new/stroke.py) |
| UCI CKD + Pima dual evaluation | ✅ Done | [uclandpima.py](file:///c:/Users/gokul/Desktop/Research/new/uclandpima.py) |
| Standalone evaluation pipeline | ✅ Done | [evaluate_base.py](file:///c:/Users/gokul/Desktop/Research/evaluate_base.py) |

### 2. Adaptive SMOTE-MRS (Our Innovation)
| Task | Status | Details |
|------|--------|---------|
| Adaptive algorithm designed | ✅ Done | IR-based cluster-wise decision |
| Tested on Pima Diabetes | ✅ Done | See results below |
| Tested on Stroke Prediction | ✅ Done | See results below |
| Comparison with base method | ✅ Done | Improvement shown in all metrics |

### 3. Documentation & Presentation
| Task | Status | File |
|------|--------|------|
| Project documentation | ✅ Done | [PROJECT_DOCUMENTATION.md](file:///c:/Users/gokul/Desktop/Research/mdfiles/PROJECT_DOCUMENTATION.md) |
| Q&A for guide review | ✅ Done | [QUESTIONS_AND_ANSWERS.md](file:///c:/Users/gokul/Desktop/Research/mdfiles/QUESTIONS_AND_ANSWERS.md) |
| PPT slide content | ✅ Done | [PPT_SLIDES_CONTENT.md](file:///c:/Users/gokul/Desktop/Research/mdfiles/PPT_SLIDES_CONTENT.md) |
| PowerPoint files | ✅ Done | [others/](file:///c:/Users/gokul/Desktop/Research/others) folder |
| Flowchart generator | ✅ Done | [main.py](file:///c:/Users/gokul/Desktop/Research/new/main.py) |
| Reference papers collected | ✅ Done | 10 papers in [refpapers/](file:///c:/Users/gokul/Desktop/Research/refpapers) |

---

## 📊 Results Achieved So Far

### Base SMOTE-MRS (Replication)

| Dataset | Model | Accuracy | Recall | F1 | AUC |
|---------|-------|----------|--------|-----|-----|
| Pima Diabetes | Random Forest | 74.73% | 63.39% | 63.43% | 0.808 |
| Pima Diabetes | Naïve Bayes | — | — | — | — |
| Pima Diabetes | SVM | — | — | — | — |
| Stroke | Random Forest | 92.27% | 6.82% | 7.87% | 0.798 |
| Stroke | Naïve Bayes | 70.82% | 70.28% | 19.31% | 0.766 |
| Stroke | Logistic Reg. | 71.51% | 61.87% | 17.49% | 0.744 |

### Adaptive SMOTE-MRS (Our Contribution)

| Dataset | Model | Accuracy | Recall | F1 | AUC |
|---------|-------|----------|--------|-----|-----|
| Pima Diabetes | Random Forest | **76.29%** | **66.35%** | **65.99%** | **0.810** |
| Stroke | Random Forest | **92.43%** | **8.02%** | **9.25%** | **0.801** |

### Improvement Summary

| Dataset | Accuracy ↑ | Recall ↑ | AUC ↑ |
|---------|-----------|----------|-------|
| Pima Diabetes | **+1.56%** | **+2.96%** | **+0.16%** |
| Stroke | **+0.16%** | **+1.20%** | **+0.25%** |

> [!IMPORTANT]
> The **Recall improvement** is the most critical metric — it means we're catching more sick patients. This is the primary goal in medical data classification.

---

## 📂 Complete File Structure

```
Research/
├── PROJECT_STATUS.md              ← This file (complete project overview)
├── PROJECT_DOCUMENTATION.md       ← Detailed project docs (copy of mdfiles version)
├── smote_mrs_base.py              ← Core SMOTE-MRS class (uses imblearn)
├── evaluate_base.py               ← Evaluation pipeline with 10-fold CV
├── SMOTE-MRS_A_Novel_...pdf       ← Base paper PDF
│
├── new/                           ← Kaggle notebook scripts
│   ├── base.py                    ← Base paper replication on Pima (uses imblearn SMOTE)
│   ├── stroke.py                  ← Stroke dataset evaluation (custom SMOTE)
│   ├── uclandpima.py              ← Pima + UCI CKD dual evaluation (custom SMOTE)
│   └── main.py                    ← PowerPoint + flowchart generator (Graphviz)
│
├── mdfiles/                       ← Documentation
│   ├── PROJECT_DOCUMENTATION.md   ← Full project explanation
│   ├── QUESTIONS_AND_ANSWERS.md   ← Viva preparation Q&A
│   ├── PPT_SLIDES_CONTENT.md      ← 20-slide presentation content
│   └── aravind_project.md         ← Friend's project reference
│
├── others/                        ← Additional resources
│   ├── Adaptive_SMOTE_MRS_Presentation.pptx
│   ├── Adaptive_SMOTE_MRS_With_Diagram.pptx
│   ├── Essential Research Metrics and Terms.txt
│   ├── Imbalanced Medical Datasets.pdf
│   ├── Presentation Audio.mpeg
│   └── Project.pdf
│
└── refpapers/                     ← 10 reference papers
    ├── basepaper.pdf
    ├── Emerging_SMOTE_and_GAN_Variants_...pdf
    ├── IJST-2024-3904.pdf
    └── ... (7 more)
```

---

## 🗺️ Datasets Used

| # | Dataset | Samples | Features | Imbalance Ratio | Source |
|---|---------|---------|----------|-----------------|--------|
| 1 | Pima Indians Diabetes | 768 | 8 (Glucose, BMI, Age, etc.) | 1.87:1 | UCI Repository |
| 2 | Stroke Prediction | 5,110 | 10 (Age, Hypertension, etc.) | 19.52:1 | Kaggle |
| 3 | UCI CKD (Kidney Disease) | 400 | 24 (various) | ~2.5:1 | Kaggle/UCI |

---

## 🔧 Technical Stack

- **Language:** Python 3.x
- **ML Libraries:** scikit-learn, imbalanced-learn (imblearn)
- **Data:** NumPy, Pandas
- **Visualization:** Matplotlib, Seaborn
- **Presentation:** python-pptx, Graphviz
- **Environment:** Kaggle Notebooks (GPU-enabled)
- **Validation:** 10-Fold Stratified Cross-Validation

---

## ⚠️ Key Points About Results

> [!NOTE]
> **Why our accuracy (74-76%) is lower than the base paper's claimed 97-99%:**
>
> 1. **Data Leakage in Base Paper** — They likely applied SMOTE *before* train/test split, causing synthetic test samples to be very similar to training samples
> 2. **Different Datasets** — Base paper used a private 128-sample Anemia dataset; we use larger, public datasets (768 and 5,110 samples)
> 3. **Correct Methodology** — We apply SMOTE *only on training data* after the split (no leakage)
> 4. **Proper CV** — 10-Fold Stratified Cross-Validation gives realistic, defensible results

---

## 🔮 What Needs to Be Done Next

### Priority: HIGH 🔴

| # | Task | Description | Estimated Effort |
|---|------|-------------|-----------------|
| 1 | **Run Adaptive SMOTE-MRS on UCI CKD dataset** | Third dataset to prove generalizability | 1-2 hours |
| 2 | **Hyperparameter tuning** | Test different IR thresholds (2.0, 3.0) and cluster counts (R=7, 10) | 2-3 hours |
| 3 | **Add XGBoost/LightGBM classifiers** | Modern models that may boost performance significantly | 2-3 hours |
| 4 | **Generate comparison visualizations** | ROC curves, confusion matrices, bar charts comparing Base vs Adaptive | 2-3 hours |

### Priority: MEDIUM 🟡

| # | Task | Description | Estimated Effort |
|---|------|-------------|-----------------|
| 5 | **Statistical significance testing** | Wilcoxon signed-rank test to prove improvement is statistically significant | 1-2 hours |
| 6 | **Compare with other methods** | Benchmark against plain SMOTE, ADASYN, Borderline-SMOTE | 3-4 hours |
| 7 | **Feature importance analysis** | Show which features matter most after adaptive oversampling | 1-2 hours |
| 8 | **Clean and consolidate codebase** | Merge duplicate implementations into one clean module | 2-3 hours |

### Priority: LOW 🟢

| # | Task | Description | Estimated Effort |
|---|------|-------------|-----------------|
| 9 | **Build Streamlit web demo** | Upload CSV → Apply Adaptive SMOTE-MRS → Download balanced data | 3-4 hours |
| 10 | **Write IEEE-format project report** | Final submission document | 1-2 days |
| 11 | **Finalize presentation slides** | Polish PPT with actual visualizations and results | 3-4 hours |
| 12 | **Prepare for viva** | Practice answers from the Q&A document | Ongoing |

---

## 🚀 Recommended Next Steps (In Order)

1. **Open Kaggle notebook** and run the Adaptive SMOTE-MRS on UCI CKD dataset
2. **Try different hyperparameters** — IR thresholds and cluster counts in a grid:
   ```python
   for R in [3, 5, 7, 10]:
       for threshold in [1.5, 2.0, 3.0]:
           # Run Adaptive SMOTE-MRS and record results
   ```
3. **Add XGBoost** (single most impactful improvement for results):
   ```python
   from xgboost import XGBClassifier
   models['XGBoost'] = XGBClassifier(n_estimators=200, max_depth=6, random_state=42)
   ```
4. **Generate all visualizations** for the final report and presentation
5. **Run statistical tests** and consolidate everything into the final report

---

## 📋 Overall Progress

```
██████████████████░░░░░░░░░░░░░░ 55% Complete
```

| Phase | Status |
|-------|--------|
| Literature Review | ✅ Complete |
| Base Paper Replication | ✅ Complete |
| Adaptive Algorithm Design | ✅ Complete |
| Initial Testing (2 datasets) | ✅ Complete |
| Documentation & PPT Draft | ✅ Complete |
| Extended Testing (3rd dataset) | ⬜ Not Started |
| Hyperparameter Tuning | ⬜ Not Started |
| Advanced Models (XGBoost) | ⬜ Not Started |
| Visualization & Charts | ⬜ Not Started |
| Statistical Testing | ⬜ Not Started |
| Benchmarking Other Methods | ⬜ Not Started |
| Web Demo (Optional) | ⬜ Not Started |
| Final Report (IEEE) | ⬜ Not Started |
| Final PPT | ⬜ Not Started |
| Viva Preparation | 🔄 In Progress |

---

*Generated: February 12, 2026*
