# Adaptive SMOTE-MRS for Imbalanced Medical Data Classification

## 📋 Project Overview

| Item | Details |
|------|---------|
| **Project Title** | Adaptive SMOTE-MRS: An Enhanced Multi-Resolution Sampling Technique for Imbalanced Medical Data |
| **Base Paper** | SMOTE-MRS: A Novel SMOTE-Multiresolution Sampling Technique for Imbalanced Distribution to Improve Prediction of Anemia |
| **Domain** | Machine Learning / Medical Data Analysis |
| **Contribution** | Adaptive cluster-wise oversampling based on Imbalance Ratio |

---

## 📖 Understanding the Problem

### What is Class Imbalance?
In medical datasets, we often have **many more healthy patients than sick ones**. For example:
- 1000 healthy patients vs 50 diabetic patients
- This is called **class imbalance** (ratio 20:1)

### Why is it a Problem?
When you train a machine learning model on imbalanced data:
- The model learns to predict "Healthy" most of the time
- It achieves **high accuracy** (95%) but **misses most sick patients**
- Example: A model might correctly identify 950/1000 healthy patients but miss 45/50 sick patients

### The Goal
We want models that can **detect sick patients (minority class)** even when they are rare. The key metric is **Recall** (sensitivity) — the percentage of sick patients correctly identified.

---

## 📚 Base Paper Explanation

### Paper Title
**"SMOTE-MRS: A Novel SMOTE-Multiresolution Sampling Technique for Imbalanced Distribution to Improve Prediction of Anemia"**

### What the Paper Proposes
The paper introduces a **3-step hybrid data balancing technique**:

```
Step 1: K-Means Clustering (Multi-Resolution)
        ↓
Step 2: SMOTE (Synthetic Minority Oversampling)
        ↓
Step 3: Random Over Sampling (ROS)
```

### Step-by-Step Breakdown

#### Step 1: K-Means Clustering
- Divides the minority class data into **R clusters** (neighborhoods)
- Each cluster contains similar patients
- **Why?** Without clustering, SMOTE might create "impossible" synthetic patients that combine features from very different patient profiles

#### Step 2: SMOTE (per cluster)
- For each cluster, creates **synthetic (fake but realistic)** minority samples
- **How SMOTE works:**
  1. Pick a minority sample
  2. Find its k nearest neighbors (similar patients)
  3. Draw a line between them
  4. Create a new point somewhere on that line
- This **generalizes** the data rather than just duplicating

#### Step 3: Random Over Sampling (ROS)
- After SMOTE, some clusters may still be slightly imbalanced
- ROS duplicates minority samples until perfect 1:1 balance

### Paper's Claimed Results
| Dataset | Model | Accuracy |
|---------|-------|----------|
| Anemia (128 samples) | Random Forest | **97.3%** |
| Pima Diabetes | Random Forest | ~90% |

### Models Used in Base Paper
- **Random Forest** (best performer)
- **Naïve Bayes**
- **Support Vector Machine (SVM)**

---

## ❓ Why My Results Differ from the Paper

### My Replication Results
| Dataset | Model | Accuracy | Recall | AUC |
|---------|-------|----------|--------|-----|
| Pima Diabetes | Random Forest | 74.73% | 63.39% | 0.808 |
| Stroke (19:1 imbalance) | Random Forest | 92.27% | 6.82% | 0.798 |

### Why the Difference? (Important for Your Guide!)

#### 1. **Data Leakage in Base Paper**
The base paper likely applied SMOTE **before** splitting data into train/test sets. This means:
- Synthetic samples in training set are very similar to test set samples
- Model "memorizes" rather than "learns"
- Results in artificially high accuracy (99%)

**My Implementation:** I apply SMOTE only on **training data** after splitting (correct methodology).

#### 2. **Different Dataset**
- Base paper uses private Anemia dataset (128 samples only)
- I use public datasets: Pima Diabetes (768 samples), Stroke (5110 samples)
- Larger, real-world datasets are harder to classify

#### 3. **10-Fold Cross Validation**
- I use proper 10-fold CV which gives realistic estimates
- Each fold tests on completely unseen data

#### 4. **Realistic Results are Better**
- My 74-76% accuracy is **scientifically valid**
- It can be defended in viva/presentation
- 99% accuracy would raise questions about data leakage

---

## 🚀 My Contribution: Adaptive SMOTE-MRS

### Problem with Base Paper's Approach
The base paper applies SMOTE **uniformly** to all clusters. But in reality:
- Some clusters are already balanced
- Some clusters are noisy (outliers)
- Some clusters are borderline regions (important)

### My Solution: Adaptive Oversampling

```python
For each cluster:
    1. Calculate Imbalance Ratio (IR) = Majority / Minority
    2. If IR < threshold (1.5):
        → Cluster is balanced, skip SMOTE
    3. If IR >= threshold:
        → Cluster is imbalanced, apply SMOTE
```

### Why This is Better
1. **Avoids noise amplification** — Doesn't oversample already-balanced clusters
2. **Efficient** — Saves computation on balanced clusters
3. **Targeted** — Focuses resources on truly imbalanced regions

---

## 📊 Results Comparison

### Base SMOTE-MRS vs Adaptive SMOTE-MRS

| Dataset | Method | Accuracy | Recall | F1 | AUC |
|---------|--------|----------|--------|-----|-----|
| Pima Diabetes | Base | 74.73% | 63.39% | 63.43% | 0.808 |
| Pima Diabetes | **Adaptive** | **76.29%** | **66.35%** | **65.99%** | **0.810** |
| Stroke | Base | 92.27% | 6.82% | 7.87% | 0.798 |
| Stroke | **Adaptive** | **92.43%** | **8.02%** | **9.25%** | **0.801** |

### Improvements Achieved
| Dataset | Accuracy | Recall | AUC |
|---------|----------|--------|-----|
| Pima Diabetes | +1.56% | +2.96% | +0.16% |
| Stroke | +0.16% | +1.20% | +0.25% |

### Key Insight
The improvement in **Recall** is most important — it means we're catching more sick patients!

---

## 🔧 Technical Implementation

### Technologies Used
- **Python 3.x**
- **scikit-learn** — ML models, preprocessing
- **NumPy/Pandas** — Data manipulation
- **Matplotlib/Seaborn** — Visualizations
- **Kaggle Notebooks** — Execution environment

### Key Code Components

#### 1. Custom SMOTE Function
```python
def custom_resample(X, y, k_neighbors=3):
    # Find minority class
    # Find k nearest neighbors
    # Interpolate to create synthetic samples
```

#### 2. Base SMOTE-MRS Class
```python
class SMOTEMRS_Base:
    def fit_resample(X, y):
        # K-Means clustering
        # For each cluster: Apply SMOTE + ROS
```

#### 3. Adaptive SMOTE-MRS Class (My Innovation)
```python
class AdaptiveSMOTEMRS:
    def fit_resample(X, y):
        # K-Means clustering
        # For each cluster:
        #   Calculate IR
        #   If IR > threshold: Apply SMOTE
        #   Else: Keep as is
```

---

## 📈 Datasets Used

### 1. Pima Indians Diabetes
| Property | Value |
|----------|-------|
| Samples | 768 |
| Features | 8 (Glucose, BMI, Age, etc.) |
| Classes | Diabetic (268), Healthy (500) |
| Imbalance Ratio | 1.87:1 |
| Source | UCI Repository |

### 2. Stroke Prediction
| Property | Value |
|----------|-------|
| Samples | 5110 |
| Features | 10 (Age, Hypertension, Heart Disease, etc.) |
| Classes | Stroke (249), No Stroke (4861) |
| Imbalance Ratio | 19.52:1 |
| Source | Kaggle |

---

## ✅ What We Completed

- [x] Understood base paper's SMOTE-MRS algorithm
- [x] Implemented base paper replication in Python
- [x] Tested on Pima Diabetes dataset
- [x] Tested on Stroke Prediction dataset (highly imbalanced)
- [x] Implemented Adaptive SMOTE-MRS (our contribution)
- [x] Compared results with base paper approach
- [x] Generated visualizations (ROC curves, confusion matrices, comparison charts)

---

## 🔮 Future Work / What's Next

- [ ] Test with different IR thresholds (2.0, 3.0)
- [ ] Test with different number of clusters (R=7, R=10)
- [ ] Add XGBoost/LightGBM classifiers (more modern than RF)
- [ ] Create web interface for prediction demo
- [ ] Write final project report with IEEE format
- [ ] Prepare presentation slides

---

## 💡 Key Points for Guide Review

1. **Base paper claims 99% accuracy** — This is likely due to data leakage
2. **My 74-76% accuracy is scientifically correct** — Proper train/test split
3. **Adaptive SMOTE-MRS shows improvement** — Especially in Recall
4. **Tested on 2 datasets** — Proves generalizability
5. **Clear contribution** — Cluster-wise adaptive oversampling

---

## 📝 Glossary

| Term | Meaning |
|------|---------|
| **SMOTE** | Synthetic Minority Over-sampling Technique |
| **MRS** | Multi-Resolution Sampling |
| **IR** | Imbalance Ratio (Majority/Minority) |
| **Recall** | % of actual positives correctly identified |
| **AUC** | Area Under ROC Curve (overall classifier quality) |
| **K-Means** | Clustering algorithm that groups similar data |
| **Cross-Validation** | Testing method that uses all data for training and testing |

---

## 📂 Project Files

| File | Description |
|------|-------------|
| `uclandpima.py` | Base paper implementation on both datasets |
| `stroke.py` | Stroke dataset specific implementation |
| `smote_mrs_base.py` | Core SMOTE-MRS algorithm |
| `evaluate_base.py` | Evaluation pipeline |
| `PROJECT_DOCUMENTATION.md` | This file |

---

*Last Updated: January 10, 2026*
