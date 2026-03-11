"""
=============================================================================
MODEL EVALUATION CURVES — Adaptive SMOTE-MRS Project
=============================================================================
Generates for ALL 3 datasets (Pima, CKD, Stroke):
  1. ROC Curves (Base vs Adaptive, 3 models)
  2. Precision-Recall Curves
  3. F1-Confidence Curves (F1 vs threshold)
  4. Confusion Matrices

Uses: Random Forest, XGBoost, LightGBM
Evaluation: 10-Fold Stratified Cross-Validation
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import requests
import warnings
import random

from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score,
    confusion_matrix, precision_score
)
from sklearn.preprocessing import StandardScaler, LabelEncoder

warnings.filterwarnings('ignore')

# ============================================================================
# STYLE CONFIG
# ============================================================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.labelsize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'figure.facecolor': 'white',
})
sns.set_theme(style="whitegrid")

# Color scheme for models
COLORS = {
    'Random Forest_Base':     '#2196F3',  # Blue
    'Random Forest_Adaptive': '#0D47A1',  # Dark Blue
    'XGBoost_Base':           '#FF9800',  # Orange
    'XGBoost_Adaptive':       '#E65100',  # Dark Orange
    'LightGBM_Base':          '#4CAF50',  # Green
    'LightGBM_Adaptive':      '#1B5E20',  # Dark Green
}
LINESTYLES = {'Base': '--', 'Adaptive': '-'}


# ============================================================================
# 1. SMOTE IMPLEMENTATIONS
# ============================================================================
def custom_resample(X, y, k_neighbors=3):
    X = np.array(X)
    y = np.array(y)
    counts = pd.Series(y).value_counts()
    minority_class = counts.idxmin()
    X_min = X[y == minority_class]
    n_min = len(X_min)

    if n_min > 1:
        nn = NearestNeighbors(n_neighbors=min(n_min, k_neighbors + 1)).fit(X_min)
        dist, idxs = nn.kneighbors(X_min)

        synthetic = []
        for _ in range(counts.max() - n_min):
            i = random.randint(0, n_min - 1)
            neighbor = random.choice(idxs[i][1:])
            diff = X_min[neighbor] - X_min[i]
            synthetic.append(X_min[i] + random.random() * diff)

        if synthetic:
            X = np.vstack([X, np.array(synthetic)])
            y = np.append(y, [minority_class] * len(synthetic))

    return X, y


class SMOTEMRS_Base:
    def __init__(self, R=5):
        self.R = R

    def fit_resample(self, X, y):
        kmeans = KMeans(n_clusters=self.R, random_state=42, n_init='auto')
        clusters = kmeans.fit_predict(X)

        X_res, y_res = [], []
        for r in range(self.R):
            mask = (clusters == r)
            if not any(mask): continue

            X_sub, y_sub = X[mask], y[mask]
            if len(np.unique(y_sub)) > 1:
                X_s, y_s = custom_resample(X_sub, y_sub)
                X_res.append(X_s)
                y_res.append(y_s)
            else:
                X_res.append(X_sub)
                y_res.append(y_sub)

        return np.vstack(X_res), np.concatenate(y_res)


class AdaptiveSMOTEMRS:
    def __init__(self, R=5, ir_threshold=1.5):
        self.R = R
        self.ir_threshold = ir_threshold

    def fit_resample(self, X, y):
        kmeans = KMeans(n_clusters=self.R, random_state=42, n_init='auto')
        clusters = kmeans.fit_predict(X)

        X_res, y_res = [], []

        for r in range(self.R):
            mask = (clusters == r)
            if not any(mask):
                continue

            X_sub, y_sub = X[mask], y[mask]
            unique_classes = np.unique(y_sub)

            if len(unique_classes) < 2:
                X_res.append(X_sub)
                y_res.append(y_sub)
                continue

            counts = pd.Series(y_sub).value_counts()
            ir = counts.max() / counts.min()

            if ir >= self.ir_threshold:
                X_s, y_s = custom_resample(X_sub, y_sub)
                X_res.append(X_s)
                y_res.append(y_s)
            else:
                X_res.append(X_sub)
                y_res.append(y_sub)

        return np.vstack(X_res), np.concatenate(y_res)


# ============================================================================
# 2. DATASET LOADERS
# ============================================================================
def load_diabetes():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    cols = ['Preg', 'Glu', 'BP', 'Skin', 'Ins', 'BMI', 'Ped', 'Age', 'Outcome']
    df = pd.read_csv(io.StringIO(requests.get(url).content.decode('utf-8')), names=cols)
    X = StandardScaler().fit_transform(df.drop('Outcome', axis=1))
    y = df['Outcome'].values
    return X, y, "Pima Diabetes"


def load_ckd():
    # UPDATE THIS PATH for your Kaggle notebook
    df = pd.read_csv('/kaggle/input/datasets/mansoordaku/ckdisease/kidney_disease.csv')

    if 'id' in df.columns:
        df = df.drop('id', axis=1)

    df = df.replace('?', np.nan)
    df['classification'] = df['classification'].str.strip()
    df['classification'] = df['classification'].map({'ckd': 1, 'notckd': 0})
    df = df.dropna(subset=['classification'])

    for col in df.columns:
        if col == 'classification':
            continue
        if df[col].dtype == 'object':
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)

    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col].astype(str))

    X = StandardScaler().fit_transform(df.drop('classification', axis=1))
    y = df['classification'].astype(int).values
    return X, y, "UCI CKD"


def load_stroke():
    # UPDATE THIS PATH for your Kaggle notebook
    df = pd.read_csv('/kaggle/input/datasets/fedesoriano/stroke-prediction-dataset/healthcare-dataset-stroke-data.csv')
    df = df.drop('id', axis=1)
    df['bmi'].fillna(df['bmi'].median(), inplace=True)

    le = LabelEncoder()
    for col in ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']:
        df[col] = le.fit_transform(df[col].astype(str))

    X = StandardScaler().fit_transform(df.drop('stroke', axis=1))
    y = df['stroke'].values
    return X, y, "Stroke"


# ============================================================================
# 3. COLLECT PREDICTIONS (all folds aggregated)
# ============================================================================
def collect_predictions(X, y, method_name, models_dict):
    """
    Run 10-fold CV and collect all predictions.
    Returns dict: {model_name: {'y_true': [], 'y_pred': [], 'y_prob': []}}
    """
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    all_preds = {}

    for model_name, model_fn in models_dict.items():
        y_true_all, y_pred_all, y_prob_all = [], [], []

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_te, y_te = X[test_idx], y[test_idx]

            # Apply SMOTE method
            if method_name == 'Base':
                engine = SMOTEMRS_Base(R=5)
            else:
                engine = AdaptiveSMOTEMRS(R=5, ir_threshold=1.5)

            X_res, y_res = engine.fit_resample(X_tr, y_tr)

            # Get fresh model instance
            model = model_fn()
            model.fit(X_res, y_res)

            preds = model.predict(X_te)
            probs = model.predict_proba(X_te)[:, 1]

            y_true_all.extend(y_te)
            y_pred_all.extend(preds)
            y_prob_all.extend(probs)

        all_preds[model_name] = {
            'y_true': np.array(y_true_all),
            'y_pred': np.array(y_pred_all),
            'y_prob': np.array(y_prob_all),
        }
        print(f"  ✅ {method_name} - {model_name} done")

    return all_preds


# ============================================================================
# 4. PLOTTING FUNCTIONS
# ============================================================================

def plot_roc_curves(base_preds, adaptive_preds, dataset_name, models_list):
    """Plot ROC curves: 3 models × 2 methods = 6 lines"""
    fig, ax = plt.subplots(figsize=(8, 7))

    for model_name in models_list:
        for method_name, preds in [('Base', base_preds), ('Adaptive', adaptive_preds)]:
            y_true = preds[model_name]['y_true']
            y_prob = preds[model_name]['y_prob']
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc_val = roc_auc_score(y_true, y_prob)

            color = COLORS[f'{model_name}_{method_name}']
            ls = LINESTYLES[method_name]
            label = f'{model_name} ({method_name}) AUC={auc_val:.4f}'
            ax.plot(fpr, tpr, color=color, linestyle=ls, linewidth=2, label=label)

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, linewidth=1, label='Random (AUC=0.5)')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve — {dataset_name}\nBase vs Adaptive SMOTE-MRS')
    ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f'roc_{dataset_name.lower().replace(" ", "_")}.png'
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"  📁 Saved: {fname}")


def plot_pr_curves(base_preds, adaptive_preds, dataset_name, models_list):
    """Plot Precision-Recall curves"""
    fig, ax = plt.subplots(figsize=(8, 7))

    for model_name in models_list:
        for method_name, preds in [('Base', base_preds), ('Adaptive', adaptive_preds)]:
            y_true = preds[model_name]['y_true']
            y_prob = preds[model_name]['y_prob']
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            ap = average_precision_score(y_true, y_prob)

            color = COLORS[f'{model_name}_{method_name}']
            ls = LINESTYLES[method_name]
            label = f'{model_name} ({method_name}) AP={ap:.4f}'
            ax.plot(recall, precision, color=color, linestyle=ls, linewidth=2, label=label)

    # Baseline (proportion of positive class)
    pos_rate = np.mean(base_preds[models_list[0]]['y_true'])
    ax.axhline(y=pos_rate, color='gray', linestyle=':', alpha=0.5,
               label=f'Baseline (Pos Rate={pos_rate:.3f})')

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'Precision-Recall Curve — {dataset_name}\nBase vs Adaptive SMOTE-MRS')
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f'pr_{dataset_name.lower().replace(" ", "_")}.png'
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"  📁 Saved: {fname}")


def plot_f1_confidence(base_preds, adaptive_preds, dataset_name, models_list):
    """Plot F1 Score vs Classification Threshold"""
    fig, ax = plt.subplots(figsize=(8, 7))

    for model_name in models_list:
        for method_name, preds in [('Base', base_preds), ('Adaptive', adaptive_preds)]:
            y_true = preds[model_name]['y_true']
            y_prob = preds[model_name]['y_prob']

            thresholds = np.arange(0.05, 0.96, 0.02)
            f1_scores = []
            for t in thresholds:
                y_pred_t = (y_prob >= t).astype(int)
                if len(np.unique(y_pred_t)) < 2:
                    f1_scores.append(0)
                else:
                    f1_scores.append(f1_score(y_true, y_pred_t))

            color = COLORS[f'{model_name}_{method_name}']
            ls = LINESTYLES[method_name]

            # Find best threshold
            best_idx = np.argmax(f1_scores)
            best_t = thresholds[best_idx]
            best_f1 = f1_scores[best_idx]

            label = f'{model_name} ({method_name}) Best F1={best_f1:.3f} @{best_t:.2f}'
            ax.plot(thresholds, f1_scores, color=color, linestyle=ls, linewidth=2, label=label)

            # Mark the best point
            ax.plot(best_t, best_f1, 'o', color=color, markersize=7, zorder=5)

    ax.set_xlabel('Confidence Threshold')
    ax.set_ylabel('F1 Score')
    ax.set_title(f'F1-Confidence Curve — {dataset_name}\nBase vs Adaptive SMOTE-MRS')
    ax.legend(loc='best', fontsize=7, framealpha=0.9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, max(0.5, ax.get_ylim()[1] + 0.05)])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f'f1_confidence_{dataset_name.lower().replace(" ", "_")}.png'
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"  📁 Saved: {fname}")


def plot_confusion_matrices(base_preds, adaptive_preds, dataset_name, models_list):
    """Plot confusion matrices: 2 rows (Base/Adaptive) × 3 cols (models)"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f'Confusion Matrices — {dataset_name}\nTop: Base SMOTE-MRS | Bottom: Adaptive SMOTE-MRS',
                 fontsize=15, fontweight='bold', y=1.03)

    for col, model_name in enumerate(models_list):
        for row, (method_name, preds) in enumerate([('Base', base_preds), ('Adaptive', adaptive_preds)]):
            ax = axes[row][col]
            y_true = preds[model_name]['y_true']
            y_pred = preds[model_name]['y_pred']

            cm = confusion_matrix(y_true, y_pred)
            # Normalize for display (percentages)
            cm_pct = cm.astype('float') / cm.sum() * 100

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       linewidths=1.5, linecolor='white',
                       xticklabels=['Predicted 0', 'Predicted 1'],
                       yticklabels=['Actual 0', 'Actual 1'],
                       cbar=False)

            # Add percentage annotations
            for ii in range(cm.shape[0]):
                for jj in range(cm.shape[1]):
                    ax.text(jj + 0.5, ii + 0.7, f'({cm_pct[ii, jj]:.1f}%)',
                           ha='center', va='center', fontsize=8, color='gray')

            acc = accuracy_score(y_true, y_pred)
            rec = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)

            ax.set_title(f'{model_name} ({method_name})\n'
                        f'Acc={acc:.3f} | Rec={rec:.3f} | F1={f1:.3f}',
                        fontsize=10, fontweight='bold')

    plt.tight_layout()
    fname = f'cm_{dataset_name.lower().replace(" ", "_")}.png'
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"  📁 Saved: {fname}")


# ============================================================================
# 5. MAIN — RUN EVERYTHING
# ============================================================================

# Model factory functions
models_dict = {
    'Random Forest': lambda: RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'XGBoost': lambda: XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                      random_state=42, tree_method='hist', n_jobs=-1, verbose=0),
    'LightGBM': lambda: LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                        is_unbalance=True, random_state=42, n_jobs=-1, verbose=-1),
}
model_names = ['Random Forest', 'XGBoost', 'LightGBM']

print("=" * 60)
print("  GENERATING MODEL EVALUATION CURVES")
print("  (ROC, PR, F1-Confidence, Confusion Matrix)")
print("=" * 60)

all_datasets = []

# Dataset 1: Pima Diabetes
print("\n📊 Loading Pima Diabetes...")
X1, y1, name1 = load_diabetes()
print(f"   Samples: {len(y1)}, Class dist: {np.bincount(y1)}")
all_datasets.append((X1, y1, name1))

# Dataset 2: UCI CKD
print("\n📊 Loading UCI CKD...")
X2, y2, name2 = load_ckd()
print(f"   Samples: {len(y2)}, Class dist: {np.bincount(y2)}")
all_datasets.append((X2, y2, name2))

# Dataset 3: Stroke
print("\n📊 Loading Stroke...")
X3, y3, name3 = load_stroke()
print(f"   Samples: {len(y3)}, Class dist: {np.bincount(y3)}")
all_datasets.append((X3, y3, name3))


# Run for each dataset
for X, y, dataset_name in all_datasets:
    print(f"\n{'='*60}")
    print(f"  DATASET: {dataset_name}")
    print(f"{'='*60}")

    # Collect predictions with Base SMOTE-MRS
    print(f"\n🔵 Running Base SMOTE-MRS...")
    base_preds = collect_predictions(X, y, 'Base', models_dict)

    # Collect predictions with Adaptive SMOTE-MRS
    print(f"\n🔴 Running Adaptive SMOTE-MRS...")
    adaptive_preds = collect_predictions(X, y, 'Adaptive', models_dict)

    # Generate all 4 chart types
    print(f"\n📊 Generating ROC Curves...")
    plot_roc_curves(base_preds, adaptive_preds, dataset_name, model_names)

    print(f"\n📊 Generating Precision-Recall Curves...")
    plot_pr_curves(base_preds, adaptive_preds, dataset_name, model_names)

    print(f"\n📊 Generating F1-Confidence Curves...")
    plot_f1_confidence(base_preds, adaptive_preds, dataset_name, model_names)

    print(f"\n📊 Generating Confusion Matrices...")
    plot_confusion_matrices(base_preds, adaptive_preds, dataset_name, model_names)


# ============================================================================
# DONE
# ============================================================================
print("\n" + "=" * 60)
print("  🎉 ALL EVALUATION CURVES GENERATED!")
print("=" * 60)
print("\nFiles saved per dataset:")
print("  📁 roc_<dataset>.png            — ROC Curves")
print("  📁 pr_<dataset>.png             — Precision-Recall Curves")
print("  📁 f1_confidence_<dataset>.png   — F1 vs Threshold")
print("  📁 cm_<dataset>.png             — Confusion Matrices")
print(f"\nTotal: {3 * 4} charts generated")
print("=" * 60)
