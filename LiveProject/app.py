"""
Adaptive SMOTE-MRS Framework — Dataset Balancing Tool
======================================================
A Streamlit application for:
  1. Uploading an imbalanced medical dataset (CSV)
  2. Selecting the target/class column
  3. Analysing class distribution & imbalance ratio
  4. Applying the Adaptive SMOTE-MRS balancing strategy
  5. Comparing before/after class distributions
  6. Downloading the balanced dataset
  7. Training classifiers and displaying evaluation metrics
"""

import streamlit as st
import numpy as np
import pandas as pd
import random
import warnings
import io
import plotly.graph_objects as go
import plotly.express as px

from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Adaptive SMOTE-MRS Framework",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="st-"] {
    font-family: 'Inter', sans-serif;
}

.main .block-container {
    padding-top: 2rem;
    max-width: 1200px;
}

/* Hero */
.hero-card {
    background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 50%, #3b82f6 100%);
    border-radius: 22px;
    padding: 2.2rem 2.5rem;
    margin-bottom: 1.5rem;
    color: white;
    box-shadow: 0 20px 40px rgba(37,99,235,0.25);
}
.hero-card h1 {
    color: white !important;
    font-size: 2rem;
    font-weight: 800;
    margin: 0 0 0.6rem 0;
    letter-spacing: -0.02em;
}
.hero-card p {
    color: rgba(255,255,255,0.9);
    font-size: 0.95rem;
    margin: 0;
    line-height: 1.7;
}

/* Upload box */
.upload-zone {
    border: 2.5px dashed #93c5fd;
    background: linear-gradient(135deg, #eff6ff, #f8fbff);
    border-radius: 18px;
    padding: 2rem;
    text-align: center;
    margin: 1rem 0;
    transition: border-color 0.3s, background 0.3s;
}
.upload-zone:hover {
    border-color: #3b82f6;
    background: linear-gradient(135deg, #dbeafe, #eff6ff);
}

/* Stat cards */
.stat-card {
    background: linear-gradient(135deg, #f8fafc, #ffffff);
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 1.2rem 1.4rem;
    text-align: center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.04);
    transition: transform 0.2s, box-shadow 0.2s;
}
.stat-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.08);
}
.stat-label {
    font-size: 0.72rem;
    font-weight: 700;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.25rem;
}
.stat-value {
    font-size: 1.8rem;
    font-weight: 800;
    letter-spacing: -0.02em;
}

/* Badge */
.badge {
    display: inline-block;
    padding: 0.35rem 0.8rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.03em;
}
.badge-orange { background: #fef3c7; color: #b45309; }
.badge-green  { background: #dcfce7; color: #15803d; }
.badge-blue   { background: #dbeafe; color: #1d4ed8; }
.badge-red    { background: #fee2e2; color: #dc2626; }

/* Workflow steps */
.workflow-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 1.5rem 2rem;
    box-shadow: 0 4px 12px rgba(0,0,0,0.04);
}
.workflow-step {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    padding: 0.6rem 0;
    font-size: 0.92rem;
    color: #334155;
}
.step-num {
    background: linear-gradient(135deg, #2563eb, #3b82f6);
    color: white;
    width: 28px;
    height: 28px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.75rem;
    font-weight: 700;
    flex-shrink: 0;
}
.step-done {
    background: linear-gradient(135deg, #16a34a, #22c55e) !important;
}

/* Info note */
.note-box {
    background: linear-gradient(135deg, #fffbeb, #fef3c7);
    border: 1px solid #fde68a;
    border-radius: 14px;
    padding: 1rem 1.3rem;
    font-size: 0.82rem;
    color: #92400e;
    line-height: 1.6;
    margin-top: 1rem;
}

/* Success box */
.success-box {
    background: linear-gradient(135deg, #ecfdf5, #d1fae5);
    border: 1px solid #6ee7b7;
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    color: #065f46;
    font-size: 0.9rem;
    line-height: 1.6;
}

/* Section headers */
.section-header {
    font-size: 1.3rem;
    font-weight: 700;
    color: #0f172a;
    margin: 1.5rem 0 1rem;
    letter-spacing: -0.01em;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a, #1e293b);
}
section[data-testid="stSidebar"] * {
    color: white !important;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] .stSlider label {
    color: rgba(255,255,255,0.9) !important;
    font-weight: 600;
}

div[data-testid="stExpander"] {
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ML PIPELINE — Extracted from Codes.ipynb
# ═══════════════════════════════════════════════════════════════════════════════

def custom_resample(X, y, k_neighbors=3):
    """SMOTE-based synthetic sample generation for a single cluster."""
    X = np.array(X, dtype=float)
    y = np.array(y)
    counts = pd.Series(y).value_counts()
    minority_class = counts.idxmin()
    X_min = X[y == minority_class]
    n_min = len(X_min)

    if n_min > 1:
        nn = NearestNeighbors(n_neighbors=min(n_min, k_neighbors + 1)).fit(X_min)
        _, idxs = nn.kneighbors(X_min)

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


class AdaptiveSMOTEMRS:
    """
    Adaptive SMOTE-MRS: Applies SMOTE only to clusters with high Imbalance Ratio.
    Innovation: Calculates IR per cluster, skips balanced clusters → better quality
    synthetic data, no noise amplification.
    """
    def __init__(self, R=5, ir_threshold=1.5):
        self.R = R
        self.ir_threshold = ir_threshold

    def fit_resample(self, X, y):
        n_clusters = min(self.R, len(X))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        clusters = kmeans.fit_predict(X)

        X_res, y_res = [], []
        cluster_info = []

        for r in range(n_clusters):
            mask = clusters == r
            if not any(mask):
                continue

            X_sub, y_sub = X[mask], y[mask]
            unique_classes = np.unique(y_sub)

            if len(unique_classes) < 2:
                X_res.append(X_sub)
                y_res.append(y_sub)
                cluster_info.append({
                    "cluster": r, "size": len(y_sub),
                    "ir": float("inf"), "action": "Skipped (pure cluster)"
                })
                continue

            counts = pd.Series(y_sub).value_counts()
            ir = counts.max() / counts.min()

            if ir >= self.ir_threshold:
                X_s, y_s = custom_resample(X_sub, y_sub)
                X_res.append(X_s)
                y_res.append(y_s)
                cluster_info.append({
                    "cluster": r, "size": len(y_sub),
                    "ir": round(ir, 2), "action": f"SMOTE applied (IR={ir:.2f} ≥ {self.ir_threshold})"
                })
            else:
                X_res.append(X_sub)
                y_res.append(y_sub)
                cluster_info.append({
                    "cluster": r, "size": len(y_sub),
                    "ir": round(ir, 2), "action": f"Kept original (IR={ir:.2f} < {self.ir_threshold})"
                })

        return np.vstack(X_res), np.concatenate(y_res), cluster_info


class SMOTEMRS_Base:
    """Base SMOTE-MRS: Applies SMOTE to all mixed clusters regardless of IR."""
    def __init__(self, R=5):
        self.R = R

    def fit_resample(self, X, y):
        n_clusters = min(self.R, len(X))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        clusters = kmeans.fit_predict(X)

        X_res, y_res = [], []
        cluster_info = []

        for r in range(n_clusters):
            mask = clusters == r
            if not any(mask):
                continue

            X_sub, y_sub = X[mask], y[mask]
            if len(np.unique(y_sub)) > 1:
                X_s, y_s = custom_resample(X_sub, y_sub)
                X_res.append(X_s)
                y_res.append(y_s)
                cluster_info.append({
                    "cluster": r, "size": len(y_sub),
                    "ir": round(pd.Series(y_sub).value_counts().max() /
                                pd.Series(y_sub).value_counts().min(), 2),
                    "action": "SMOTE applied"
                })
            else:
                X_res.append(X_sub)
                y_res.append(y_sub)
                cluster_info.append({
                    "cluster": r, "size": len(y_sub),
                    "ir": float("inf"), "action": "Skipped (pure cluster)"
                })

        return np.vstack(X_res), np.concatenate(y_res), cluster_info


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_models(X_balanced, y_balanced, X_original, y_original):
    """Train and evaluate multiple classifiers using 10-fold CV on the balanced data."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_balanced)

    n_classes = len(np.unique(y_balanced))
    if n_classes < 2:
        return pd.DataFrame()

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                                  random_state=42, tree_method="hist", verbose=0),
        "LightGBM": LGBMClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                                    random_state=42, verbose=-1),
        "Naïve Bayes": GaussianNB(),
    }

    n_splits = min(10, min(pd.Series(y_balanced).value_counts()))
    if n_splits < 2:
        n_splits = 2

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = []

    for name, model in models.items():
        m = {"acc": [], "rec": [], "f1": [], "auc": []}
        for train_idx, test_idx in skf.split(X_scaled, y_balanced):
            X_tr, y_tr = X_scaled[train_idx], y_balanced[train_idx]
            X_te, y_te = X_scaled[test_idx], y_balanced[test_idx]

            model.fit(X_tr, y_tr)
            preds = model.predict(X_te)

            m["acc"].append(accuracy_score(y_te, preds))
            m["rec"].append(recall_score(y_te, preds, average="weighted", zero_division=0))
            m["f1"].append(f1_score(y_te, preds, average="weighted", zero_division=0))

            try:
                if n_classes == 2:
                    probs = model.predict_proba(X_te)[:, 1]
                    m["auc"].append(roc_auc_score(y_te, probs))
                else:
                    probs = model.predict_proba(X_te)
                    m["auc"].append(roc_auc_score(y_te, probs, multi_class="ovr", average="weighted"))
            except Exception:
                m["auc"].append(0.0)

        results.append({
            "Model": name,
            "Accuracy": round(np.mean(m["acc"]), 4),
            "Recall": round(np.mean(m["rec"]), 4),
            "F1 Score": round(np.mean(m["f1"]), 4),
            "AUC-ROC": round(np.mean(m["auc"]), 4),
        })

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    pipeline_type = st.radio(
        "🔬 Balancing Method",
        ["Adaptive SMOTE-MRS", "Base SMOTE-MRS"],
        index=0,
        help="Adaptive applies SMOTE only to high-IR clusters (threshold ≥ IR)",
    )

    if pipeline_type == "Adaptive SMOTE-MRS":
        n_clusters = st.slider("📊 Number of Clusters (R)", 2, 15, 5,
                                help="Number of KMeans clusters for the MRS strategy")
        ir_threshold = st.slider("📏 IR Threshold", 1.0, 5.0, 1.5, 0.1,
                                  help="Imbalance Ratio threshold — SMOTE is applied only if cluster IR ≥ this value")
    else:
        n_clusters = st.slider("📊 Number of Clusters (R)", 2, 15, 5)
        ir_threshold = None

    run_evaluation = st.checkbox("📈 Run Model Evaluation", value=False,
                                  help="Train classifiers on the balanced data and show metrics")

    st.markdown("---")

    # Workflow steps sidebar
    st.markdown("### 📋 Workflow")
    steps = [
        "Upload imbalanced dataset",
        "Select target class column",
        "Analyse class distribution",
        "Apply Adaptive SMOTE-MRS",
        "Review balanced results",
        "Download balanced dataset",
    ]

    upload_done = "uploaded_df" in st.session_state
    target_done = "target_col" in st.session_state and st.session_state.get("target_col")
    balanced_done = "balanced_df" in st.session_state

    for i, step in enumerate(steps, 1):
        done = False
        if i == 1 and upload_done:
            done = True
        elif i == 2 and target_done:
            done = True
        elif i >= 3 and balanced_done:
            done = True

        icon = "✅" if done else f"**{i}.**"
        st.markdown(f"{icon} {step}")

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; opacity:0.5; font-size:0.72rem;'>"
        "MSc Research Project<br>Adaptive SMOTE-MRS Framework</div>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════════════════

# Hero
st.markdown(
    '<div class="hero-card">'
    '  <h1>⚗️ Adaptive SMOTE-MRS Framework</h1>'
    '  <p>Intelligent preprocessing platform for imbalanced medical dataset analysis. '
    'Upload a CSV dataset, select the target class column, inspect class distribution '
    'statistics, and generate a balanced output dataset using the Adaptive SMOTE-MRS workflow.</p>'
    '</div>',
    unsafe_allow_html=True,
)

# ─── Step 1: Upload Dataset ──────────────────────────────────────────────────
st.markdown('<div class="section-header">📁 1. Upload Dataset</div>', unsafe_allow_html=True)
st.caption("Supports CSV files with a header row. Medical datasets like Pima Diabetes, CKD, or Stroke are ideal.")

st.markdown('<div class="upload-zone">', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Select a medical dataset file (.csv)",
    type=["csv"],
    key="csv_uploader",
    label_visibility="collapsed",
)
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    try:
        raw_df = pd.read_csv(uploaded_file)
        st.session_state["uploaded_df"] = raw_df
        st.session_state["filename"] = uploaded_file.name
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        st.stop()

if "uploaded_df" not in st.session_state:
    st.info("👆 Upload a CSV dataset to get started.")
    st.stop()

raw_df = st.session_state["uploaded_df"]

# ─── Preview ────────────────────────────────────────────────────────────────
with st.expander(f"👁️ Dataset Preview — {st.session_state.get('filename', 'dataset.csv')}", expanded=False):
    st.dataframe(raw_df.head(10), use_container_width=True, hide_index=True)
    st.caption(f"{raw_df.shape[0]} rows × {raw_df.shape[1]} columns")

# ─── Step 2: Select Target Column ────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-header">🎯 2. Select Target / Class Column</div>', unsafe_allow_html=True)

target_col = st.selectbox(
    "Choose the column that contains the class labels:",
    options=["— Select —"] + list(raw_df.columns),
    index=0,
    key="target_selector",
)

if target_col == "— Select —":
    st.warning("⬆️ Please select the target class column to proceed.")
    st.stop()

st.session_state["target_col"] = target_col

# ─── Prepare data ────────────────────────────────────────────────────────────
df = raw_df.copy()

# Encode target if string
target_is_string = df[target_col].dtype == "object"
target_label_map = None
if target_is_string:
    le_target = LabelEncoder()
    df[target_col] = le_target.fit_transform(df[target_col].astype(str))
    target_label_map = dict(zip(le_target.transform(le_target.classes_), le_target.classes_))

# Handle missing values for features
feature_cols = [c for c in df.columns if c != target_col]
for col in feature_cols:
    if df[col].dtype == "object":
        df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "unknown", inplace=True)
        le_feat = LabelEncoder()
        df[col] = le_feat.fit_transform(df[col].astype(str))
    else:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col].fillna(df[col].median(), inplace=True)

df = df.dropna(subset=[target_col])
y_original = df[target_col].astype(int).values
X_original = df.drop(target_col, axis=1).values.astype(float)

# Class distribution
class_counts = pd.Series(y_original).value_counts().sort_index()
majority_label = class_counts.idxmax()
minority_label = class_counts.idxmin()
majority_count = class_counts.max()
minority_count = class_counts.min()
ir_before = majority_count / minority_count if minority_count > 0 else float("inf")

# ─── Step 3: Analysis Dashboard ──────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-header">📊 3. Analysis Dashboard</div>', unsafe_allow_html=True)

# Stat cards
s1, s2, s3, s4 = st.columns(4)
with s1:
    st.markdown(
        '<div class="stat-card"><div class="stat-label">Rows</div>'
        f'<div class="stat-value" style="color:#2563eb">{len(y_original):,}</div></div>',
        unsafe_allow_html=True,
    )
with s2:
    st.markdown(
        '<div class="stat-card"><div class="stat-label">Columns</div>'
        f'<div class="stat-value" style="color:#2563eb">{len(feature_cols)}</div></div>',
        unsafe_allow_html=True,
    )
with s3:
    maj_display = target_label_map[majority_label] if target_label_map else str(majority_label)
    st.markdown(
        '<div class="stat-card"><div class="stat-label">Majority Class</div>'
        f'<div class="stat-value" style="color:#16a34a">{maj_display}</div></div>',
        unsafe_allow_html=True,
    )
with s4:
    min_display = target_label_map[minority_label] if target_label_map else str(minority_label)
    st.markdown(
        '<div class="stat-card"><div class="stat-label">Minority Class</div>'
        f'<div class="stat-value" style="color:#dc2626">{min_display}</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# Class distribution bar chart
dist_labels = [target_label_map[k] if target_label_map else str(k) for k in class_counts.index]
colors = ["#3b82f6" if k == majority_label else "#f59e0b" for k in class_counts.index]

fig_dist = go.Figure(go.Bar(
    x=dist_labels,
    y=class_counts.values,
    marker=dict(color=colors, line=dict(width=0)),
    text=class_counts.values,
    textposition="auto",
    textfont=dict(size=14, family="Inter", color="white"),
))
fig_dist.update_layout(
    title="Before Processing — Class Distribution",
    title_font=dict(size=15, family="Inter"),
    height=320,
    margin=dict(l=40, r=20, t=50, b=40),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(showgrid=False, title="Class"),
    yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.06)", title="Count"),
    font=dict(family="Inter"),
)
st.plotly_chart(fig_dist, use_container_width=True)

# Summary table
st.markdown("**Imbalance Summary**")
summary_cols = st.columns(3)
with summary_cols[0]:
    st.metric("Majority Count", f"{majority_count:,}")
with summary_cols[1]:
    st.metric("Minority Count", f"{minority_count:,}")
with summary_cols[2]:
    st.metric("Imbalance Ratio", f"{ir_before:.2f} : 1")

st.markdown(
    f'<span class="badge badge-orange">⚠️ Imbalanced Dataset</span> &nbsp; '
    f'The minority class has <strong>{minority_count:,}</strong> samples vs '
    f'<strong>{majority_count:,}</strong> majority samples.',
    unsafe_allow_html=True,
)


# ─── Step 4: Apply Balancing ─────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-header">⚗️ 4. Apply Adaptive SMOTE-MRS</div>', unsafe_allow_html=True)

col_btn, col_info = st.columns([1, 2])
with col_btn:
    balance_clicked = st.button(
        "🚀 Generate Balanced Dataset",
        type="primary",
        use_container_width=True,
    )
with col_info:
    method = pipeline_type
    extra = f" (R={n_clusters}, IR threshold={ir_threshold})" if ir_threshold else f" (R={n_clusters})"
    st.markdown(
        f'<div class="note-box">Method: <strong>{method}{extra}</strong><br>'
        'The algorithm clusters the data via K-Means, then selectively applies '
        'SMOTE only to clusters with high imbalance ratios.</div>',
        unsafe_allow_html=True,
    )

if balance_clicked:
    with st.spinner("⏳ Applying Adaptive SMOTE-MRS balancing..."):
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_original)

        # Apply pipeline
        if pipeline_type == "Adaptive SMOTE-MRS":
            engine = AdaptiveSMOTEMRS(R=n_clusters, ir_threshold=ir_threshold)
        else:
            engine = SMOTEMRS_Base(R=n_clusters)

        X_balanced, y_balanced, cluster_info = engine.fit_resample(X_scaled, y_original)

        # Inverse-transform the features back to original scale
        X_balanced_original = scaler.inverse_transform(X_balanced)

        # Build the balanced dataframe
        balanced_df = pd.DataFrame(X_balanced_original, columns=feature_cols)
        if target_label_map:
            balanced_df[target_col] = pd.Series(y_balanced).map(target_label_map)
        else:
            balanced_df[target_col] = y_balanced

        st.session_state["balanced_df"] = balanced_df
        st.session_state["y_balanced"] = y_balanced
        st.session_state["X_balanced_scaled"] = X_balanced
        st.session_state["cluster_info"] = cluster_info


# ─── Step 5: Results ─────────────────────────────────────────────────────────
if "balanced_df" in st.session_state:
    balanced_df = st.session_state["balanced_df"]
    y_balanced = st.session_state["y_balanced"]
    cluster_info = st.session_state["cluster_info"]

    st.markdown("---")
    st.markdown('<div class="section-header">✅ 5. Balancing Results</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="success-box">'
        '✅ <strong>Dataset balanced successfully!</strong> '
        'The Adaptive SMOTE-MRS pipeline has generated synthetic samples for '
        'under-represented classes to achieve a balanced distribution.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # After-balancing stats
    balanced_counts = pd.Series(y_balanced).value_counts().sort_index()
    after_majority = balanced_counts.max()
    after_minority = balanced_counts.min()
    ir_after = after_majority / after_minority if after_minority > 0 else 1.0

    # Comparison table
    comp_data = {
        "Metric": ["Majority Count", "Minority Count", "Imbalance Ratio", "Total Samples", "Status"],
        "Before Processing": [
            f"{majority_count:,}",
            f"{minority_count:,}",
            f"{ir_before:.2f} : 1",
            f"{len(y_original):,}",
            "⚠️ Imbalanced",
        ],
        "After Adaptive SMOTE-MRS": [
            f"{after_majority:,}",
            f"{after_minority:,}",
            f"{ir_after:.2f} : 1",
            f"{len(y_balanced):,}",
            "✅ Balanced",
        ],
    }
    st.table(pd.DataFrame(comp_data))

    # Before vs After bar chart
    before_labels = [target_label_map[k] if target_label_map else str(k) for k in class_counts.index]
    after_labels = [target_label_map[k] if target_label_map else str(k) for k in balanced_counts.index]

    fig_compare = go.Figure()
    fig_compare.add_trace(go.Bar(
        name="Before",
        x=before_labels,
        y=class_counts.values,
        marker_color="rgba(59,130,246,0.5)",
        text=class_counts.values,
        textposition="auto",
    ))
    fig_compare.add_trace(go.Bar(
        name="After SMOTE-MRS",
        x=after_labels,
        y=balanced_counts.values,
        marker_color="#16a34a",
        text=balanced_counts.values,
        textposition="auto",
    ))
    fig_compare.update_layout(
        title="Before vs After — Class Distribution",
        barmode="group",
        height=350,
        margin=dict(l=40, r=20, t=50, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, title="Class"),
        yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.06)", title="Count"),
        font=dict(family="Inter"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_compare, use_container_width=True)

    # Cluster details
    with st.expander("🔍 Cluster-Level Details", expanded=False):
        cluster_df = pd.DataFrame(cluster_info)
        cluster_df.columns = ["Cluster", "Samples", "IR", "Action"]
        st.dataframe(cluster_df, use_container_width=True, hide_index=True)

    # Balanced data preview
    with st.expander("👁️ Balanced Dataset Preview", expanded=False):
        st.dataframe(balanced_df.head(15), use_container_width=True, hide_index=True)
        st.caption(f"Showing first 15 of {len(balanced_df):,} rows.")

    # ─── Step 6: Download ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">📥 6. Download Balanced Dataset</div>', unsafe_allow_html=True)

    csv_data = balanced_df.to_csv(index=False).encode('utf-8')

    d1, d2 = st.columns([1, 2])
    with d1:
        st.download_button(
            label="⬇️ Download Balanced CSV",
            data=csv_data,
            file_name="adaptive_smote_mrs_balanced_dataset.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True,
        )
    with d2:
        st.markdown(
            f'<div class="note-box">'
            f'File: <strong>adaptive_smote_mrs_balanced_dataset.csv</strong><br>'
            f'Size: {len(balanced_df):,} rows × {len(balanced_df.columns)} columns<br>'
            f'New synthetic samples added: <strong>{len(y_balanced) - len(y_original):,}</strong>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ─── Optional: Model Evaluation ──────────────────────────────────────────
    if run_evaluation:
        st.markdown("---")
        st.markdown('<div class="section-header">📈 7. Model Evaluation on Balanced Data</div>',
                    unsafe_allow_html=True)

        st.caption("Training Random Forest, XGBoost, LightGBM, and Naïve Bayes using "
                    "Stratified K-Fold Cross-Validation on the balanced dataset.")

        with st.spinner("🔄 Training and evaluating models..."):
            X_bal = st.session_state["X_balanced_scaled"]
            y_bal = st.session_state["y_balanced"]
            results_df = evaluate_models(X_bal, y_bal, X_original, y_original)

        if not results_df.empty:
            # Metric highlight cards
            best_model = results_df.loc[results_df["F1 Score"].idxmax()]
            m_cols = st.columns(4)
            metrics = [
                ("Best Accuracy", results_df["Accuracy"].max(), "#3b82f6"),
                ("Best Recall", results_df["Recall"].max(), "#10b981"),
                ("Best F1 Score", results_df["F1 Score"].max(), "#8b5cf6"),
                ("Best AUC-ROC", results_df["AUC-ROC"].max(), "#f59e0b"),
            ]
            for col, (label, value, color) in zip(m_cols, metrics):
                with col:
                    st.markdown(
                        f'<div class="stat-card"><div class="stat-label">{label}</div>'
                        f'<div class="stat-value" style="color:{color}">{value:.4f}</div></div>',
                        unsafe_allow_html=True,
                    )

            st.markdown("<br>", unsafe_allow_html=True)

            # Full results table
            st.dataframe(results_df, use_container_width=True, hide_index=True)

            st.markdown(
                '<div class="note-box">'
                f'🏆 Best performing model (by F1 Score): <strong>{best_model["Model"]}</strong> — '
                f'Accuracy: {best_model["Accuracy"]:.4f}, Recall: {best_model["Recall"]:.4f}, '
                f'F1: {best_model["F1 Score"]:.4f}, AUC: {best_model["AUC-ROC"]:.4f}'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.warning("Could not evaluate models. The dataset may have too few samples per class.")


# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#94a3b8; font-size:0.78rem; padding:1rem 0 2rem;'>"
    "Adaptive SMOTE-MRS Framework &nbsp;|&nbsp; "
    "MSc Research Project &nbsp;|&nbsp; "
    "Intelligent preprocessing for imbalanced medical datasets"
    "</div>",
    unsafe_allow_html=True,
)
