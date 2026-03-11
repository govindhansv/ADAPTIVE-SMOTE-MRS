# Adaptive SMOTE-MRS

An advanced data preprocessing and machine learning project focusing on balancing highly imbalanced datasets. This project introduces **Adaptive SMOTE-MRS**, a technique for creating synthetic samples more intelligently, leading to superior recall and overall model performance compared to standard balancing methods.

### 🌐 Live Hosted Link

[🔗 **Try the Adaptive SMOTE-MRS Live App here**](https://adaptive-smote-mrs.streamlit.app/) *(Replace `#` with your live hosted Streamlit URL, e.g., Streamlit Community Cloud or Heroku)*

---

## 📖 Project Overview

Imbalanced datasets present a significant challenge in machine learning, often leading to models that accurately predict the majority class but fail critically on the minority class (e.g., detecting a rare disease or fraud).

**Adaptive SMOTE-MRS** is an evolution of Synthetic Minority Over-sampling Technique (SMOTE), designed to adaptively synthesize data points and minimize noise insertion. 

### Key Features
- **Data Balancing**: Upload your imbalanced `.csv` datasets and balance them instantly using Adaptive SMOTE-MRS.
- **Interactive Web App**: A user-friendly Streamlit application where users can visualize class distributions before and after balancing.
- **Model Evaluation**: Compare performance metrics (Precision, Recall, F1-Score, ROC AUC, PR Curves) using robust machine learning models:
  - XGBoost
  - LightGBM
  - Random Forest
- **Performance Visualizations**: High-quality plots including Confusion Matrices, ROC Curves, Precision-Recall Curves, and Radar charts to deeply evaluate minority class performance.

---

## 🚀 How to Run Locally

### Prerequisites
Make sure you have Python 3.10+ installed.

### 1. Clone the repository
```bash
git clone https://github.com/your-username/ADAPTIVE-SMOTE-MRS.git
cd ADAPTIVE-SMOTE-MRS
```

### 2. Set up a virtual environment (Recommended)
```bash
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On Mac/Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
cd LiveProject
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
```bash
streamlit run app.py
```
The application will automatically open in your default browser at `http://localhost:8501`.

---

## 📂 Project Structure

```text
ADAPTIVE-SMOTE-MRS/
├── datasets/                 # Sample imbalanced datasets (e.g., stroke dataset)
├── LiveProject/              # Contains the Streamlit Web Application
│   ├── app.py                # Main Streamlit app script
│   └── requirements.txt      # Python dependencies for the app
├── Project Report/           # Documentation, presentations, and draft reports
├── results and others/       # Visualizations, diagrams, and evaluation notebooks
├── Generated auc roc f1 cm/  # Saved generated metric evaluation graphs
└── README.md                 # Project documentation (this file)
```

---

## 🔬 Results & Discoveries
Using Adaptive SMOTE-MRS significantly improves minority class **Recall** while maintaining competitive precision scores across standard models. Detailed results, including comprehensive reports on performance comparisons with benchmark techniques, can be found in the `/Project Report` directory.

---

## 🛡️ License & Acknowledgements

- Built using [Streamlit](https://streamlit.io/)
- Machine learning operations powered by [Scikit-Learn](https://scikit-learn.org/), [XGBoost](https://xgboost.readthedocs.io/), and [LightGBM](https://lightgbm.readthedocs.io/).

---

*For inquiries or issues regarding this application, please open an issue on the repository or contact the contributor.*
