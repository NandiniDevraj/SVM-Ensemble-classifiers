# SVM & Ensemble Classifiers — Hands-On Playground

A self-contained ML practice project covering **Support Vector Machines from scratch** and **ensemble classifier experiments** using scikit-learn / XGBoost.

---

## Project Structure

```
svm-ensemble-playground/
├── 01_svm_from_scratch.py        # SVM core: kernels, hard/soft margin, SMO solver
├── 02_kernel_svm_experiments.py  # Kernel SVM (Poly & RBF), validation curves, real datasets
├── 03_multiclass_imbalance.py    # One-vs-Rest multi-class SVM + class imbalance handling
├── 04_ensemble_methods.py        # Bagging, RF, AdaBoost, XGBoost, Stacking, fair comparison
└── README.md
```

---

## Topics Covered

### File 1 — SVM from Scratch (`01_svm_from_scratch.py`)
- Linear/Polynomial/RBF kernel functions
- SMO-style dual solver (via cvxopt QP)
- Hard-margin and soft-margin SVMs
- Decision boundary visualization with margin lines and support vectors

### File 2 — Kernel SVM Experiments (`02_kernel_svm_experiments.py`)
- Kernelized dual SVM on 2D synthetic datasets
- External datasets: Breast Cancer + Titanic (with preprocessing pipeline)
- Validation curves: effect of C, γ, degree on accuracy
- Column transformer for mixed numeric/categorical data

### File 3 — Multi-class & Class Imbalance (`03_multiclass_imbalance.py`)
- One-vs-Rest wrapper for K-class SVM
- Evaluation on digits dataset (per-class F1, macro-F1, confusion matrix)
- Skewed class distributions (1:5 ratio)
- Plain SVM vs. class-weighted SVM
- Precision–Recall curves and AUC comparison

### File 4 — Ensemble Methods (`04_ensemble_methods.py`)
- BaggingClassifier vs RandomForestClassifier (n_estimators, depth, OOB score)
- AdaBoost, GradientBoosting, XGBoost — effect of learning rate, depth, subsample
- StackingClassifier (SVM + RF + XGBoost → LogisticRegression meta-learner)
- Fair nested-CV comparison: mean ± std across folds

---

## Requirements

```bash
pip install numpy matplotlib scikit-learn seaborn cvxopt xgboost
```

## Running

Each file is fully self-contained and can be run independently:

```bash
python 01_svm_from_scratch.py
python 02_kernel_svm_experiments.py
python 03_multiclass_imbalance.py
python 04_ensemble_methods.py
```
