"""
03_multiclass_imbalance.py
==========================
One-vs-Rest Multi-class SVM  +  Class Imbalance Handling

Topics:
  - Wrapping a binary SVM into a One-vs-Rest (OvR) classifier
  - Per-class F1, macro-F1, confusion matrix on a multi-class dataset
  - Simulating class imbalance (1:5 ratio)
  - Plain SVM vs. class-weighted SVM
  - Precision–Recall curves and AUC comparison

Datasets:
  - sklearn Digits (multi-class, 10 classes)
  - Synthetic binary (imbalanced, generated via make_classification)
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve, auc,
)
from cvxopt import matrix, solvers

solvers.options["show_progress"] = False


# ─────────────────────────────────────────────
# 1. Kernel Functions
# ─────────────────────────────────────────────

def rbf_kernel(X, Y, gamma=0.5):
    if X.ndim == 1: X = X[np.newaxis, :]
    if Y.ndim == 1: Y = Y[np.newaxis, :]
    diff = X[:, np.newaxis, :] - Y[np.newaxis, :, :]
    return np.exp(-gamma * np.sum(diff ** 2, axis=2))


def linear_kernel(X, Y):
    return X @ Y.T


# ─────────────────────────────────────────────
# 2. Binary SVM (cvxopt dual solver)
# ─────────────────────────────────────────────

class BinarySVM:
    """
    Kernel SVM for binary classification (labels ∈ {−1, +1}).
    Supports optional per-class C weighting for imbalance handling.
    """

    def __init__(self, C=1.0, kernel="rbf", gamma=0.5, class_weights=None):
        self.C = C
        self.gamma = gamma
        self.class_weights = class_weights or {1: 1.0, -1: 1.0}
        if kernel == "rbf":
            self._kfn = lambda X, Y: rbf_kernel(X, Y, self.gamma)
        elif kernel == "linear":
            self._kfn = linear_kernel
        else:
            raise ValueError(f"Unknown kernel: {kernel}")

    def fit(self, X, y):
        y = y.astype(float)
        n = X.shape[0]
        K = self._kfn(X, X).astype(float)

        # Per-sample C values (encode class weights)
        C_vec = np.array([self.class_weights.get(yi, 1.0) * self.C for yi in y])

        P = matrix(np.outer(y, y) * K)
        q = matrix(-np.ones(n))
        G = matrix(np.vstack((-np.eye(n), np.eye(n))))
        h = matrix(np.hstack((np.zeros(n), C_vec)))
        A = matrix(y.reshape(1, -1), tc="d")
        b_eq = matrix(0.0)

        sol = solvers.qp(P, q, G, h, A, b_eq)
        alphas = np.ravel(sol["x"])

        sv = alphas > 1e-5
        self._alphas = alphas[sv]
        self._sv_X   = X[sv]
        self._sv_y   = y[sv]
        self._K_sv   = K[np.ix_(sv, sv)]
        self._b = float(np.mean(
            self._sv_y - np.sum(self._alphas * self._sv_y * self._K_sv, axis=1)
        ))
        return self

    def decision_function(self, X):
        K = self._kfn(X, self._sv_X)
        return K @ (self._alphas * self._sv_y) + self._b

    def predict(self, X):
        return np.sign(self.decision_function(X))


# ─────────────────────────────────────────────
# 3. One-vs-Rest Multi-class Wrapper
# ─────────────────────────────────────────────

class OneVsRestSVM:
    """
    Wraps BinarySVM into a K-class One-vs-Rest classifier.
    Prediction selects the class with the highest confidence score.
    """

    def __init__(self, C=1.0, kernel="rbf", gamma=0.5):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self._models = {}

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        for cls in self.classes_:
            print(f"  Training class {cls} vs rest …", end="\r")
            y_bin = np.where(y == cls, 1, -1)
            svm = BinarySVM(C=self.C, kernel=self.kernel, gamma=self.gamma)
            svm.fit(X, y_bin)
            self._models[cls] = svm
        print(" " * 50, end="\r")
        return self

    def predict(self, X):
        scores = np.stack(
            [self._models[c].decision_function(X) for c in self.classes_], axis=1
        )
        return self.classes_[np.argmax(scores, axis=1)]


# ─────────────────────────────────────────────
# 4. Experiment A — Multi-class Digits Dataset
# ─────────────────────────────────────────────

def experiment_multiclass():
    print("=" * 58)
    print("Experiment A: One-vs-Rest SVM on Digits Dataset (10 classes)")
    print("=" * 58)

    digits = load_digits()
    X, y = digits.data, digits.target

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"\n  Training set : {X_train.shape[0]} samples")
    print(f"  Test set     : {X_test.shape[0]} samples\n")

    ovr = OneVsRestSVM(C=1.0, kernel="rbf", gamma=0.05)
    ovr.fit(X_train, y_train)

    y_pred = ovr.predict(X_test)

    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=digits.target_names)
    fig, ax = plt.subplots(figsize=(9, 7))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("One-vs-Rest SVM — Confusion Matrix (Digits)")
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# 5. Experiment B — Class Imbalance: Plain vs Weighted SVM
# ─────────────────────────────────────────────

def experiment_class_imbalance():
    print("\n" + "=" * 58)
    print("Experiment B: Class Imbalance — Plain SVM vs Weighted SVM")
    print("=" * 58)

    # Generate imbalanced binary dataset (approx 1:5 minority:majority)
    X, y = make_classification(
        n_samples=900, n_features=2, n_informative=2,
        n_redundant=0, n_clusters_per_class=1,
        weights=[0.83, 0.17], random_state=42,
    )
    y = np.where(y == 1, 1, -1)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    n_neg = np.sum(y_train == -1)
    n_pos = np.sum(y_train ==  1)
    ratio = n_neg / n_pos
    print(f"\n  Minority (+1) : {n_pos}  |  Majority (−1) : {n_neg}  |  Ratio ≈ {ratio:.1f}:1\n")

    # --- Plain SVM (no weighting) ---
    svm_plain = BinarySVM(C=1.0, kernel="rbf", gamma=0.5)
    svm_plain.fit(X_train, y_train)
    y_pred_plain  = svm_plain.predict(X_test)
    scores_plain  = svm_plain.decision_function(X_test)

    # --- Class-Weighted SVM (penalise minority misclassification more) ---
    svm_wt = BinarySVM(
        C=1.0, kernel="rbf", gamma=0.5,
        class_weights={1: ratio, -1: 1.0}
    )
    svm_wt.fit(X_train, y_train)
    y_pred_wt    = svm_wt.predict(X_test)
    scores_wt    = svm_wt.decision_function(X_test)

    # ---- Reports ----
    print("  === Plain SVM ===")
    print(classification_report(y_test, y_pred_plain, target_names=["Majority", "Minority"]))

    print("  === Class-Weighted SVM ===")
    print(classification_report(y_test, y_pred_wt, target_names=["Majority", "Minority"]))

    # ---- Precision–Recall Curves ----
    y_bin = (y_test == 1).astype(int)
    prec_p, rec_p, _ = precision_recall_curve(y_bin, scores_plain)
    prec_w, rec_w, _ = precision_recall_curve(y_bin, scores_wt)
    auc_p = auc(rec_p, prec_p)
    auc_w = auc(rec_w, prec_w)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(rec_p, prec_p, label=f"Plain SVM  (PR-AUC = {auc_p:.3f})")
    ax.plot(rec_w, prec_w, "--", label=f"Weighted SVM (PR-AUC = {auc_w:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curve: Imbalanced Dataset")
    ax.legend()
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()

    # ---- Visual decision boundary comparison ----
    _plot_imbalance_boundary(svm_plain, svm_wt, X_test, y_test)


def _plot_imbalance_boundary(svm_plain, svm_wt, X, y):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, model, title in zip(
        axes,
        [svm_plain, svm_wt],
        ["Plain SVM", "Class-Weighted SVM"],
    ):
        x_lo, x_hi = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_lo, y_hi = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_lo, x_hi, 250),
                             np.linspace(y_lo, y_hi, 250))
        Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        ax.contourf(xx, yy, np.sign(Z), levels=[-1, 0, 1], alpha=0.15,
                    colors=["#6baed6", "#fd8d3c"])
        ax.contour(xx, yy, Z, levels=[0], colors="k", linewidths=1.5)
        ax.scatter(X[y == -1, 0], X[y == -1, 1], c="#2171b5",
                   edgecolors="k", s=20, alpha=0.5, label="Majority")
        ax.scatter(X[y ==  1, 0], X[y ==  1, 1], c="#d94801",
                   edgecolors="k", s=60, label="Minority")
        ax.set_title(title)
        ax.legend(fontsize=8)
    plt.suptitle("Class Imbalance: Decision Boundary Comparison")
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    experiment_multiclass()
    experiment_class_imbalance()
