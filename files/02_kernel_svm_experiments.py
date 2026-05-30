"""
02_kernel_svm_experiments.py
============================
Kernel SVM on Real-World Datasets — Hyperparameter Study

Topics:
  - Preprocessing pipeline for mixed numeric/categorical/ordinal features
  - Validation curves: effect of C, γ, and polynomial degree
  - RBF and polynomial kernels on Breast Cancer + Titanic datasets
  - Comparison of our cvxopt SVM vs sklearn SVC on small 2D sets

Datasets: sklearn Breast Cancer (numeric), Titanic via seaborn (mixed + missing)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_blobs, load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import validation_curve, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score

from cvxopt import matrix, solvers

solvers.options["show_progress"] = False


# ─────────────────────────────────────────────
# 1. Kernel Definitions (reused from file 01)
# ─────────────────────────────────────────────

def polynomial_kernel(X, Y, degree=3, coef0=1.0):
    return (X @ Y.T + coef0) ** degree


def rbf_kernel(X, Y, gamma=None):
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    X2 = np.sum(X ** 2, axis=1).reshape(-1, 1)
    Y2 = np.sum(Y ** 2, axis=1).reshape(1, -1)
    return np.exp(-gamma * (X2 + Y2 - 2.0 * X @ Y.T))


# ─────────────────────────────────────────────
# 2. cvxopt Dual Solver (portable, self-contained)
# ─────────────────────────────────────────────

def solve_svm_dual(X, y, C=1.0, kernel_fn=rbf_kernel, kernel_params=None):
    kernel_params = kernel_params or {}
    n = X.shape[0]
    y = y.astype(float)
    K = kernel_fn(X, X, **kernel_params)

    P = matrix(np.outer(y, y) * K)
    q = matrix(-np.ones(n))
    G = matrix(np.vstack((-np.eye(n), np.eye(n))))
    h = matrix(np.hstack((np.zeros(n), np.ones(n) * C)))
    A = matrix(y.reshape(1, -1), tc="d")
    b_eq = matrix(0.0)

    sol = solvers.qp(P, q, G, h, A, b_eq)
    alphas = np.ravel(sol["x"])
    sv_idx = np.where(alphas > 1e-6)[0]

    margin_mask = (alphas > 1e-6) & (alphas < C - 1e-8)
    if margin_mask.any():
        bias = float(np.mean([
            y[i] - np.sum(alphas * y * K[:, i])
            for i in np.where(margin_mask)[0]
        ]))
    else:
        sv = alphas > 1e-6
        bias = float(np.mean(
            y[sv] - np.sum((alphas * y)[:, None][sv] * K[sv][:, sv], axis=1)
        ))

    def decision_fn(X_test):
        return (alphas * y) @ kernel_fn(X, X_test, **kernel_params) + bias

    return alphas, bias, sv_idx, decision_fn


# ─────────────────────────────────────────────
# 3. Visualization Helper (2D decision boundary)
# ─────────────────────────────────────────────

def plot_2d_boundary(decision_fn, X, y, sv_idx, title=""):
    x_lo, x_hi = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_lo, y_hi = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_lo, x_hi, 300),
                         np.linspace(y_lo, y_hi, 300))
    Z = decision_fn(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.contourf(xx, yy, np.sign(Z), levels=[-1, 0, 1], alpha=0.12,
                colors=["#6baed6", "#fd8d3c"])
    ax.contour(xx, yy, Z, levels=[-1, 0, 1],
               linestyles=["--", "-", "--"], colors="k", linewidths=1)
    ax.scatter(X[y == -1, 0], X[y == -1, 1], c="#2171b5", edgecolors="k",
               s=45, label="Class −1")
    ax.scatter(X[y ==  1, 0], X[y ==  1, 1], c="#d94801", edgecolors="k",
               s=45, label="Class +1")
    if len(sv_idx):
        ax.scatter(X[sv_idx, 0], X[sv_idx, 1], s=140, facecolors="none",
                   edgecolors="green", linewidths=2, label="SVs")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# 4. Validation Curve Helper (uses sklearn SVC for speed on real datasets)
# ─────────────────────────────────────────────

def plot_validation_curve(X, y, param_name, param_range,
                          kernel="rbf", fixed_params=None,
                          scoring="accuracy", cv=5, title=None):
    """
    Sweep one hyperparameter and plot train vs CV score.
    Uses sklearn SVC (fast) for high-dimensional / large datasets.
    """
    fixed_params = fixed_params or {}
    estimator = SVC(kernel=kernel, **fixed_params)
    y_01 = np.where(y == -1, 0, y)  # sklearn expects 0/1 for some scorers

    train_sc, val_sc = validation_curve(
        estimator, X, y_01,
        param_name=param_name, param_range=param_range,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
        scoring=scoring, n_jobs=-1,
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogx(param_range, train_sc.mean(axis=1), "o-", label="Train")
    ax.semilogx(param_range, val_sc.mean(axis=1),   "s--", label="CV val")
    ax.fill_between(param_range,
                    val_sc.mean(axis=1) - val_sc.std(axis=1),
                    val_sc.mean(axis=1) + val_sc.std(axis=1), alpha=0.15)
    ax.set_xlabel(param_name)
    ax.set_ylabel(scoring)
    ax.set_title(title or f"Validation curve — {param_name}")
    ax.legend()
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()
    return train_sc.mean(axis=1), val_sc.mean(axis=1)


# ─────────────────────────────────────────────
# 5. Preprocessing Pipeline for Titanic (mixed + missing)
# ─────────────────────────────────────────────

def load_titanic():
    """
    Load Titanic from seaborn, build a preprocessing pipeline, and return
    processed feature matrix X, binary label vector y ∈ {−1, +1}.
    """
    df = sns.load_dataset("titanic")
    df = df[["survived", "pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]]
    df = df.sample(frac=1, random_state=1).reset_index(drop=True)

    X_raw = df.drop(columns="survived")
    y = np.where(df["survived"].values == 0, -1, 1)

    num_cols = ["age", "sibsp", "parch", "fare"]
    cat_cols = ["sex", "embarked"]   # nominal → one-hot
    ord_cols = ["pclass"]            # ordinal → pass through as integer

    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale",  StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe",    OneHotEncoder(handle_unknown="ignore")),
    ])
    preproc = ColumnTransformer([
        ("ord", "passthrough",  ord_cols),
        ("num", num_pipe,        num_cols),
        ("cat", cat_pipe,        cat_cols),
    ])

    X = preproc.fit_transform(X_raw)
    return X, y


# ─────────────────────────────────────────────
# 6. Experiment A — Kernel SVM on 2D Synthetic Data (cvxopt)
# ─────────────────────────────────────────────

def experiment_2d_kernels():
    print("=" * 55)
    print("Experiment A: Kernel SVM on 2D Synthetic Data")
    print("=" * 55)
    np.random.seed(0)
    X_sep, y_sep = make_blobs(n_samples=120, centers=2, cluster_std=0.5, random_state=1)
    y_sep = np.where(y_sep == 0, -1, 1)

    X_ns,  y_ns  = make_blobs(n_samples=120, centers=2, cluster_std=1.3, random_state=2)
    y_ns = np.where(y_ns == 0, -1, 1)
    flip = np.random.choice(len(y_ns), 10, replace=False)
    y_ns[flip] *= -1

    # Polynomial on separable
    print("  Polynomial kernel (degree=3) on separable data …")
    _, _, sv_idx, dec_fn = solve_svm_dual(
        X_sep, y_sep, C=None,
        kernel_fn=polynomial_kernel, kernel_params={"degree": 3, "coef0": 1.0}
    )
    plot_2d_boundary(dec_fn, X_sep, y_sep, sv_idx,
                     "Polynomial Kernel (d=3) — Separable")

    # RBF on non-separable
    print("  RBF kernel (γ=0.7, C=1.0) on non-separable data …")
    _, _, sv_idx, dec_fn = solve_svm_dual(
        X_ns, y_ns, C=1.0,
        kernel_fn=rbf_kernel, kernel_params={"gamma": 0.7}
    )
    plot_2d_boundary(dec_fn, X_ns, y_ns, sv_idx,
                     "RBF Kernel (γ=0.7, C=1.0) — Non-Separable")


# ─────────────────────────────────────────────
# 7. Experiment B — Validation Curves on Breast Cancer
# ─────────────────────────────────────────────

def experiment_breast_cancer():
    print("\n" + "=" * 55)
    print("Experiment B: Breast Cancer — Kernel Hyperparameter Study")
    print("=" * 55)
    bc = load_breast_cancer()
    X, y = bc.data, np.where(bc.target == 0, -1, 1)

    # Scale before validation curves
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    print("  RBF: sweeping γ …")
    plot_validation_curve(
        X_sc, y, "gamma", np.logspace(-4, 1, 10),
        kernel="rbf", fixed_params={"C": 1.0},
        title="Breast Cancer — RBF kernel (sweep γ)",
    )

    print("  RBF: sweeping C …")
    plot_validation_curve(
        X_sc, y, "C", np.logspace(-2, 3, 10),
        kernel="rbf", fixed_params={"gamma": 0.01},
        title="Breast Cancer — RBF kernel (sweep C)",
    )

    print("  Polynomial: sweeping degree …")
    plot_validation_curve(
        X_sc, y, "degree", [2, 3, 4, 5],
        kernel="poly", fixed_params={"C": 1.0, "coef0": 1.0},
        title="Breast Cancer — Polynomial kernel (sweep degree)",
    )


# ─────────────────────────────────────────────
# 8. Experiment C — Validation Curves on Titanic (mixed data)
# ─────────────────────────────────────────────

def experiment_titanic():
    print("\n" + "=" * 55)
    print("Experiment C: Titanic — Mixed Feature Preprocessing + Kernel Study")
    print("=" * 55)
    X, y = load_titanic()
    print(f"  Processed feature matrix shape: {X.shape}")

    print("  RBF: sweeping C …")
    plot_validation_curve(
        X, y, "C", np.logspace(-2, 3, 8),
        kernel="rbf", fixed_params={"gamma": 0.1},
        title="Titanic — RBF kernel (sweep C)",
    )

    print("  RBF: sweeping γ …")
    plot_validation_curve(
        X, y, "gamma", np.logspace(-4, 1, 8),
        kernel="rbf", fixed_params={"C": 1.0},
        title="Titanic — RBF kernel (sweep γ)",
    )

    print("  Polynomial: sweeping degree …")
    plot_validation_curve(
        X, y, "degree", [2, 3, 4],
        kernel="poly", fixed_params={"C": 1.0, "coef0": 1.0},
        title="Titanic — Polynomial kernel (sweep degree)",
    )


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    experiment_2d_kernels()
    experiment_breast_cancer()
    experiment_titanic()
