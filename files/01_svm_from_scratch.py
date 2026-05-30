"""
01_svm_from_scratch.py
======================
Support Vector Machines — Built from the Ground Up

Topics:
  - Kernel functions (linear, polynomial, RBF)
  - Hard-margin SVM (linearly separable data)
  - Soft-margin SVM with slack variables (non-separable data)
  - SMO-style dual QP solver (via cvxopt)
  - Decision boundary + margin + support vector visualization

Datasets: Synthetic 2D (separable and non-separable)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.svm import SVC  # used only for sanity-check comparison
from cvxopt import matrix, solvers

solvers.options["show_progress"] = False

# ─────────────────────────────────────────────
# 1. Kernel Functions
# ─────────────────────────────────────────────

def linear_kernel(X, Y):
    """K(x, x') = x · x'"""
    return X @ Y.T


def polynomial_kernel(X, Y, degree=3, coef0=1.0):
    """K(x, x') = (x · x' + r)^d"""
    return (X @ Y.T + coef0) ** degree


def rbf_kernel(X, Y, gamma=None):
    """K(x, x') = exp(-γ ||x - x'||²)"""
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    X2 = np.sum(X ** 2, axis=1).reshape(-1, 1)
    Y2 = np.sum(Y ** 2, axis=1).reshape(1, -1)
    return np.exp(-gamma * (X2 + Y2 - 2.0 * X @ Y.T))


# ─────────────────────────────────────────────
# 2. Core SVM: Dual QP Solver (SMO-style via cvxopt)
# ─────────────────────────────────────────────

def solve_svm_dual(X, y, C=None, kernel_fn=linear_kernel, kernel_params=None):
    """
    Solve the kernelized SVM dual:
      maximize  Σ αᵢ  −  ½ ΣΣ αᵢ αⱼ yᵢ yⱼ K(xᵢ, xⱼ)
      s.t.      Σ αᵢ yᵢ = 0,   0 ≤ αᵢ ≤ C  (or αᵢ ≥ 0 for hard-margin)

    Returns: alphas, bias b, support-vector indices, decision_function callable
    """
    kernel_params = kernel_params or {}
    n = X.shape[0]
    y = y.astype(float)
    K = kernel_fn(X, X, **kernel_params)

    P = matrix(np.outer(y, y) * K)
    q = matrix(-np.ones(n))
    A = matrix(y.reshape(1, -1), tc="d")
    b_eq = matrix(0.0)

    if C is None:  # hard-margin: αᵢ ≥ 0
        G = matrix(-np.eye(n))
        h = matrix(np.zeros(n))
    else:          # soft-margin: 0 ≤ αᵢ ≤ C
        G = matrix(np.vstack((-np.eye(n), np.eye(n))))
        h = matrix(np.hstack((np.zeros(n), np.ones(n) * C)))

    sol = solvers.qp(P, q, G, h, A, b_eq)
    alphas = np.ravel(sol["x"])

    sv_mask = alphas > 1e-6
    sv_idx = np.where(sv_mask)[0]

    # Bias: average over margin support vectors (0 < α < C)
    if C is not None:
        margin_mask = (alphas > 1e-6) & (alphas < C - 1e-8)
    else:
        margin_mask = sv_mask

    if margin_mask.any():
        b_vals = [
            y[i] - np.sum(alphas * y * K[:, i])
            for i in np.where(margin_mask)[0]
        ]
        bias = float(np.mean(b_vals))
    else:
        bias = float(np.mean(
            y[sv_idx] - np.sum((alphas * y)[:, None][sv_mask] * K[sv_mask][:, sv_mask], axis=1)
        ))

    def decision_function(X_test):
        K_t = kernel_fn(X, X_test, **kernel_params)
        return (alphas * y) @ K_t + bias

    return alphas, bias, sv_idx, decision_function


# ─────────────────────────────────────────────
# 3. Visualization Utility
# ─────────────────────────────────────────────

def plot_svm_2d(decision_fn, X, y, sv_idx, title="SVM Decision Boundary"):
    """Plot data, decision boundary, margin lines, and support vectors."""
    x_lo, x_hi = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_lo, y_hi = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_lo, x_hi, 300),
                         np.linspace(y_lo, y_hi, 300))
    Z = decision_fn(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contourf(xx, yy, np.sign(Z), levels=[-1, 0, 1], alpha=0.12,
                colors=["#6baed6", "#fd8d3c"])
    ax.contour(xx, yy, Z, levels=[-1, 0, 1],
               linestyles=["--", "-", "--"], colors="k", linewidths=1.2)
    ax.scatter(X[y == -1, 0], X[y == -1, 1], c="#2171b5",
               edgecolors="k", s=50, label="Class −1")
    ax.scatter(X[y ==  1, 0], X[y ==  1, 1], c="#d94801",
               edgecolors="k", s=50, label="Class +1")
    if len(sv_idx):
        ax.scatter(X[sv_idx, 0], X[sv_idx, 1], s=160,
                   facecolors="none", edgecolors="green",
                   linewidths=2, label="Support Vectors")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# 4. Dataset Generators
# ─────────────────────────────────────────────

def make_separable(n=120, seed=42):
    X, y = make_blobs(n_samples=n, centers=2, cluster_std=0.55, random_state=seed)
    return X, np.where(y == 0, -1, 1)


def make_nonseparable(n=160, seed=24):
    X, y = make_blobs(n_samples=n, centers=2, cluster_std=1.3, random_state=seed)
    y = np.where(y == 0, -1, 1)
    rng = np.random.default_rng(seed)
    flip_idx = rng.choice(len(y), size=int(0.15 * n), replace=False)
    y[flip_idx] *= -1   # inject label noise
    return X, y


# ─────────────────────────────────────────────
# 5. Experiment A — Hard-Margin SVM on Separable Data
# ─────────────────────────────────────────────

def experiment_hard_margin():
    print("=" * 55)
    print("Experiment A: Hard-Margin Linear SVM")
    print("=" * 55)
    X, y = make_separable()
    alphas, bias, sv_idx, dec_fn = solve_svm_dual(X, y, C=None, kernel_fn=linear_kernel)

    print(f"  Support vectors found : {len(sv_idx)}")
    print(f"  Bias (b)              : {bias:.4f}")
    print(f"  Training accuracy     : {np.mean(np.sign(dec_fn(X)) == y):.3f}")

    # Sanity-check against sklearn's hard-margin SVC (very large C ≈ hard margin)
    ref = SVC(kernel="linear", C=1e10)
    ref.fit(X, y)
    print(f"  sklearn reference acc : {ref.score(X, y):.3f}")

    plot_svm_2d(dec_fn, X, y, sv_idx, title="Hard-Margin Linear SVM — Separable Data")


# ─────────────────────────────────────────────
# 6. Experiment B — Soft-Margin SVM: Effect of C
# ─────────────────────────────────────────────

def experiment_soft_margin():
    print("\n" + "=" * 55)
    print("Experiment B: Soft-Margin SVM — Varying C")
    print("=" * 55)
    X, y = make_nonseparable()

    for C in [0.1, 1.0, 10.0]:
        alphas, bias, sv_idx, dec_fn = solve_svm_dual(X, y, C=C, kernel_fn=linear_kernel)
        acc = np.mean(np.sign(dec_fn(X)) == y)
        print(f"  C={C:5.1f} | SVs={len(sv_idx):3d} | Train Acc={acc:.3f}")
        plot_svm_2d(dec_fn, X, y, sv_idx,
                    title=f"Soft-Margin Linear SVM — Non-Separable Data (C={C})")


# ─────────────────────────────────────────────
# 7. Experiment C — RBF Kernel SVM
# ─────────────────────────────────────────────

def experiment_rbf_kernel():
    print("\n" + "=" * 55)
    print("Experiment C: RBF Kernel SVM on Non-Separable Data")
    print("=" * 55)
    X, y = make_nonseparable()

    for gamma in [0.3, 0.7, 2.0]:
        alphas, bias, sv_idx, dec_fn = solve_svm_dual(
            X, y, C=1.0, kernel_fn=rbf_kernel, kernel_params={"gamma": gamma}
        )
        acc = np.mean(np.sign(dec_fn(X)) == y)
        print(f"  γ={gamma} | SVs={len(sv_idx):3d} | Train Acc={acc:.3f}")
        plot_svm_2d(dec_fn, X, y, sv_idx,
                    title=f"RBF Kernel SVM (γ={gamma}, C=1.0)")


# ─────────────────────────────────────────────
# 8. Experiment D — Polynomial Kernel SVM
# ─────────────────────────────────────────────

def experiment_poly_kernel():
    print("\n" + "=" * 55)
    print("Experiment D: Polynomial Kernel SVM on Separable Data")
    print("=" * 55)
    X, y = make_separable()

    for degree in [2, 3, 5]:
        alphas, bias, sv_idx, dec_fn = solve_svm_dual(
            X, y, C=1.0, kernel_fn=polynomial_kernel,
            kernel_params={"degree": degree, "coef0": 1.0}
        )
        acc = np.mean(np.sign(dec_fn(X)) == y)
        print(f"  degree={degree} | SVs={len(sv_idx):3d} | Train Acc={acc:.3f}")
        plot_svm_2d(dec_fn, X, y, sv_idx,
                    title=f"Polynomial Kernel SVM (degree={degree}, C=1.0)")


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    experiment_hard_margin()
    experiment_soft_margin()
    experiment_rbf_kernel()
    experiment_poly_kernel()
