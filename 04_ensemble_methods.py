"""
04_ensemble_methods.py
======================
Ensemble Classifiers — A Practical Comparison

Topics:
  - Bagging vs Random Forest: n_estimators, max_depth, OOB score
  - AdaBoost, GradientBoosting, XGBoost: learning rate, depth, subsample
  - Stacking: SVM + RF + XGBoost → Logistic Regression meta-learner
  - Fair nested cross-validation comparison (mean ± std across folds)

Dataset: sklearn Breast Cancer (569 samples, 30 features, binary)
"""

import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    train_test_split, StratifiedKFold,
    GridSearchCV, cross_validate, validation_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    BaggingClassifier, RandomForestClassifier,
    AdaBoostClassifier, GradientBoostingClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False
    print("⚠  XGBoost not installed — XGBoost experiments will be skipped.")
    print("   Install with: pip install xgboost")


# ─────────────────────────────────────────────
# 0. Data Setup
# ─────────────────────────────────────────────

def load_data():
    bc = load_breast_cancer()
    X, y = bc.data, bc.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    return X, y, X_train, X_test, y_train, y_test, bc.target_names


X_all, y_all, X_tr, X_te, y_tr, y_te, class_names = load_data()


# ─────────────────────────────────────────────
# 1. Bagging vs Random Forest
# ─────────────────────────────────────────────

def experiment_bagging_vs_rf():
    print("=" * 60)
    print("Experiment 1: Bagging vs Random Forest")
    print("=" * 60)

    n_list = [10, 50, 100, 200]
    bag_rows, rf_rows = [], []

    for n in n_list:
        # Bagging
        t0 = time.time()
        bag = BaggingClassifier(
            estimator=DecisionTreeClassifier(random_state=42),
            n_estimators=n, oob_score=True, n_jobs=-1, random_state=42,
        )
        bag.fit(X_tr, y_tr)
        bag_acc = accuracy_score(y_te, bag.predict(X_te))
        bag_rows.append(dict(n=n, acc=bag_acc, oob=bag.oob_score_,
                             time=time.time() - t0))

        # Random Forest
        t0 = time.time()
        rf = RandomForestClassifier(
            n_estimators=n, oob_score=True, n_jobs=-1, random_state=42,
        )
        rf.fit(X_tr, y_tr)
        rf_acc = accuracy_score(y_te, rf.predict(X_te))
        rf_rows.append(dict(n=n, acc=rf_acc, oob=rf.oob_score_,
                            time=time.time() - t0))

    bag_df = pd.DataFrame(bag_rows)
    rf_df  = pd.DataFrame(rf_rows)

    print("\n  Bagging results:")
    print(bag_df.to_string(index=False))
    print("\n  Random Forest results:")
    print(rf_df.to_string(index=False))

    # ── Plot ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(bag_df.n, bag_df.acc,  "o-",  label="Bagging  test acc")
    axes[0].plot(bag_df.n, bag_df.oob,  "o--", label="Bagging  OOB")
    axes[0].plot(rf_df.n,  rf_df.acc,   "s-",  label="RF  test acc")
    axes[0].plot(rf_df.n,  rf_df.oob,   "s--", label="RF  OOB")
    axes[0].set_xlabel("n_estimators")
    axes[0].set_ylabel("Score")
    axes[0].set_title("Accuracy / OOB vs. n_estimators")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(bag_df.n, bag_df.time, "o-", label="Bagging")
    axes[1].plot(rf_df.n,  rf_df.time,  "s-", label="RandomForest")
    axes[1].set_xlabel("n_estimators")
    axes[1].set_ylabel("Training time (s)")
    axes[1].set_title("Training Time vs. n_estimators")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.suptitle("Bagging vs Random Forest — Breast Cancer")
    plt.tight_layout()
    plt.show()

    # ── RF: effect of max_depth and max_features ──
    depths  = [None, 3, 5, 10]
    feats   = ["sqrt", 0.3, 0.5, 1.0]
    rows = []
    for d in depths:
        for f in feats:
            rf_tmp = RandomForestClassifier(
                n_estimators=100, max_depth=d, max_features=f,
                n_jobs=-1, random_state=42,
            )
            rf_tmp.fit(X_tr, y_tr)
            rows.append(dict(depth=str(d), features=str(f),
                             acc=accuracy_score(y_te, rf_tmp.predict(X_te))))
    rf_hyp = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for d in depths:
        sub = rf_hyp[rf_hyp.depth == str(d)]
        axes[0].plot(range(len(feats)), sub.acc.values, "o-", label=f"depth={d}")
    axes[0].set_xticks(range(len(feats)))
    axes[0].set_xticklabels([str(f) for f in feats])
    axes[0].set_xlabel("max_features")
    axes[0].set_ylabel("Test Accuracy")
    axes[0].set_title("RF: Depth × Max Features")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    # Feature importance (best RF)
    best_rf = RandomForestClassifier(n_estimators=200, max_depth=None,
                                     n_jobs=-1, random_state=42)
    best_rf.fit(X_tr, y_tr)
    importances = best_rf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:15]
    axes[1].barh(range(15), importances[top_idx][::-1], color="steelblue")
    axes[1].set_yticks(range(15))
    axes[1].set_yticklabels([f"feat {i}" for i in top_idx[::-1]], fontsize=7)
    axes[1].set_xlabel("Importance")
    axes[1].set_title("RF — Top-15 Feature Importances")

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# 2. AdaBoost + GradientBoosting + XGBoost
# ─────────────────────────────────────────────

def experiment_boosting():
    print("\n" + "=" * 60)
    print("Experiment 2: AdaBoost / GradientBoosting / XGBoost")
    print("=" * 60)

    n_list    = [50, 100, 200]
    lr_list   = [0.01, 0.1, 0.3, 1.0]
    depth_list = [1, 3, 5]
    rows = []

    # AdaBoost
    for lr in lr_list:
        for n in n_list:
            ada = AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=1),
                n_estimators=n, learning_rate=lr, random_state=42,
            )
            ada.fit(X_tr, y_tr)
            rows.append(dict(
                model="AdaBoost", lr=lr, n=n, depth=1, subsample=None,
                acc=accuracy_score(y_te, ada.predict(X_te)),
            ))
            print(f"  AdaBoost  lr={lr:.2f} n={n:3d}  acc={rows[-1]['acc']:.3f}")

    # Gradient Boosting
    for lr in lr_list:
        for n in n_list:
            for d in depth_list:
                gb = GradientBoostingClassifier(
                    learning_rate=lr, n_estimators=n, max_depth=d, random_state=42,
                )
                gb.fit(X_tr, y_tr)
                rows.append(dict(
                    model="GradBoost", lr=lr, n=n, depth=d, subsample=None,
                    acc=accuracy_score(y_te, gb.predict(X_te)),
                ))

    # XGBoost
    if _HAS_XGB:
        for lr in lr_list:
            for n in n_list:
                for d in depth_list:
                    xg = xgb.XGBClassifier(
                        learning_rate=lr, n_estimators=n, max_depth=d,
                        subsample=0.8, colsample_bytree=0.8,
                        eval_metric="logloss", random_state=42,
                        n_jobs=-1, verbosity=0,
                    )
                    xg.fit(X_tr, y_tr)
                    rows.append(dict(
                        model="XGBoost", lr=lr, n=n, depth=d, subsample=0.8,
                        acc=accuracy_score(y_te, xg.predict(X_te)),
                    ))

    df = pd.DataFrame(rows)

    # ── Summary ──
    print("\n  Best accuracy per model:")
    print(df.groupby("model")["acc"].max().to_string())

    # ── Plots ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    models = df.model.unique()
    colors = {"AdaBoost": "#1f77b4", "GradBoost": "#ff7f0e", "XGBoost": "#2ca02c"}

    # lr effect
    for m in models:
        sub = df[df.model == m].groupby("lr")["acc"].mean()
        axes[0].semilogx(sub.index, sub.values, "o-", label=m, color=colors[m])
    axes[0].set_xlabel("Learning rate (log scale)")
    axes[0].set_ylabel("Mean accuracy")
    axes[0].set_title("Effect of Learning Rate")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    # n_estimators effect
    for m in models:
        sub = df[df.model == m].groupby("n")["acc"].mean()
        axes[1].plot(sub.index, sub.values, "o-", label=m, color=colors[m])
    axes[1].set_xlabel("n_estimators")
    axes[1].set_title("Effect of n_estimators")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    # depth effect (gradient / xgb only)
    for m in ["GradBoost", "XGBoost"]:
        sub = df[df.model == m].groupby("depth")["acc"].mean()
        axes[2].plot(sub.index, sub.values, "o-", label=m, color=colors[m])
    axes[2].set_xlabel("max_depth")
    axes[2].set_title("Effect of Tree Depth")
    axes[2].legend(); axes[2].grid(alpha=0.3)

    plt.suptitle("Boosting: Hyperparameter Study — Breast Cancer")
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# 3. Stacking: SVM + RF + XGBoost → LogReg
# ─────────────────────────────────────────────

def experiment_stacking():
    print("\n" + "=" * 60)
    print("Experiment 3: Stacking Ensemble")
    print("=" * 60)

    base_learners = [
        ("svm", SVC(kernel="linear", probability=True, C=1.0, random_state=42)),
        ("rf",  RandomForestClassifier(n_estimators=200, max_depth=5, n_jobs=-1, random_state=42)),
    ]
    if _HAS_XGB:
        base_learners.append((
            "xgb",
            xgb.XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric="logloss", random_state=42, n_jobs=-1, verbosity=0,
            )
        ))

    meta = LogisticRegression(max_iter=2000, random_state=42)
    stack = StackingClassifier(
        estimators=base_learners,
        final_estimator=meta,
        cv=5, n_jobs=-1,
    )

    models = {name: est for name, est in base_learners}
    models["Stacking"] = stack

    rows = []
    for name, model in models.items():
        t0 = time.time()
        model.fit(X_tr, y_tr)
        acc = accuracy_score(y_te, model.predict(X_te))
        rows.append(dict(model=name, accuracy=acc, time=time.time() - t0))
        print(f"  {name:20s} | acc={acc:.3f}")

    print("\n  Classification Report — Stacking:")
    print(classification_report(y_te, stack.predict(X_te),
                                 target_names=class_names))

    df = pd.DataFrame(rows).sort_values("accuracy", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#d62728" if r["model"] == "Stacking" else "#aec7e8"
              for _, r in df.iterrows()]
    ax.barh(df.model, df.accuracy, color=colors)
    ax.set_xlim(0.9, 1.0)
    ax.set_xlabel("Test Accuracy")
    ax.set_title("Stacking vs Individual Base Learners")
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# 4. Fair Nested-CV Comparison (mean ± std)
# ─────────────────────────────────────────────

def experiment_fair_comparison():
    print("\n" + "=" * 60)
    print("Experiment 4: Fair Nested Cross-Validation Comparison")
    print("=" * 60)
    print("  (Inner 3-fold GridSearchCV, outer 5-fold StratifiedKFold)")

    def make_pipe(clf):
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
    outer_cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scoring = {"accuracy": "accuracy", "f1": "f1_macro", "roc_auc": "roc_auc"}

    # Define (pipeline, param_grid) pairs
    candidates = {
        "SVM (tuned)": (
            make_pipe(SVC(probability=True, random_state=42)),
            {"clf__C": [0.1, 1, 10],
             "clf__kernel": ["rbf"],
             "clf__gamma": [0.01, 0.1]},
        ),
        "RandomForest (tuned)": (
            make_pipe(RandomForestClassifier(random_state=42, n_jobs=-1)),
            {"clf__n_estimators": [100, 200],
             "clf__max_depth": [None, 5],
             "clf__max_features": ["sqrt", 0.5]},
        ),
    }
    if _HAS_XGB:
        candidates["XGBoost (tuned)"] = (
            make_pipe(xgb.XGBClassifier(
                eval_metric="logloss", random_state=42, n_jobs=-1, verbosity=0,
            )),
            {"clf__n_estimators": [100, 200],
             "clf__max_depth": [3, 6],
             "clf__learning_rate": [0.1, 0.3]},
        )

    # Build stacking inside the outer fold too
    stack_base = [
        ("svm", SVC(kernel="rbf", probability=True, C=1.0, gamma=0.01, random_state=42)),
        ("rf",  RandomForestClassifier(n_estimators=200, max_depth=5, n_jobs=-1, random_state=42)),
    ]
    if _HAS_XGB:
        stack_base.append((
            "xgb",
            xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                               eval_metric="logloss", random_state=42, n_jobs=-1, verbosity=0)
        ))
    stacking_pipe = make_pipe(
        StackingClassifier(estimators=stack_base,
                           final_estimator=LogisticRegression(max_iter=2000),
                           cv=5, n_jobs=-1)
    )
    candidates["Stacking"] = (
        stacking_pipe,
        {"clf__final_estimator__C": [0.1, 1.0, 10.0]},
    )

    summary = []
    for name, (pipe, grid) in candidates.items():
        print(f"\n  → Evaluating: {name}")
        gs = GridSearchCV(pipe, grid, cv=inner_cv, scoring="accuracy",
                          n_jobs=-1, refit=True)
        cv_res = cross_validate(gs, X_all, y_all, cv=outer_cv,
                                scoring=scoring, n_jobs=1)
        summary.append({
            "Model": name,
            "Accuracy": f"{cv_res['test_accuracy'].mean():.3f} ± {cv_res['test_accuracy'].std():.3f}",
            "F1 (macro)": f"{cv_res['test_f1'].mean():.3f} ± {cv_res['test_f1'].std():.3f}",
            "ROC-AUC":    f"{cv_res['test_roc_auc'].mean():.3f} ± {cv_res['test_roc_auc'].std():.3f}",
            "Fit time(s)": f"{cv_res['fit_time'].mean():.1f}",
        })

    summary_df = pd.DataFrame(summary).set_index("Model")
    print("\n\n  ══ Nested-CV Fair Comparison (5 outer folds) ══")
    print(summary_df.to_string())

    # Bar chart of mean accuracies
    means = [float(r["Accuracy"].split(" ±")[0]) for r in summary]
    stds  = [float(r["Accuracy"].split("± ")[1]) for r in summary]
    labels = summary_df.index.tolist()

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color="#5b9bd5", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("CV Accuracy")
    ax.set_ylim(0.88, 1.00)
    ax.set_title("Fair Nested-CV Model Comparison — Breast Cancer")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

    return summary_df


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    experiment_bagging_vs_rf()
    experiment_boosting()
    experiment_stacking()
    experiment_fair_comparison()
