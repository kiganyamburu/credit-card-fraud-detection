import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train fraud-detection baselines and evaluate with AUPRC."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="creditcard.csv",
        help="Path to dataset CSV file.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set fraction used in train/test split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold used for confusion matrix, precision, and recall.",
    )
    parser.add_argument(
        "--target-recall",
        type=float,
        default=None,
        help=(
            "Tune threshold to satisfy recall >= target and maximize precision. "
            "Mutually exclusive with --target-precision."
        ),
    )
    parser.add_argument(
        "--target-precision",
        type=float,
        default=None,
        help=(
            "Tune threshold to satisfy precision >= target and maximize recall. "
            "Mutually exclusive with --target-recall."
        ),
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of stratified CV folds for AUPRC estimation.",
    )
    parser.add_argument(
        "--ci-level",
        type=float,
        default=0.95,
        help="Confidence interval level for CV mean AUPRC (e.g., 0.95).",
    )
    parser.add_argument(
        "--cv-bootstrap-iters",
        type=int,
        default=2000,
        help="Bootstrap iterations for CV mean AUPRC confidence intervals.",
    )
    return parser.parse_args()


def load_data(csv_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required = {"Class"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    y = df["Class"].astype(int)
    X = df.drop(columns=["Class"])
    return X, y


def get_models(seed: int) -> dict[str, object]:
    # Logistic Regression is a strong linear baseline for standardized tabular data.
    logistic = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=seed,
                ),
            ),
        ]
    )

    # Histogram GBM provides a nonlinear baseline and supports class weights.
    hgb = HistGradientBoostingClassifier(
        max_depth=6,
        max_iter=200,
        learning_rate=0.05,
        random_state=seed,
        class_weight="balanced",
    )

    return {
        "logistic_balanced": logistic,
        "hist_gradient_boosting": hgb,
    }


def evaluate_threshold_metrics(
    y_true: pd.Series, y_score, threshold: float
) -> tuple[float, float, tuple[int, int, int, int]]:
    y_pred = (y_score >= threshold).astype(int)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return precision, recall, (tn, fp, fn, tp)


def tune_threshold(
    y_true: pd.Series,
    y_score,
    target_recall: float | None,
    target_precision: float | None,
    fallback_threshold: float,
) -> tuple[float, str]:
    if target_recall is None and target_precision is None:
        return fallback_threshold, "fixed"

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)

    if thresholds.size == 0:
        return fallback_threshold, "fixed_no_candidates"

    # The last PR point has no matching threshold and is excluded.
    precision = precision[:-1]
    recall = recall[:-1]

    if target_recall is not None:
        valid_mask = recall >= target_recall
        mode = "target_recall"
    else:
        valid_mask = precision >= target_precision
        mode = "target_precision"

    valid_indices = np.flatnonzero(valid_mask)
    if valid_indices.size == 0:
        return fallback_threshold, f"{mode}_not_met_using_fallback"

    if target_recall is not None:
        # Meet recall constraint while maximizing precision.
        best_metric = precision[valid_indices]
        secondary = recall[valid_indices]
    else:
        # Meet precision constraint while maximizing recall.
        best_metric = recall[valid_indices]
        secondary = precision[valid_indices]

    ranked = np.lexsort(
        (
            -thresholds[valid_indices],
            -secondary,
            -best_metric,
        )
    )
    chosen_idx = valid_indices[ranked[0]]
    return float(thresholds[chosen_idx]), mode


def bootstrap_mean_ci(
    values: np.ndarray,
    ci_level: float,
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = values.size
    if n == 1:
        return float(values[0]), float(values[0])

    samples = rng.choice(values, size=(iterations, n), replace=True)
    means = samples.mean(axis=1)
    alpha = 1.0 - ci_level
    lower = float(np.quantile(means, alpha / 2.0))
    upper = float(np.quantile(means, 1.0 - alpha / 2.0))
    return lower, upper


def cross_validate_auprc(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int,
    ci_level: float,
    bootstrap_iters: int,
    seed: int,
) -> tuple[float, float, float, float]:
    splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    fold_scores = []

    for fold_idx, (train_idx, valid_idx) in enumerate(splitter.split(X, y)):
        fold_model = clone(model)
        X_fold_train = X.iloc[train_idx]
        y_fold_train = y.iloc[train_idx]
        X_fold_valid = X.iloc[valid_idx]
        y_fold_valid = y.iloc[valid_idx]

        fold_model.fit(X_fold_train, y_fold_train)
        if hasattr(fold_model, "predict_proba"):
            y_score = fold_model.predict_proba(X_fold_valid)[:, 1]
        else:
            y_score = fold_model.decision_function(X_fold_valid)

        fold_ap = average_precision_score(y_fold_valid, y_score)
        fold_scores.append(float(fold_ap))

    scores = np.asarray(fold_scores, dtype=float)
    mean_auprc = float(scores.mean())
    std_auprc = float(scores.std(ddof=1)) if scores.size > 1 else 0.0
    ci_lower, ci_upper = bootstrap_mean_ci(
        scores,
        ci_level=ci_level,
        iterations=bootstrap_iters,
        seed=seed,
    )
    return mean_auprc, std_auprc, ci_lower, ci_upper


def save_pr_curve(y_true: pd.Series, y_score, output_path: Path) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_score)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, linewidth=2)
    ax.set_title("Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)

    if not 0.0 < args.test_size < 1.0:
        raise ValueError("--test-size must be between 0 and 1.")
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("--threshold must be between 0 and 1.")
    if args.target_recall is not None and not 0.0 < args.target_recall <= 1.0:
        raise ValueError("--target-recall must be in (0, 1].")
    if args.target_precision is not None and not 0.0 < args.target_precision <= 1.0:
        raise ValueError("--target-precision must be in (0, 1].")
    if args.target_recall is not None and args.target_precision is not None:
        raise ValueError(
            "Use only one tuning target at a time: --target-recall or --target-precision."
        )
    if args.cv_folds < 2:
        raise ValueError("--cv-folds must be >= 2.")
    if not 0.0 < args.ci_level < 1.0:
        raise ValueError("--ci-level must be between 0 and 1.")
    if args.cv_bootstrap_iters < 100:
        raise ValueError("--cv-bootstrap-iters must be >= 100.")

    X, y = load_data(csv_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    models = get_models(args.seed)
    rows = []

    best_model_name = None
    best_ap = -1.0
    best_scores = None

    for name, model in models.items():
        cv_mean_ap, cv_std_ap, cv_ci_low, cv_ci_high = cross_validate_auprc(
            model,
            X_train,
            y_train,
            cv_folds=args.cv_folds,
            ci_level=args.ci_level,
            bootstrap_iters=args.cv_bootstrap_iters,
            seed=args.seed,
        )

        model.fit(X_train, y_train)

        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        else:
            y_score = model.decision_function(X_test)

        threshold_used, threshold_mode = tune_threshold(
            y_test,
            y_score,
            target_recall=args.target_recall,
            target_precision=args.target_precision,
            fallback_threshold=args.threshold,
        )

        ap = average_precision_score(y_test, y_score)
        precision_at_t, recall_at_t, (tn, fp, fn, tp) = evaluate_threshold_metrics(
            y_test, y_score, threshold_used
        )

        rows.append(
            {
                "model": name,
                "auprc": ap,
                "cv_mean_auprc": cv_mean_ap,
                "cv_std_auprc": cv_std_ap,
                "cv_ci_lower": cv_ci_low,
                "cv_ci_upper": cv_ci_high,
                "threshold_used": threshold_used,
                "threshold_mode": threshold_mode,
                "precision@threshold": precision_at_t,
                "recall@threshold": recall_at_t,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
            }
        )

        if ap > best_ap:
            best_ap = ap
            best_model_name = name
            best_scores = y_score

    metrics_df = pd.DataFrame(rows).sort_values(by="auprc", ascending=False)

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics_summary.txt"

    with metrics_path.open("w", encoding="utf-8") as f:
        f.write("Fraud Detection Baseline Metrics\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Samples: {len(y)}\n")
        f.write(f"Fraud count: {int(y.sum())}\n")
        f.write(f"Fraud rate: {float(y.mean()):.6f}\n")
        f.write(f"Test size: {args.test_size}\n")
        f.write(f"CV folds: {args.cv_folds}\n")
        f.write(f"CI level: {args.ci_level}\n")
        f.write(f"CV bootstrap iterations: {args.cv_bootstrap_iters}\n")
        f.write(f"Threshold fallback: {args.threshold}\n")
        f.write(f"Target recall: {args.target_recall}\n")
        f.write(f"Target precision: {args.target_precision}\n\n")
        f.write(metrics_df.to_string(index=False))
        f.write("\n")

    print("\nModel comparison (sorted by AUPRC):")
    print(metrics_df.to_string(index=False))

    if best_model_name is None or best_scores is None:
        raise RuntimeError("No model was trained successfully.")

    pr_curve_path = output_dir / "pr_curve.png"
    save_pr_curve(y_test, best_scores, pr_curve_path)

    print("\nBest model:", best_model_name)
    print("AUPRC:", f"{best_ap:.6f}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved PR curve to: {pr_curve_path}")


if __name__ == "__main__":
    main()
