import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare imbalance handling methods (SMOTE, undersampling, calibration) "
            "for credit card fraud detection."
        )
    )
    parser.add_argument("--csv", type=str, default="creditcard.csv", help="Path to CSV.")
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set fraction used in train/test split.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for confusion-matrix-derived metrics.",
    )
    parser.add_argument(
        "--bootstrap-iters",
        type=int,
        default=2000,
        help="Bootstrap iterations used for AUPRC confidence intervals on test set.",
    )
    parser.add_argument(
        "--ci-level",
        type=float,
        default=0.95,
        help="Confidence level for AUPRC bootstrap CI.",
    )
    parser.add_argument(
        "--calibration-method",
        choices=["sigmoid", "isotonic"],
        default="sigmoid",
        help="Probability calibration method for calibrated experiment.",
    )
    return parser.parse_args()


def load_data(csv_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "Class" not in df.columns:
        raise ValueError("Column 'Class' is required.")

    y = df["Class"].astype(int)
    X = df.drop(columns=["Class"])
    return X, y


def bootstrap_auprc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    iterations: int,
    ci_level: float,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = y_true.shape[0]
    stats = []

    while len(stats) < iterations:
        idx = rng.integers(0, n, size=n)
        sample_y = y_true[idx]
        if np.unique(sample_y).size < 2:
            continue
        sample_score = y_score[idx]
        stats.append(average_precision_score(sample_y, sample_score))

    arr = np.asarray(stats, dtype=float)
    alpha = 1.0 - ci_level
    lower = float(np.quantile(arr, alpha / 2.0))
    upper = float(np.quantile(arr, 1.0 - alpha / 2.0))
    return lower, upper


def evaluate_predictions(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    ci_level: float,
    bootstrap_iters: int,
    seed: int,
) -> dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    auprc = average_precision_score(y_true, y_score)
    auprc_ci_low, auprc_ci_high = bootstrap_auprc_ci(
        y_true,
        y_score,
        iterations=bootstrap_iters,
        ci_level=ci_level,
        seed=seed,
    )

    return {
        "auprc": float(auprc),
        "auprc_ci_lower": auprc_ci_low,
        "auprc_ci_upper": auprc_ci_high,
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "brier": float(brier_score_loss(y_true, y_score)),
        "precision@threshold": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall@threshold": float(recall_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def build_experiments(seed: int, calibration_method: str) -> dict[str, object]:
    baseline = Pipeline(
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

    undersampling = ImbPipeline(
        steps=[
            ("undersample", RandomUnderSampler(random_state=seed)),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    solver="lbfgs",
                    random_state=seed,
                ),
            ),
        ]
    )

    smote = ImbPipeline(
        steps=[
            ("smote", SMOTE(random_state=seed)),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    solver="lbfgs",
                    random_state=seed,
                ),
            ),
        ]
    )

    calibrated_under = CalibratedClassifierCV(
        estimator=ImbPipeline(
            steps=[
                ("undersample", RandomUnderSampler(random_state=seed)),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        solver="lbfgs",
                        random_state=seed,
                    ),
                ),
            ]
        ),
        method=calibration_method,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=seed),
    )

    return {
        "baseline_class_weight": baseline,
        "random_undersampling": undersampling,
        "smote": smote,
        "undersampling_calibrated_probs": calibrated_under,
    }


def main() -> None:
    args = parse_args()

    if not 0.0 < args.test_size < 1.0:
        raise ValueError("--test-size must be in (0, 1).")
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("--threshold must be in (0, 1).")
    if args.bootstrap_iters < 100:
        raise ValueError("--bootstrap-iters must be >= 100.")
    if not 0.0 < args.ci_level < 1.0:
        raise ValueError("--ci-level must be in (0, 1).")

    X, y = load_data(Path(args.csv))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        stratify=y,
        random_state=args.seed,
    )

    experiments = build_experiments(args.seed, args.calibration_method)

    rows = []
    for name, model in experiments.items():
        model.fit(X_train, y_train)
        y_score = model.predict_proba(X_test)[:, 1]
        metrics = evaluate_predictions(
            y_true=y_test.to_numpy(),
            y_score=y_score,
            threshold=args.threshold,
            ci_level=args.ci_level,
            bootstrap_iters=args.bootstrap_iters,
            seed=args.seed,
        )
        rows.append({"experiment": name, **metrics})

    summary = pd.DataFrame(rows).sort_values(by="auprc", ascending=False)

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "imbalance_methods_summary.txt"
    csv_path = output_dir / "imbalance_methods_summary.csv"

    with summary_path.open("w", encoding="utf-8") as f:
        f.write("Imbalance Methods Comparison\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Samples: {len(y)}\n")
        f.write(f"Fraud count: {int(y.sum())}\n")
        f.write(f"Fraud rate: {float(y.mean()):.6f}\n")
        f.write(f"Test size: {args.test_size}\n")
        f.write(f"Threshold: {args.threshold}\n")
        f.write(f"Bootstrap iterations: {args.bootstrap_iters}\n")
        f.write(f"CI level: {args.ci_level}\n")
        f.write(f"Calibration method: {args.calibration_method}\n\n")
        f.write(summary.to_string(index=False))
        f.write("\n")

    summary.to_csv(csv_path, index=False)

    print("\nImbalance methods comparison (sorted by AUPRC):")
    print(summary.to_string(index=False))
    print(f"\nSaved text summary to: {summary_path}")
    print(f"Saved CSV summary to: {csv_path}")


if __name__ == "__main__":
    main()
