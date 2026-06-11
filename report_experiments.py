import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from imbalance_experiments import build_experiments, evaluate_predictions
from train_baseline import (
    cross_validate_auprc,
    evaluate_threshold_metrics,
    get_models,
    load_data,
    tune_threshold,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a compact experiment report that consolidates baseline and "
            "imbalance-method results into one table and one plot set."
        )
    )
    parser.add_argument("--csv", type=str, default="creditcard.csv", help="Path to CSV.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set fraction.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold.")
    parser.add_argument(
        "--target-recall",
        type=float,
        default=None,
        help="Optional threshold tuning target: recall constraint.",
    )
    parser.add_argument(
        "--target-precision",
        type=float,
        default=None,
        help="Optional threshold tuning target: precision constraint.",
    )
    parser.add_argument("--cv-folds", type=int, default=5, help="Stratified CV folds.")
    parser.add_argument("--ci-level", type=float, default=0.95, help="CI level.")
    parser.add_argument(
        "--cv-bootstrap-iters",
        type=int,
        default=2000,
        help="Bootstrap iterations for CV AUPRC confidence intervals.",
    )
    parser.add_argument(
        "--bootstrap-iters",
        type=int,
        default=1000,
        help="Bootstrap iterations for test-set AUPRC confidence intervals.",
    )
    parser.add_argument(
        "--calibration-method",
        choices=["sigmoid", "isotonic"],
        default="sigmoid",
        help="Calibration method for the undersampling-calibrated experiment.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 < args.test_size < 1.0:
        raise ValueError("--test-size must be between 0 and 1.")
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("--threshold must be between 0 and 1.")
    if args.target_recall is not None and not 0.0 < args.target_recall <= 1.0:
        raise ValueError("--target-recall must be in (0, 1].")
    if args.target_precision is not None and not 0.0 < args.target_precision <= 1.0:
        raise ValueError("--target-precision must be in (0, 1].")
    if args.target_recall is not None and args.target_precision is not None:
        raise ValueError("Use only one threshold tuning target at a time.")
    if args.cv_folds < 2:
        raise ValueError("--cv-folds must be >= 2.")
    if not 0.0 < args.ci_level < 1.0:
        raise ValueError("--ci-level must be between 0 and 1.")
    if args.cv_bootstrap_iters < 100:
        raise ValueError("--cv-bootstrap-iters must be >= 100.")
    if args.bootstrap_iters < 100:
        raise ValueError("--bootstrap-iters must be >= 100.")


def summarize_baselines(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    rows = []
    for name, model in get_models(args.seed).items():
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
        precision_at_t, recall_at_t, (tn, fp, fn, tp) = evaluate_threshold_metrics(
            y_test, y_score, threshold_used
        )

        test_auprc = float(average_precision_score(y_test, y_score))
        test_roc_auc = float(roc_auc_score(y_test, y_score))
        test_brier = float(brier_score_loss(y_test, y_score))

        rows.append(
            {
                "category": "baseline",
                "experiment": name,
                "auprc": test_auprc,
                "cv_mean_auprc": cv_mean_ap,
                "cv_std_auprc": cv_std_ap,
                "cv_ci_lower": cv_ci_low,
                "cv_ci_upper": cv_ci_high,
                "threshold_used": threshold_used,
                "threshold_mode": threshold_mode,
                "precision@threshold": precision_at_t,
                "recall@threshold": recall_at_t,
                "roc_auc": test_roc_auc,
                "brier": test_brier,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
                "source": "train_baseline.py",
            }
        )

    return rows


def summarize_imbalance_methods(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    rows = []
    for name, model in build_experiments(args.seed, args.calibration_method).items():
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
        rows.append(
            {
                "category": "imbalance",
                "experiment": name,
                "cv_mean_auprc": float(np.nan),
                "cv_std_auprc": float(np.nan),
                "cv_ci_lower": float(np.nan),
                "cv_ci_upper": float(np.nan),
                "threshold_used": args.threshold,
                "threshold_mode": "fixed",
                "source": "imbalance_experiments.py",
                **metrics,
            }
        )

    return rows


def render_report_plot(summary: pd.DataFrame, output_path: Path) -> None:
    ordered = summary.sort_values(by=["category", "cv_mean_auprc", "auprc"], ascending=[True, False, False])

    fig, axes = plt.subplots(1, 3, figsize=(19, 6))

    bar_metric = ordered["cv_mean_auprc"].fillna(ordered["auprc"])
    bar_lower = ordered["cv_ci_lower"].fillna(bar_metric)
    bar_upper = ordered["cv_ci_upper"].fillna(bar_metric)
    yerr = np.vstack([
        bar_metric - bar_lower,
        bar_upper - bar_metric,
    ])

    colors = ordered["category"].map({"baseline": "#2F6BFF", "imbalance": "#E45756"})
    axes[0].barh(ordered["experiment"], bar_metric, color=colors)
    axes[0].errorbar(bar_metric, ordered["experiment"], xerr=yerr, fmt="none", ecolor="black", capsize=3)
    axes[0].set_title("AUPRC / CV Mean AUPRC")
    axes[0].set_xlabel("Score")

    axes[1].scatter(
        ordered["recall@threshold"],
        ordered["precision@threshold"],
        c=colors,
        s=90,
        alpha=0.9,
    )
    for _, row in ordered.iterrows():
        axes[1].annotate(row["experiment"], (row["recall@threshold"], row["precision@threshold"]), xytext=(5, 4), textcoords="offset points", fontsize=8)
    axes[1].set_title("Threshold Precision vs Recall")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].grid(alpha=0.25)

    axes[2].scatter(ordered["brier"].fillna(0.0), bar_metric, c=colors, s=90, alpha=0.9)
    for _, row in ordered.iterrows():
        brier_value = 0.0 if pd.isna(row["brier"]) else row["brier"]
        axes[2].annotate(
            row["experiment"],
            (brier_value, row["auprc_display"]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
        )
    axes[2].set_title("Calibration View")
    axes[2].set_xlabel("Brier score")
    axes[2].set_ylabel("AUPRC")
    axes[2].grid(alpha=0.25)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    validate_args(args)

    X, y = load_data(Path(args.csv))
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        stratify=y,
        random_state=args.seed,
    )

    rows = []
    rows.extend(summarize_baselines(X_train, y_train, X_test, y_test, args))
    rows.extend(summarize_imbalance_methods(X_train, y_train, X_test, y_test, args))

    summary = pd.DataFrame(rows)
    summary["auprc_display"] = summary["auprc"].fillna(summary["cv_mean_auprc"])
    summary = summary.sort_values(by=["category", "auprc_display"], ascending=[True, False])

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    table_path = output_dir / "experiment_report.csv"
    text_path = output_dir / "experiment_report.txt"
    plot_path = output_dir / "experiment_report_plots.png"

    summary.to_csv(table_path, index=False)

    with text_path.open("w", encoding="utf-8") as f:
        f.write("Experiment Report\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Samples: {len(y)}\n")
        f.write(f"Fraud count: {int(y.sum())}\n")
        f.write(f"Fraud rate: {float(y.mean()):.6f}\n")
        f.write(f"Test size: {args.test_size}\n")
        f.write(f"Threshold: {args.threshold}\n")
        f.write(f"CV folds: {args.cv_folds}\n")
        f.write(f"CV bootstrap iterations: {args.cv_bootstrap_iters}\n")
        f.write(f"Test bootstrap iterations: {args.bootstrap_iters}\n")
        f.write(f"CI level: {args.ci_level}\n")
        f.write(f"Calibration method: {args.calibration_method}\n")
        f.write(f"Target recall: {args.target_recall}\n")
        f.write(f"Target precision: {args.target_precision}\n\n")
        f.write(summary.drop(columns=["auprc_display"]).to_string(index=False))
        f.write("\n")

    render_report_plot(summary, plot_path)

    print("\nExperiment report table:")
    print(summary.drop(columns=["auprc_display"]).to_string(index=False))
    print(f"\nSaved table to: {table_path}")
    print(f"Saved text report to: {text_path}")
    print(f"Saved plot set to: {plot_path}")


if __name__ == "__main__":
    main()