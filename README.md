# Credit Card Fraud Detection Baseline

This project provides a reproducible baseline for binary fraud detection on the
`creditcard.csv` dataset (European cardholders, September 2013).

The dataset is highly imbalanced:

- Total transactions: 284,807
- Fraud cases (`Class = 1`): 492
- Fraud rate: 0.172%

Because of this imbalance, this project evaluates models with **Area Under the
Precision-Recall Curve (AUPRC / Average Precision)** instead of raw accuracy.

## Dataset schema

- `Time`: seconds elapsed between each transaction and the first transaction
- `V1` ... `V28`: anonymized PCA-transformed features
- `Amount`: transaction amount
- `Class`: target (`1` fraud, `0` non-fraud)

## What is included

- `train_baseline.py`: trains and evaluates strong classical baselines.
  - Stratified train/test split.
  - Stratified k-fold cross-validation on training data.
  - Models with class imbalance handling.
  - AUPRC-first evaluation.
  - Bootstrap confidence intervals for mean CV AUPRC.
  - Precision, recall, and confusion matrix at a configurable or tuned threshold.
  - Saved precision-recall curve image.
- `imbalance_experiments.py`: compares imbalance handling methods.
  - Baseline class-weighted logistic regression.
  - Random undersampling.
  - SMOTE oversampling.
  - Undersampling with calibrated probabilities
  - AUPRC, ROC-AUC, Brier score, and threshold metrics
- `report_experiments.py`: generates a compact unified report
  - One combined results table for baseline and imbalance experiments
  - One multi-panel plot set for AUPRC, threshold trade-offs, and calibration

## Quick start

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run baseline training:

```bash
python train_baseline.py
```

Optional arguments:

```bash
python train_baseline.py --csv creditcard.csv --test-size 0.2 --threshold 0.5 --seed 42
```

Cross-validation and confidence intervals:

```bash
python train_baseline.py --cv-folds 5 --ci-level 0.95 --cv-bootstrap-iters 2000
```

Threshold tuning options (choose one):

```bash
# Constrain recall and maximize precision
python train_baseline.py --target-recall 0.90

# Constrain precision and maximize recall
python train_baseline.py --target-precision 0.30
```

Imbalance methods experiment module:

```bash
python imbalance_experiments.py --csv creditcard.csv --threshold 0.5
```

Optional calibration and CI settings:

```bash
python imbalance_experiments.py --calibration-method sigmoid --bootstrap-iters 2000 --ci-level 0.95
```

Unified experiment report:

```bash
python report_experiments.py --csv creditcard.csv
```

## Outputs

After running, the script writes:

- `outputs/metrics_summary.txt`
- `outputs/pr_curve.png`

Experiment module outputs:

- `outputs/imbalance_methods_summary.txt`
- `outputs/imbalance_methods_summary.csv`

Unified report outputs:

- `outputs/experiment_report.csv`
- `outputs/experiment_report.txt`
- `outputs/experiment_report_plots.png`

and prints a concise comparison table in the terminal.

## Notes on methodology

- Accuracy is not a meaningful primary metric for this dataset.
- AUPRC is used as the ranking metric.
- CV mean AUPRC with confidence intervals provides variance-aware model comparison.
- Thresholded metrics are reported to help downstream operations tuning.
- You can tune threshold for operational constraints:
  - maximize precision subject to recall >= target
  - maximize recall subject to precision >= target

## Citation and acknowledgement

If you use this dataset in academic work, please cite the original studies from:

- Andrea Dal Pozzolo et al., _Calibrating Probability with Undersampling for Unbalanced Classification_ (CIDM, IEEE, 2015)
- Dal Pozzolo et al., _Learned lessons in credit card fraud detection from a practitioner perspective_ (Expert Systems with Applications, 2014)
- Dal Pozzolo et al., _Credit card fraud detection: a realistic modeling and a novel learning strategy_ (IEEE TNNLS, 2018)
- Dal Pozzolo, _Adaptive Machine learning for credit card fraud detection_ (PhD thesis)
- Carcillo et al., _SCARFF: a scalable framework for streaming credit card fraud detection with Spark_ (Information Fusion, 2018)
- Carcillo et al., _Streaming active learning strategies for real-life credit card fraud detection: assessment and visualization_ (IJDASA, 2018)
- Lebichot et al., _Deep-Learning Domain Adaptation Techniques for Credit Cards Fraud Detection_ (INNSBDDL, 2019)
- Carcillo et al., _Combining Unsupervised and Supervised Learning in Credit Card Fraud Detection_ (Information Sciences, 2019)
- Le Borgne and Bontempi, _Reproducible machine Learning for Credit Card Fraud Detection - Practical Handbook_
- Lebichot et al., _Incremental learning strategies for credit cards fraud detection_ (IJDASA)

Additional simulator resource:

- https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_3_GettingStarted/SimulatedDataset.html
