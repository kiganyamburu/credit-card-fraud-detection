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

- `train_baseline.py`: trains and evaluates strong classical baselines
	- Stratified train/test split
	- Models with class imbalance handling
	- AUPRC-first evaluation
	- Precision, recall, and confusion matrix at a configurable threshold
	- Saved precision-recall curve image

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

## Outputs

After running, the script writes:

- `outputs/metrics_summary.txt`
- `outputs/pr_curve.png`

and prints a concise comparison table in the terminal.

## Notes on methodology

- Accuracy is not a meaningful primary metric for this dataset.
- AUPRC is used as the ranking metric.
- Thresholded metrics are reported to help downstream operations tuning.

## Citation and acknowledgement

If you use this dataset in academic work, please cite the original studies from:

- Andrea Dal Pozzolo et al., *Calibrating Probability with Undersampling for Unbalanced Classification* (CIDM, IEEE, 2015)
- Dal Pozzolo et al., *Learned lessons in credit card fraud detection from a practitioner perspective* (Expert Systems with Applications, 2014)
- Dal Pozzolo et al., *Credit card fraud detection: a realistic modeling and a novel learning strategy* (IEEE TNNLS, 2018)
- Dal Pozzolo, *Adaptive Machine learning for credit card fraud detection* (PhD thesis)
- Carcillo et al., *SCARFF: a scalable framework for streaming credit card fraud detection with Spark* (Information Fusion, 2018)
- Carcillo et al., *Streaming active learning strategies for real-life credit card fraud detection: assessment and visualization* (IJDASA, 2018)
- Lebichot et al., *Deep-Learning Domain Adaptation Techniques for Credit Cards Fraud Detection* (INNSBDDL, 2019)
- Carcillo et al., *Combining Unsupervised and Supervised Learning in Credit Card Fraud Detection* (Information Sciences, 2019)
- Le Borgne and Bontempi, *Reproducible machine Learning for Credit Card Fraud Detection - Practical Handbook*
- Lebichot et al., *Incremental learning strategies for credit cards fraud detection* (IJDASA)

Additional simulator resource:

- https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_3_GettingStarted/SimulatedDataset.html
