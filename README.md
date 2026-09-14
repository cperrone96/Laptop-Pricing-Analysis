# Laptop Price Modeling with Leakage-Controlled Validation

> **Scope:** An independently remediated IBM *Data Analysis with Python* coursework artifact. The extension adds a reproducible predictive design, held-out evaluation, uncertainty, and tests; it is not a production pricing system.

## Question and decision context

How much better can hardware and category specifications predict the dataset's listed laptop prices than a training-mean baseline? This is a small-sample modeling exercise for comparing validation practice—not a current appraisal tool.

## Design and data contract

- Observation unit: one laptop configuration; target: the dataset's USD-denominated `Price` field.
- Source: fixed IBM Skills Network CSV (238 rows), retrieved 2026-09-14. URL, checksum, and licensing caveat are in [`data/SOURCES.json`](data/SOURCES.json).
- Split: seeded 60% training / 20% interval calibration / 20% untouched test.
- Selection: ridge penalty chosen with five-fold cross-validation inside the training set only.
- Leakage control: imputation, scaling, and category levels are learned from each training fold; test data are evaluated once.
- Comparators: training-mean baseline and ridge regression.
- Uncertainty: test-RMSE case-bootstrap interval and an absolute-residual interval calibrated on a separate split.

## Findings

Ridge regression achieved test RMSE **$397** (case-bootstrap 95% interval: **$222–$606**), MAE **$246**, and R² **0.583**, versus **$615** RMSE for the training-mean baseline. The separately calibrated nominal 90% interval covered 93.8% of the 48 test cases with a $600 half-width. The small test set makes all estimates imprecise; coefficients describe this fitted model and are not causal effects. `python analysis.py` reproduces [`outputs/metrics.json`](outputs/metrics.json).

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python analysis.py
python -m unittest discover -s tests -v
jupyter execute laptop_pricing.ipynb
```

Outputs include test predictions/residuals, model coefficients, and machine-readable metrics. CI repeats the analysis, invariant tests, and clean-kernel notebook run.

## Limitations and nonclaims

- The source's sampling frame, collection date, and detailed currency/market context are not documented in the file.
- The dataset is small and may contain related configurations; no model-family grouping identifier is available.
- Random-split performance can overstate generalization to future products, new manufacturers, or market shifts.
- Association between a specification and price is not a causal price effect. This model is not suitable for automated pricing or purchasing decisions.
