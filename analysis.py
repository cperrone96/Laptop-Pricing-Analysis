"""Leakage-controlled ridge regression for the IBM laptop-pricing dataset."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "raw" / "laptop_pricing_dataset_base.csv"
HEADERS = [
    "Manufacturer", "Category", "Screen", "GPU", "OS", "CPU_core",
    "Screen_Size_inch", "CPU_frequency", "RAM_GB", "Storage_GB_SSD",
    "Weight_kg", "Price",
]
CATEGORICAL = ["Manufacturer", "Category", "Screen", "GPU", "OS"]
NUMERIC = ["CPU_core", "Screen_Size_inch", "CPU_frequency", "RAM_GB", "Storage_GB_SSD", "Weight_kg"]


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    data = pd.read_csv(path, header=None, names=HEADERS, na_values="?")
    if list(data.columns) != HEADERS or data.empty:
        raise ValueError("Dataset does not satisfy the expected schema")
    for column in NUMERIC + ["Price"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if data["Price"].isna().any() or (data["Price"] <= 0).any():
        raise ValueError("Target price must be complete and positive")
    return data


@dataclass
class Encoder:
    medians: dict[str, float]
    means: dict[str, float]
    scales: dict[str, float]
    levels: dict[str, list[str]]
    feature_names: list[str]

    @classmethod
    def fit(cls, frame: pd.DataFrame) -> "Encoder":
        medians = {c: float(frame[c].median()) for c in NUMERIC}
        means = {c: float(frame[c].fillna(medians[c]).mean()) for c in NUMERIC}
        scales = {c: float(frame[c].fillna(medians[c]).std(ddof=0)) or 1.0 for c in NUMERIC}
        levels = {c: sorted(frame[c].astype(str).unique().tolist()) for c in CATEGORICAL}
        names = NUMERIC + [f"{c}={level}" for c in CATEGORICAL for level in levels[c]]
        return cls(medians, means, scales, levels, names)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        blocks = []
        numeric = np.column_stack([
            (frame[c].fillna(self.medians[c]).to_numpy(float) - self.means[c]) / self.scales[c]
            for c in NUMERIC
        ])
        blocks.append(numeric)
        for column in CATEGORICAL:
            values = frame[column].astype(str).to_numpy()
            blocks.append(np.column_stack([(values == level).astype(float) for level in self.levels[column]]))
        return np.column_stack(blocks)


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    design = np.column_stack([np.ones(len(x)), x])
    penalty = np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + alpha * penalty, design.T @ y)


def predict(x: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(x)), x]) @ coefficients


def regression_metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    residual = y - prediction
    mse = float(np.mean(residual**2))
    denominator = float(np.sum((y - y.mean()) ** 2))
    return {
        "rmse_usd": float(np.sqrt(mse)),
        "mae_usd": float(np.mean(np.abs(residual))),
        "r2": float(1 - np.sum(residual**2) / denominator),
        "mean_error_usd": float(np.mean(residual)),
    }


def cross_validated_alpha(frame: pd.DataFrame, alphas: list[float]) -> tuple[float, dict[str, float]]:
    folds = np.arange(len(frame)) % 5
    scores: dict[str, float] = {}
    for alpha in alphas:
        errors = []
        for fold in range(5):
            train, validation = frame.iloc[folds != fold], frame.iloc[folds == fold]
            encoder = Encoder.fit(train)
            coefficients = fit_ridge(encoder.transform(train), train["Price"].to_numpy(float), alpha)
            estimate = predict(encoder.transform(validation), coefficients)
            errors.extend((validation["Price"].to_numpy(float) - estimate) ** 2)
        scores[str(alpha)] = float(np.sqrt(np.mean(errors)))
    selected = min(alphas, key=lambda alpha: scores[str(alpha)])
    return selected, scores


def bootstrap_rmse_ci(y: np.ndarray, estimate: np.ndarray, seed: int = 42) -> list[float]:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(2000):
        idx = rng.integers(0, len(y), len(y))
        values.append(float(np.sqrt(np.mean((y[idx] - estimate[idx]) ** 2))))
    return [float(x) for x in np.quantile(values, [0.025, 0.975])]


def run(output_dir: Path | None = None) -> dict:
    output_dir = output_dir or ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_data()
    rng = np.random.default_rng(42)
    order = rng.permutation(len(data))
    train_end, calibration_end = int(0.60 * len(data)), int(0.80 * len(data))
    train = data.iloc[order[:train_end]].copy()
    calibration = data.iloc[order[train_end:calibration_end]].copy()
    test = data.iloc[order[calibration_end:]].copy()

    alphas = [0.1, 1.0, 10.0, 100.0]
    selected_alpha, cv_rmse = cross_validated_alpha(train, alphas)
    encoder = Encoder.fit(train)
    coefficients = fit_ridge(encoder.transform(train), train["Price"].to_numpy(float), selected_alpha)
    calibration_prediction = predict(encoder.transform(calibration), coefficients)
    test_prediction = predict(encoder.transform(test), coefficients)
    baseline_prediction = np.repeat(train["Price"].mean(), len(test))

    interval_half_width = float(np.quantile(np.abs(calibration["Price"].to_numpy(float) - calibration_prediction), 0.90))
    y_test = test["Price"].to_numpy(float)
    model_metrics = regression_metrics(y_test, test_prediction)
    lower, upper = test_prediction - interval_half_width, test_prediction + interval_half_width
    model_metrics["bootstrap_rmse_95_ci_usd"] = bootstrap_rmse_ci(y_test, test_prediction)
    model_metrics["empirical_90pct_interval_coverage"] = float(np.mean((y_test >= lower) & (y_test <= upper)))
    model_metrics["empirical_90pct_interval_half_width_usd"] = interval_half_width
    baseline_metrics = regression_metrics(y_test, baseline_prediction)

    predictions = test[["Manufacturer", "Price"]].copy()
    predictions["predicted_price"] = test_prediction
    predictions["residual"] = y_test - test_prediction
    predictions["interval_lower"] = lower
    predictions["interval_upper"] = upper
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)

    coefficient_table = pd.DataFrame({"feature": encoder.feature_names, "coefficient": coefficients[1:]})
    coefficient_table["absolute_coefficient"] = coefficient_table["coefficient"].abs()
    coefficient_table.sort_values("absolute_coefficient", ascending=False).to_csv(output_dir / "coefficients.csv", index=False)

    metrics = {
        "analysis_type": "predictive_coursework",
        "observation_unit": "laptop configuration",
        "target": "listed price in the dataset's USD-denominated field",
        "rows": int(len(data)),
        "split": {"train": len(train), "calibration": len(calibration), "test": len(test), "seed": 42},
        "missing_values": {c: int(data[c].isna().sum()) for c in NUMERIC},
        "selected_alpha": selected_alpha,
        "five_fold_training_cv_rmse_usd": cv_rmse,
        "ridge_test": model_metrics,
        "mean_baseline_test": baseline_metrics,
        "nonclaim": "Performance on 48 held-out records does not establish accuracy for current or out-of-distribution laptop markets.",
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir), indent=2))
