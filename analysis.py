"""Laptop price prediction with leakage-controlled, grouped validation.

Two stages:

1. Model comparison: repeated grouped 5-fold cross-validation over all rows,
   with hyperparameters tuned in an inner grouped CV loop (nested CV). Rows
   with identical specifications share a group, so the same configuration is
   never in both a training and a validation fold.
2. Prediction intervals: the CV-selected model family is fit on a 60% training
   split, a split-conformal 90% interval is calibrated on a separate 20%, and
   coverage is checked once on the remaining 20%.

Run ``python analysis.py`` to regenerate ``outputs/``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "raw" / "laptop_pricing_dataset_base.csv"
HEADERS = [
    "Manufacturer", "Category", "Screen", "GPU", "OS", "CPU_core",
    "Screen_Size_inch", "CPU_frequency", "RAM_GB", "Storage_GB_SSD",
    "Weight_kg", "Price",
]
# Category, GPU and OS are integer codes in the source file, so they are treated as nominal.
CATEGORICAL = ["Manufacturer", "Category", "Screen", "GPU", "OS"]
NUMERIC = ["CPU_core", "Screen_Size_inch", "CPU_frequency", "RAM_GB", "Storage_GB_SSD", "Weight_kg"]
FEATURES = CATEGORICAL + NUMERIC
SEED = 42
N_REPEATS = 10
N_FOLDS = 5


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Read the headerless course CSV and validate the schema and target."""
    data = pd.read_csv(path, header=None, names=HEADERS, na_values="?")
    if list(data.columns) != HEADERS or data.empty:
        raise ValueError("Dataset does not satisfy the expected schema")
    for column in NUMERIC + ["Price"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if data["Price"].isna().any() or (data["Price"] <= 0).any():
        raise ValueError("Target price must be complete and positive")
    for column in CATEGORICAL:
        data[column] = data[column].astype(str)
    return data


def spec_groups(data: pd.DataFrame) -> np.ndarray:
    """Integer group id per row; rows with identical specifications share an id."""
    return data[FEATURES].astype(str).agg("|".join, axis=1).factorize()[0]


def preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = OneHotEncoder(handle_unknown="ignore")
    return ColumnTransformer([("num", numeric, NUMERIC), ("cat", categorical, CATEGORICAL)])


def candidate_models() -> dict[str, tuple[object, dict[str, list]]]:
    return {
        "mean_baseline": (DummyRegressor(strategy="mean"), {}),
        "ridge": (Ridge(), {"model__alpha": [0.1, 1.0, 10.0, 100.0]}),
        "random_forest": (
            RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=1),
            {"model__min_samples_leaf": [1, 3, 5], "model__max_features": [0.33, 1.0]},
        ),
    }


def build_estimator(model, grid: dict):
    pipe = Pipeline([("prep", preprocessor()), ("model", model)])
    if not grid:
        return pipe
    return GridSearchCV(pipe, grid, cv=GroupKFold(n_splits=N_FOLDS), scoring="neg_root_mean_squared_error", n_jobs=-1)


def regression_metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "rmse_usd": float(np.sqrt(mean_squared_error(y, prediction))),
        "mae_usd": float(mean_absolute_error(y, prediction)),
        "r2": float(r2_score(y, prediction)),
    }


def shuffled_group_folds(groups: np.ndarray, seed: int) -> np.ndarray:
    """Assign each group to one of N_FOLDS folds at random (groups never split)."""
    rng = np.random.default_rng(seed)
    unique = rng.permutation(np.unique(groups))
    fold_of_group = {g: i % N_FOLDS for i, g in enumerate(unique)}
    return np.array([fold_of_group[g] for g in groups])


def nested_grouped_cv(data: pd.DataFrame, model, grid) -> list[dict[str, float]]:
    """Per repeat, pool out-of-fold predictions for every row and score them."""
    X, y, groups = data[FEATURES], data["Price"].to_numpy(float), spec_groups(data)
    results = []
    for repeat in range(N_REPEATS):
        folds = shuffled_group_folds(groups, SEED + repeat)
        prediction = np.empty(len(y))
        for k in range(N_FOLDS):
            train, test = folds != k, folds == k
            est = build_estimator(model, grid)
            fit_kwargs = {"groups": groups[train]} if grid else {}
            est.fit(X[train], y[train], **fit_kwargs)
            prediction[test] = est.predict(X[test])
        results.append(regression_metrics(y, prediction))
    return results


def grouped_split(groups: np.ndarray, fractions=(0.6, 0.2, 0.2), seed: int = SEED) -> list[np.ndarray]:
    """Split rows into train/calibration/test by whole groups, close to the target fractions."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(np.unique(groups))
    sizes = pd.Series(groups).value_counts()
    cumulative = sizes.loc[order].cumsum().to_numpy() / len(groups)
    bounds = np.cumsum(fractions)[:-1]
    part_of_group = dict(zip(order, np.searchsorted(bounds, cumulative, side="left")))
    part = np.array([part_of_group[g] for g in groups])
    return [np.flatnonzero(part == i) for i in range(len(fractions))]


def conformal_interval_check(data: pd.DataFrame, model_name: str) -> tuple[dict, pd.DataFrame]:
    model, grid = candidate_models()[model_name]
    X, y, groups = data[FEATURES], data["Price"].to_numpy(float), spec_groups(data)
    train, calibration, test = grouped_split(groups)
    est = build_estimator(model, grid)
    est.fit(X.iloc[train], y[train], **({"groups": groups[train]} if grid else {}))
    residual = np.abs(y[calibration] - est.predict(X.iloc[calibration]))
    n_cal = len(calibration)
    # Finite-sample split-conformal quantile for 90% coverage.
    level = min(1.0, np.ceil(0.9 * (n_cal + 1)) / n_cal)
    half_width = float(np.quantile(residual, level, method="higher"))
    prediction = est.predict(X.iloc[test])
    covered = (y[test] >= prediction - half_width) & (y[test] <= prediction + half_width)
    table = data.iloc[test][["Manufacturer", "Price"]].copy()
    table["predicted_price"] = prediction
    table["interval_lower"] = prediction - half_width
    table["interval_upper"] = prediction + half_width
    summary = {
        "model": model_name,
        "split_rows": {"train": len(train), "calibration": n_cal, "test": len(test)},
        "nominal_coverage": 0.9,
        "half_width_usd": half_width,
        "empirical_test_coverage": float(covered.mean()),
        "test_metrics": regression_metrics(y[test], prediction),
    }
    return summary, table


def summarize(results: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    frame = pd.DataFrame(results)
    return {m: {"mean": float(frame[m].mean()), "min": float(frame[m].min()), "max": float(frame[m].max())} for m in frame}


def plot_cv_rmse(comparison: pd.DataFrame, path: Path) -> None:
    """Dot-and-range chart of CV RMSE per model."""
    from matplotlib.figure import Figure  # object API: no global backend or pyplot state

    ink, muted, blue, surface = "#0b0b0b", "#52514e", "#2a78d6", "#fcfcfb"
    labels = {"mean_baseline": "Mean baseline", "ridge": "Ridge regression", "random_forest": "Random forest"}
    rows = comparison[comparison.metric.eq("rmse_usd")].set_index("model").loc[list(labels)]
    fig = Figure(figsize=(7, 2.8), dpi=150, facecolor=surface)
    ax = fig.subplots()
    ax.set_facecolor(surface)
    for i, (model, row) in enumerate(rows.iterrows()):
        y = len(rows) - 1 - i
        ax.plot([row["min"], row["max"]], [y, y], color=blue, lw=2, solid_capstyle="round")
        ax.plot(row["mean"], y, "o", ms=8, color=blue, mec=surface, mew=2)
        ax.text(row["max"] + 8, y, f"${row['mean']:,.0f}", va="center", fontsize=9, color=ink)
    ax.set_yticks(range(len(rows)), [labels[m] for m in reversed(rows.index)], color=ink)
    ax.set_xlabel("Cross-validated RMSE, USD (mean and range over 10 repeats; lower is better)", color=muted, fontsize=9)
    ax.tick_params(colors=muted, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#d6d5d0")
    ax.grid(axis="x", color="#e9e8e4", lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(0, rows["max"].max() * 1.12)
    ax.set_ylim(-0.5, len(rows) - 0.5)
    ax.set_title("Laptop price prediction error by model", loc="left", fontsize=11, color=ink)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=surface)


def run(output_dir: Path | None = None) -> dict:
    output_dir = output_dir or ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_data()
    groups = spec_groups(data)

    rows, cv_summary = [], {}
    for name, (model, grid) in candidate_models().items():
        cv_summary[name] = summarize(nested_grouped_cv(data, model, grid))
        rows += [{"model": name, "metric": m, **s} for m, s in cv_summary[name].items()]
    comparison = pd.DataFrame(rows)
    comparison.to_csv(output_dir / "cv_comparison.csv", index=False)
    plot_cv_rmse(comparison, output_dir / "figures" / "cv_rmse.png")

    best = min((m for m in cv_summary if m != "mean_baseline"), key=lambda m: cv_summary[m]["rmse_usd"]["mean"])
    interval, predictions = conformal_interval_check(data, best)
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)

    ridge = build_estimator(*candidate_models()["ridge"]).fit(data[FEATURES], data["Price"], groups=groups)
    pipe = ridge.best_estimator_
    names = pipe.named_steps["prep"].get_feature_names_out()
    coefficients = pd.DataFrame({"feature": names, "coefficient_usd": pipe.named_steps["model"].coef_})
    coefficients["abs"] = coefficients["coefficient_usd"].abs()
    coefficients.sort_values("abs", ascending=False).drop(columns="abs").to_csv(output_dir / "coefficients.csv", index=False)

    metrics = {
        "rows": int(len(data)),
        "spec_groups": int(groups.max() + 1),
        "rows_sharing_specs_with_another_row": int(pd.Series(groups).duplicated(keep=False).sum()),
        "missing_values": {c: int(data[c].isna().sum()) for c in NUMERIC},
        "price_usd": {"median": float(data["Price"].median()), "min": float(data["Price"].min()), "max": float(data["Price"].max())},
        "validation": f"{N_REPEATS} repeats x {N_FOLDS}-fold grouped CV with inner grouped grid search (seed {SEED})",
        "cv": cv_summary,
        "best_model": best,
        "conformal_interval": interval,
        "ridge_alpha_full_data": float(ridge.best_params_["model__alpha"]),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Laptop price prediction with grouped nested CV")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir), indent=2))
