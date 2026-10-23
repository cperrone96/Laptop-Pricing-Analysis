import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import analysis
from analysis import HEADERS, grouped_split, load_data, shuffled_group_folds, spec_groups

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def data():
    return load_data()


def test_source_checksum():
    sources = json.loads((ROOT / "data" / "SOURCES.json").read_text())
    digest = hashlib.sha256((ROOT / "data" / sources["file"]).read_bytes()).hexdigest()
    assert digest == sources["sha256"]


def test_schema_and_target(data):
    assert data.shape == (238, 12)
    assert list(data.columns) == HEADERS
    assert data["Price"].notna().all() and (data["Price"] > 0).all()


def test_identical_specs_share_a_group(data):
    groups = spec_groups(data)
    duplicated_specs = data[analysis.FEATURES].astype(str).duplicated(keep=False).to_numpy()
    assert (np.bincount(groups)[groups] > 1).tolist() == duplicated_specs.tolist()


def test_grouped_split_never_splits_a_group(data):
    groups = spec_groups(data)
    parts = grouped_split(groups)
    assert sum(len(p) for p in parts) == len(data)
    seen = [set(groups[p]) for p in parts]
    assert not (seen[0] & seen[1]) and not (seen[0] & seen[2]) and not (seen[1] & seen[2])
    assert abs(len(parts[0]) / len(data) - 0.6) < 0.05


def test_cv_folds_never_split_a_group(data):
    groups = spec_groups(data)
    folds = shuffled_group_folds(groups, seed=0)
    for g in np.unique(groups):
        assert len(set(folds[groups == g])) == 1


def test_ridge_beats_mean_baseline_in_nested_cv(data, monkeypatch):
    monkeypatch.setattr(analysis, "N_REPEATS", 1)
    ridge = analysis.nested_grouped_cv(data, *analysis.candidate_models()["ridge"])[0]
    baseline = analysis.nested_grouped_cv(data, *analysis.candidate_models()["mean_baseline"])[0]
    assert ridge["rmse_usd"] < baseline["rmse_usd"]
    assert baseline["r2"] <= 0.0
