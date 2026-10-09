"""Non-training regression and leakage checks. Run from the project root."""

import contextlib
import io
import re
from pathlib import Path
from unittest.mock import patch

import nbformat
import numpy as np
import pandas as pd
from catboost import Pool


def load(path, preset):
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    namespace = {}
    with patch("catboost.CatBoostClassifier.fit", side_effect=AssertionError("Training forbidden in checks")):
        with contextlib.redirect_stdout(io.StringIO()):
            for cell in notebook.cells:
                if cell.cell_type != "code":
                    continue
                source = cell.source
                if cell.id == "configuration":
                    source = re.sub(r"^RUN_MODE = .*$", 'RUN_MODE = "check"', source, flags=re.M)
                    source = re.sub(r"^EXPERIMENT = .*$", f'EXPERIMENT = "{preset}"', source, flags=re.M)
                exec(compile(source, cell.id, "exec"), namespace)
    return namespace


def main():
    ns = load(Path("catboost.ipynb"), "decision_logloss")
    assert len(ns["FEATURE_NAMES"]) == 216
    margin = np.zeros((2, 168))
    margin[0, :5] = [-1, 2, 3, -2, 4]
    margin[1] = -1
    features = ns["economic_context"](margin, margin)
    np.testing.assert_array_equal(features["economic_positive_streak_total_h"][0, :5], [0, 2, 2, 0, 1])
    np.testing.assert_array_equal(features["economic_positive_streak_margin_sum"][0, :5], [0, 5, 5, 0, 4])
    np.testing.assert_array_equal(features["economic_hours_to_next_positive"][0, :6], [1, 0, 0, 1, 0, 168])
    assert (features["economic_hours_to_next_positive"][1] == 168).all()
    assert features["economic_next_6h_margin_mean"][0, 0] == 1
    assert features["economic_previous_6h_margin_mean"][0, 1] == 0.5
    assert features["economic_next_24h_margin_mean"][1, -1] == -1
    assert features["economic_positive_streak_total_h"][1, 0] == 0

    initial = np.array([[9., 0.]])
    cumulative = np.zeros((1, 168, 2)); cumulative[:, :, 0] = 3
    remaining = np.zeros_like(cumulative); remaining[:, :, 0] = 4
    limits = np.zeros_like(cumulative); limits[:, 10:, 1] = 50
    water = ns["water_context"](initial, cumulative, remaining, limits, [10, 100])
    np.testing.assert_allclose(water["water_no_release_excess_fraction"], 2/110)
    np.testing.assert_allclose(water["water_end_no_release_excess_fraction"], 6/110)
    np.testing.assert_allclose(water["water_week_inflow_to_headroom_log_max"], np.log(8))
    np.testing.assert_allclose(water["water_initial_margin_to_future_floor_fraction"], -41/110)
    np.testing.assert_allclose(water["water_no_release_floor_deficit_fraction"][:, 10:], 50/110)
    assert (water["water_no_release_floor_deficit_fraction"][:, :10] == 0).all()

    cases = ns["train_cases"].iloc[[366, 0, 1400]].copy()
    base = ns["feature_matrix"](cases)
    expected = {g: ns["generator_features"](base, g) for g in ns["GENERATORS"]}
    single = ns["feature_matrix"](cases.iloc[[1]])
    for g in ns["GENERATORS"]:
        np.testing.assert_array_equal(expected[g].iloc[168:336], ns["generator_features"](single, g))
    # Mutate all case labels and optimizer metadata; features must be unchanged.
    poisoned = cases.copy()
    poisoned[ns["LABELS"]] = 1 - poisoned[ns["LABELS"]]
    poisoned["total_calculation_time"] = 1e20
    pd.testing.assert_frame_equal(ns["feature_matrix"](poisoned), base)
    # Modify realized storage inside a planning week, preserving its initial state.
    volumes = ns["tables"]["volumes"]
    first = cases.iloc[[0]]
    before = ns["generator_features"](ns["feature_matrix"](first), "Songa_G1")
    future = pd.date_range(first.starttime.iloc[0] + pd.Timedelta(days=1), periods=6, freq="D")
    saved = volumes.loc[future].copy()
    try:
        volumes.loc[future] = 0
        pd.testing.assert_frame_equal(before, ns["generator_features"](ns["feature_matrix"](first), "Songa_G1"))
    finally:
        volumes.loc[future] = saved
    pool = Pool(expected["Haukeli_G1"], label=ns["labels_for"](cases, "Haukeli_G1"), thread_count=4)
    assert pool.num_col() == 216 and pool.num_row() == 504
    print("Synthetic boundaries, physical units, per-reservoir excess, case isolation, leakage and CatBoost input: PASS")

    baseline = load(Path("catboost.ipynb"), "probability_logloss")
    baseline_features = baseline["FEATURE_NAMES"]
    for g in ns["GENERATORS"]:
        pd.testing.assert_frame_equal(expected[g][baseline_features], baseline["generator_features"](base, g))
    for preset, size in [("economics_logloss", 207), ("water_balance_logloss", 202),
                         ("baseline_auc", 193), ("upstream_logloss", 224), ("shared_logloss", 194)]:
        candidate = load(Path("catboost.ipynb"), preset)
        assert len(candidate["FEATURE_NAMES"]) == size
        for g in ns["GENERATORS"]:
            matrix = candidate["generator_features"](candidate["feature_matrix"](cases), g)
            if preset in {"economics_logloss", "water_balance_logloss"}:
                pd.testing.assert_frame_equal(matrix, expected[g][matrix.columns])
        print(preset, "check mode and feature schema: PASS")
    # Optional local backup: verify old presets' exact values and frozen signatures.
    backup = Path("outputs/catboost/notebook_before_decision_features.ipynb")
    if backup.exists():
        for preset in ("probability_logloss", "upstream_logloss"):
            old, new = load(backup, preset), load(Path("catboost.ipynb"), preset)
            assert old["FEATURE_SIGNATURE"] == new["FEATURE_SIGNATURE"]
            old_base, new_base = old["feature_matrix"](cases), new["feature_matrix"](cases)
            for g in ns["GENERATORS"]:
                pd.testing.assert_frame_equal(old["generator_features"](old_base, g), new["generator_features"](new_base, g))
        print("Existing feature values and frozen configuration signatures unchanged: PASS")
    print("No model fitting or dependency installation performed.")


if __name__ == "__main__":
    main()
