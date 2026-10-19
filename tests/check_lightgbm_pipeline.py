"""Check the LightGBM pipeline without installing LightGBM or fitting models."""

import ast
import contextlib
import io
import json
import re
from pathlib import Path
from unittest.mock import patch

import nbformat
import numpy as np
import pandas as pd


def load(path, preset):
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    namespace = {}
    # Also protect against accidental execution if LightGBM is already installed.
    from importlib.util import find_spec
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch("catboost.CatBoostClassifier.fit", side_effect=AssertionError("Training forbidden")))
        if find_spec("lightgbm"):
            stack.enter_context(patch("lightgbm.LGBMClassifier.fit", side_effect=AssertionError("Training forbidden")))
        with contextlib.redirect_stdout(io.StringIO()):
            for cell in notebook.cells:
                if cell.cell_type != "code":
                    continue
                source = cell.source
                if cell.id == "configuration":
                    source = re.sub(r"^RUN_MODE = .*$", 'RUN_MODE = "check"', source, flags=re.M)
                    source = re.sub(r"^EXPERIMENT = .*$", f'EXPERIMENT = "{preset}"', source, flags=re.M)
                ast.parse(source)
                exec(compile(source, cell.id, "exec"), namespace)
    return namespace, notebook


def main():
    lgb, notebook = load(Path("lightgbm.ipynb"), "decision_logloss")
    cat, _ = load(Path("catboost.ipynb"), "decision_logloss")
    assert len(lgb["FEATURE_NAMES"]) == 216
    assert lgb["FEATURE_SIGNATURE"] == cat["FEATURE_SIGNATURE"]
    assert lgb["FOLDS"] == cat["FOLDS"] and lgb["SEED"] == cat["SEED"]
    cases = pd.concat([lgb["train_cases"].iloc[[1300, 0, 2500]],
                       lgb["test_cases"].iloc[[0, -1]]], ignore_index=True)
    left, right = lgb["feature_matrix"](cases), cat["feature_matrix"](cases)
    for generator in lgb["GENERATORS"]:
        pd.testing.assert_frame_equal(lgb["generator_features"](left, generator),
                                      cat["generator_features"](right, generator))
    for start, end in lgb["FOLDS"]:
        fit, val = lgb["chronological_split"](start, end)
        fit_cat, val_cat = cat["chronological_split"](start, end)
        np.testing.assert_array_equal(fit["Run No"], fit_cat["Run No"])
        np.testing.assert_array_equal(val["Run No"], val_cat["Run No"])
        assert (fit.starttime + pd.Timedelta(hours=168)).max() <= val.starttime.min()
    # All 725 submission IDs, including the final required planning horizon.
    test_cases = lgb["test_cases"]
    predictions = {g: np.tile((np.arange(168) + i * 168) / 2352, (len(test_cases), 1))
                   for i, g in enumerate(lgb["GENERATORS"])}
    frame = lgb["assemble_predictions"](test_cases, predictions)
    assert frame.shape == (725, 2353) and frame.columns.equals(lgb["template"].columns)
    np.testing.assert_array_equal(frame["Run No"], lgb["template"]["Run No"])
    for g in lgb["GENERATORS"]:
        np.testing.assert_allclose(frame[lgb["TARGET_COLUMNS"][g]], predictions[g])
    # Verify official metric equivalence on deliberately reversed case order.
    small = lgb["train_cases"].iloc[[21, 0, 13]].copy()
    rng = np.random.default_rng(17)
    probabilities = {g: rng.random((len(small), 168)).astype(np.float32) for g in lgb["GENERATORS"]}
    score, _, predicted = lgb["evaluate"](small, probabilities)
    metric = {}
    exec(compile(Path("kaggle_metric.py").read_text(), "supplied metric", "exec"), metric)
    assert np.isclose(score, metric["score"](small[["Run No"] + lgb["LABELS"]], predicted), atol=1e-12)
    # A CatBoost frozen config with the SAME features must still be rejected.
    run_source = next(cell.source for cell in notebook.cells if cell.id == "run")
    old = dict(cat["DECISION_REFERENCE_CONFIG"])
    lgb["RUN_MODE"] = "submit"
    lgb["SELECTED_CONFIG"] = old
    try:
        exec(compile(run_source, "family guard", "exec"), lgb)
    except ValueError as error:
        assert "LightGBM configuration" in str(error)
    else:
        raise AssertionError("CatBoost frozen config was accepted")
    assert lgb["RUN_DIRECTORY"] is None
    baseline, _ = load(Path("lightgbm.ipynb"), "baseline_logloss")
    assert len(baseline["FEATURE_NAMES"]) == 193
    assert lgb["comparison"]["same_inputs_features_folds_seed"].any()
    print("PASS: check-mode cells, exact CatBoost features/folds, 725-row mapping, official metric, model-family guard, baseline ablation and comparison hashes.")
    print("No models fitted. LightGBM runtime and predictive performance remain unverified.")


if __name__ == "__main__":
    main()
