"""Audit the frozen standalone TCN from raw data without fitting parameters.

Run from the repository root. Mode tests use fabricated predictions in temporary
directories. Optimizer updates and pretrained weight loading are blocked.
"""
import argparse
import contextlib
import copy
import io
import json
import re
import tempfile
from pathlib import Path
from unittest.mock import patch

import nbformat
import numpy as np
import pandas as pd
import torch


def load_notebook(path):
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    ns = {}
    with contextlib.redirect_stdout(io.StringIO()):
        for cell in notebook.cells:
            if cell.cell_type == "code":
                source = cell.source
                if cell.id == "configuration":
                    # A user may have selected submit in the editor. Never fit here.
                    source = re.sub(r"^RUN_MODE = .*$", 'RUN_MODE = "check"', source, flags=re.M)
                exec(compile(source, cell.id, "exec"), ns)
    assert ns["RUN_DIRECTORY"] is None and ns["EXPERIMENT"] == "terminal_resolution_6h"
    ns["validate_frozen"](ns["SELECTED_CONFIG"])
    return ns


def check_inputs(ns):
    cases = ns["train_cases"].iloc[[1350, 0, 2500]].copy()
    d, s, names, static_names = ns["sequence_inputs"](cases)
    assert d.shape == (3, 168, 913) and s.shape == (3, 182)
    unlabeled = ns["prediction_metadata"](cases)
    assert not set(ns["LABELS"]) & set(unlabeled.columns)
    assert "total_calculation_time" not in unlabeled
    du, su, _, _ = ns["sequence_inputs"](unlabeled)
    np.testing.assert_array_equal(d, du)
    np.testing.assert_array_equal(s, su)
    workers = ns["BASE_PARAMETERS"]["allocation_precompute_workers"]
    try:
        results = []
        for count in (1, workers):
            ns["ALLOCATION_CACHE"].clear()
            ns["BASE_PARAMETERS"]["allocation_precompute_workers"] = count
            results.append(ns["sequence_inputs"](unlabeled))
        np.testing.assert_array_equal(results[0][0], results[1][0])
        np.testing.assert_array_equal(results[0][1], results[1][1])
    finally:
        ns["BASE_PARAMETERS"]["allocation_precompute_workers"] = workers
        ns["ALLOCATION_CACHE"].clear()
    one = unlabeled.iloc[[0]]
    before_d, before_s, _, _ = ns["sequence_inputs"](one)
    np.testing.assert_array_equal(d[:1], before_d)
    np.testing.assert_array_equal(s[:1], before_s)
    volume = ns["tables"]["volumes"].copy()
    try:
        future = pd.date_range(one.starttime.iloc[0] + pd.Timedelta(days=1), periods=7, freq="D")
        ns["tables"]["volumes"].loc[future] *= .123
        ns["ALLOCATION_CACHE"].clear()
        after_d, after_s, _, _ = ns["sequence_inputs"](one)
        np.testing.assert_array_equal(before_d, after_d)
        np.testing.assert_array_equal(before_s, after_s)
    finally:
        ns["tables"]["volumes"] = volume
        ns["ALLOCATION_CACHE"].clear()
    for start in cases.starttime:
        allocation = ns["allocation_case"](start)
        assert allocation["success"] and allocation["equality_error"] < 1e-5
        route = allocation["terminal_conversion_plant"]
        assert set(route) == set(ns["RESERVOIRS"])
        assert route["Staavatn"] == "Vesle Kjela" and route["Totak"] == "Vinje"
        assert route["Bordalsvatn"] == "Kjela" and route["Langeidvatn"] == "Haukeli"
        assert np.isfinite(allocation["terminal_global_money_per_Mm3"]).all()
        assert np.all(allocation["terminal_bestpoint_energy_MWh_per_Mm3"] >=
                      allocation["plant_energy_MWh_per_Mm3"] - 1e-8)
    base = ns["feature_matrix"](one)
    base.attrs["case_starts"] = pd.DatetimeIndex(one.starttime)
    _, projected, _ = ns["projected_hourly_state"](base)
    np.testing.assert_allclose(projected[:, 0], ns["lookup"]("volumes", one.starttime, ns["RESERVOIRS"]))
    times = pd.date_range(one.starttime.iloc[0], periods=168, freq="h")
    np.testing.assert_allclose(before_d[0, :, names.index("price")], ns["lookup"]("prices", times).ravel())
    for reservoir in ns["RESERVOIRS"]:
        terminal = ns["lookup"]("water_values", one.starttime + pd.Timedelta(hours=168), [reservoir])[0, 0]
        assert before_s[0, static_names.index(f"terminal_value__{reservoir}")] == terminal
    catalog = ns["feature_catalog"]
    pd.testing.assert_frame_equal(catalog, pd.read_csv(ns["ROOT"] / "docs/tcn_features.csv"), check_dtype=False)
    assert catalog.loc[catalog.tensor == "hourly", "feature"].tolist() == names
    assert catalog.loc[catalog.tensor == "static", "feature"].tolist() == static_names
    assert catalog["feature"].is_unique and not catalog.isna().any().any()
    return cases, d, s


def check_scaling_model_mapping(ns, cases, d, s):
    scaler = ns["fit_scaler"](d[:2], s[:2], 10)
    expected_d = np.arcsinh(d[:2] / scaler["dynamic_transform_scale"])
    expected_s = np.arcsinh(s[:2] / scaler["static_transform_scale"])
    np.testing.assert_allclose(scaler["dynamic_mean"], expected_d.mean((0, 1), dtype=np.float64), rtol=1e-6)
    np.testing.assert_allclose(scaler["static_mean"], expected_s.mean(0, dtype=np.float64), rtol=1e-6)
    scaled_d, scaled_s = ns["scale_inputs"](d, s, scaler)
    assert np.isfinite(scaled_d).all() and np.abs(scaled_d).max() <= 10
    assert np.isfinite(scaled_s).all() and np.abs(scaled_s).max() <= 10
    ns["initialize_torch"]()
    model = ns["make_model"](ns["BASE_PARAMETERS"]).eval()
    assert sum(p.numel() for p in model.parameters()) == 343034
    assert 1 + 4 * sum(ns["BASE_PARAMETERS"]["dilations"]) == 509
    logits = model(torch.from_numpy(scaled_d[:2]), torch.from_numpy(scaled_s[:2]))
    assert logits.shape == (2, 168, 14) and torch.isfinite(logits).all()
    unchanged = {key: value.detach().clone() for key, value in model.state_dict().items()}
    torch.nn.functional.binary_cross_entropy_with_logits(
        logits, torch.from_numpy(ns["sequence_labels"](cases.iloc[:2]))).backward()
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in model.parameters())
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, unchanged[key], rtol=0, atol=0)
    test = ns["test_cases"]
    probability = np.broadcast_to(np.arange(168 * 14).reshape(1, 168, 14) / (168 * 14), (725, 168, 14))
    frame = ns["sequence_frame"](test, probability)
    assert frame.shape == (725, 2353) and frame.columns.equals(ns["template"].columns)
    np.testing.assert_array_equal(frame["Run No"], ns["template"]["Run No"])
    for j, generator in enumerate(ns["GENERATORS"]):
        np.testing.assert_allclose(frame[ns["TARGET_COLUMNS"][generator]], probability[:, :, j])
    random = np.random.default_rng(12).random((len(cases), 168, 14)).astype(np.float32)
    metric = {}
    exec((ns["ROOT"] / "kaggle_metric.py").read_text(), metric)
    auc, _, _ = ns["evaluate"](cases, {g: random[:, :, j] for j, g in enumerate(ns["GENERATORS"])})
    expected = metric["score"](cases[["Run No"] + ns["LABELS"]], ns["sequence_frame"](cases, random))
    assert np.isclose(auc, expected, atol=1e-12)
    invalid = probability.copy()
    invalid[0, 0, 0] = np.nan
    try:
        ns["sequence_frame"](test, invalid)
    except ValueError:
        pass
    else:
        raise AssertionError("Nonfinite probabilities accepted")


def check_protocol(ns):
    calls = []

    def predictions(cases):
        base = (cases["Run No"].to_numpy()[:, None] % 997) / 1994 + .2
        return {g: (base + np.arange(168)[None, :] * 1e-6 + j * .001).astype(np.float32)
                for j, g in enumerate(ns["GENERATORS"])}

    def fake_fit(fitting, evaluation, parameters, epochs, folder, early_stop):
        assert not early_stop and not set(ns["LABELS"]) & set(evaluation.columns)
        assert "total_calculation_time" not in evaluation
        if ns["RUN_MODE"] != "submit":
            assert (fitting.starttime + pd.Timedelta(hours=168)).max() <= evaluation.starttime.min()
        # The final benchmark refit uses every supplied labeled case, including
        # late-December starts. Purging is mandatory for local scoring folds.
        calls.append((len(fitting), len(evaluation), epochs))
        return predictions(evaluation), {"epochs": epochs, "seconds": 0.}

    def fake_evaluation(cases, predicted, folder, prefix):
        # No new annual UC score is calculated by this orchestration test.
        for generator, expected in predictions(cases).items():
            np.testing.assert_array_equal(predicted[generator], expected)
        return {"micro_auc": .5, "logloss": .7, "brier": .25, "cases": len(cases)}

    with tempfile.TemporaryDirectory(prefix="tcn_protocol_") as directory:
        overrides = {"OUTPUT_ROOT": Path(directory), "SELECTED_CONFIG": None, "CONFIG_PATH": None,
                     "fit_and_predict": fake_fit, "save_evaluation": fake_evaluation, "RUN_MODE": "validate"}
        with patch.dict(ns, overrides), contextlib.redirect_stdout(io.StringIO()):
            folder, _ = ns["run_pipeline"]()
            assert calls == [(1820, 360, 20), (2186, 359, 42)]
            ns["SELECTED_CONFIG"] = json.loads((folder / "selected_configuration.json").read_text())
            ns["RUN_MODE"] = "freeze"
            folder, _ = ns["run_pipeline"]()
            assert len(calls) == 2
            frozen = json.loads((folder / "selected_configuration.json").read_text())
            assert frozen["epochs"] == 22
            ns["validate_frozen"](frozen)
            ns["SELECTED_CONFIG"] = frozen
            ns["RUN_MODE"] = "holdout"
            ns["run_pipeline"]()
            assert calls[-1] == (2551, 365, 22)
            ns["RUN_MODE"] = "submit"
            folder, _ = ns["run_pipeline"]()
            assert calls[-1] == (2922, 725, 22)
            frame = pd.read_csv(folder / "tcn_submission.csv")
            expected = ns["assemble_predictions"](ns["test_cases"], predictions(ns["test_cases"]))
            np.testing.assert_allclose(frame, expected, rtol=1e-7, atol=1e-7)
            for field, value in (("epochs", 23), ("feature_signature", "invalid"),
                                 ("implementation_signature", "invalid")):
                bad = copy.deepcopy(frozen)
                bad[field] = value
                try:
                    ns["validate_frozen"](bad)
                except ValueError:
                    pass
                else:
                    raise AssertionError(f"Changed frozen {field} accepted")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", default="sequence_model.ipynb")
    args = parser.parse_args()
    with patch("torch.optim.AdamW.step", side_effect=AssertionError("Training forbidden")), \
            patch("torch.load", side_effect=AssertionError("Pretrained loading forbidden")):
        ns = load_notebook(args.notebook)
        cases, dynamic, static = check_inputs(ns)
        check_scaling_model_mapping(ns, cases, dynamic, static)
        check_protocol(ns)
    print("PASS: frozen recipe, feature catalog, timestamps, label/future-volume invariance,")
    print("case isolation, scaler, gradients, official metric, mapping and fixed-budget orchestration.")
    print("No parameters fitted or pretrained weights loaded; mode tests used fabricated predictions.")


if __name__ == "__main__":
    main()
