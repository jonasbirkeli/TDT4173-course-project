"""Audit sequence alignment, fold-only normalization and mapping without fitting."""

import ast
import argparse
import contextlib
import io
import json
import re
from pathlib import Path
from unittest.mock import patch

import nbformat
import numpy as np
import pandas as pd
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="tcn_context")
    args = parser.parse_args()
    notebook = nbformat.read("sequence_model.ipynb", as_version=4)
    nbformat.validate(notebook)
    ns = {}
    with patch("torch.optim.AdamW.step", side_effect=AssertionError("Training forbidden")):
        with contextlib.redirect_stdout(io.StringIO()):
            for cell in notebook.cells:
                if cell.cell_type == "code":
                    source = cell.source
                    if cell.id == "configuration":
                        source = re.sub(r'^EXPERIMENT = .*$', f'EXPERIMENT = "{args.preset}"', source, flags=re.M)
                    ast.parse(source)
                    exec(compile(source, cell.id, "exec"), ns)
        assert ns["RUN_DIRECTORY"] is None
        cases = ns["train_cases"].iloc[[1350, 0, 2500]].copy()
        d, s, dn, sn = ns["sequence_inputs"](cases)
        unlabeled = cases.drop(columns=ns["LABELS"] + ["total_calculation_time"])
        du, su, _, _ = ns["sequence_inputs"](unlabeled)
        np.testing.assert_array_equal(d, du)
        np.testing.assert_array_equal(s, su)
        # Changing future observed storage cannot change the input for a case.
        one = cases.iloc[[0]]
        before_d, before_s, _, _ = ns["sequence_inputs"](one)
        volume = ns["tables"]["volumes"]
        original = volume.copy()
        try:
            future = pd.date_range(one.starttime.iloc[0] + pd.Timedelta(days=1), periods=7, freq="D")
            volume.loc[future] *= 0.123
            after_d, after_s, _, _ = ns["sequence_inputs"](one)
            np.testing.assert_array_equal(before_d, after_d)
            np.testing.assert_array_equal(before_s, after_s)
        finally:
            ns["tables"]["volumes"] = original
        # Price/inflow coordinates retain exact timestamp meaning.
        times = pd.date_range(one.starttime.iloc[0], periods=168, freq="h")
        np.testing.assert_array_equal(before_d[0, :, dn.index("price")], ns["lookup"]("prices", times)[:, 0])
        for reservoir in ns["RESERVOIRS"]:
            np.testing.assert_allclose(before_s[0, sn.index("initial_volume__"+reservoir)],
                                       original.at[one.starttime.iloc[0], reservoir])
            np.testing.assert_allclose(before_s[0, sn.index("terminal_value__"+reservoir)],
                ns["tables"]["water_values"].at[one.starttime.iloc[0]+pd.Timedelta(hours=168), reservoir])
        if args.preset == "tcn_topology":
            for i, generator in enumerate(ns["GROUP_REPRESENTATIVES"]):
                feeders = ns["FEEDERS"][generator]
                capacity = sum(ns["CAPACITY"][r] for r in feeders)
                levels = {r: np.interp(original.at[one.starttime.iloc[0], r],
                    ns["topology"]["model"]["reservoir"][r]["vol_head"]["x"],
                    ns["topology"]["model"]["reservoir"][r]["vol_head"]["y"]) for r in feeders}
                expected_head = sum(levels[r]*ns["CAPACITY"][r] for r in feeders)/capacity-ns["OUTLET_ELEVATION"][generator]
                np.testing.assert_allclose(before_s[0, sn.index(f"intake_{i}::initial_gross_head_proxy_m")], expected_head, rtol=1e-6)
                upstream = ns["UPSTREAM"][generator]
                natural_flow = ns["lookup"]("inflows", times, upstream).sum(axis=1)
                np.testing.assert_allclose(before_d[0, :, dn.index(f"intake_{i}::upstream_natural_inflow")], natural_flow, rtol=1e-6)
        # Scaler statistics use exactly the fitting array, with constant-column protection.
        scaler = ns["fit_scaler"](d[:2], s[:2], 10)
        np.testing.assert_allclose(scaler["dynamic_mean"], d[:2].mean((0, 1), dtype=np.float64), rtol=1e-6)
        np.testing.assert_allclose(scaler["static_mean"], s[:2].mean(0, dtype=np.float64), rtol=1e-6)
        scaled_d, scaled_s = ns["scale_inputs"](d, s, scaler)
        assert np.isfinite(scaled_d).all() and np.abs(scaled_d).max() <= 10
        assert np.isfinite(scaled_s).all() and np.abs(scaled_s).max() <= 10
        # All 725 test IDs and deliberately distinctive generator/hour probabilities.
        test = ns["test_cases"]
        probability = np.broadcast_to(np.arange(168*14).reshape(1, 168, 14)/(168*14), (len(test), 168, 14))
        frame = ns["sequence_frame"](test, probability)
        assert frame.shape == (725, 2353) and frame.columns.equals(ns["template"].columns)
        np.testing.assert_array_equal(frame["Run No"], ns["template"]["Run No"])
        for j, g in enumerate(ns["GENERATORS"]):
            np.testing.assert_allclose(frame[ns["TARGET_COLUMNS"][g]], probability[:, :, j])
        rng = np.random.default_rng(12)
        p = rng.random((len(cases), 168, 14)).astype(np.float32)
        mapped = ns["sequence_frame"](cases, p)
        metric = {}
        exec(Path("kaggle_metric.py").read_text(), metric)
        auc, _, _ = ns["evaluate"](cases, {g: p[:, :, j] for j, g in enumerate(ns["GENERATORS"])})
        assert np.isclose(auc, metric["score"](cases[["Run No"]+ns["LABELS"]], mapped), atol=1e-12)
        # Architecture/gradient shape without any optimizer step or parameter update.
        ns["initialize_torch"]()
        model = ns["make_model"](ns["BASE_PARAMETERS"])
        model.eval()
        logits = model(torch.from_numpy(scaled_d[:2]), torch.from_numpy(scaled_s[:2]))
        assert logits.shape == (2, 168, 14) and torch.isfinite(logits).all()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.from_numpy(ns["sequence_labels"](cases.iloc[:2])))
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        config = ns["frozen_configuration"]([{"selected_epochs": 7}, {"selected_epochs": 11}])
        ns["validate_frozen"](config)
        if args.preset == "tcn_context":
            old = Path("outputs/tcn/20261013T230426_238589Z_validate_tcn_context/selected_configuration.json")
            if old.exists():
                ns["validate_frozen"](json.loads(old.read_text()))
        config["model_family"] = "catboost"
        try:
            ns["validate_frozen"](config)
        except ValueError:
            pass
        else:
            raise AssertionError("Wrong model family accepted")
        assert 1 + 4*sum(ns["BASE_PARAMETERS"]["dilations"]) >= 2*168-1
    print("PASS: UTC alignment, future-volume/label invariance, case isolation, scaler, 725-case mapping,")
    print("official metric, architecture/gradient shape, frozen guards and full-week receptive field. No parameters fitted.")


if __name__ == "__main__":
    main()
