"""Run the LightGBM notebook's input/LP checks while blocking classifier fitting."""
import contextlib
import copy
import importlib.util
import io
from pathlib import Path

import nbformat
import lightgbm as lgb
import numpy as np


def forbidden(*args, **kwargs):
    raise AssertionError("Classifier fitting/pretrained loading is forbidden in this check.")


def main():
    notebook = Path("analysis/lightgbm_hydraulic_20261101.ipynb")
    document = nbformat.read(notebook, as_version=4)
    lgb.train = forbidden
    lgb.LGBMClassifier.fit = forbidden
    original = lgb.Booster.__init__

    def guarded(self, *args, **kwargs):
        if kwargs.get("model_file") is not None or kwargs.get("model_str") is not None:
            forbidden()
        original(self, *args, **kwargs)

    lgb.Booster.__init__ = guarded
    namespace = {"__name__": "lightgbm_nonfitting_check"}
    with contextlib.redirect_stdout(io.StringIO()):
        for cell in document.cells[:-1]:
            if cell.cell_type == "code":
                exec(compile(cell.source, str(notebook), "exec"), namespace)
    folder = Path("outputs/lightgbm_development/20261101T141843Z_hydraulic/checks")
    folder.mkdir(parents=True, exist_ok=True)
    result = namespace["audit_features"](folder)
    configuration = namespace.get("SELECTED_CONFIGURATION")
    if configuration is not None:
        namespace["guard_selected"](configuration)
        changed = copy.deepcopy(configuration)
        generator = namespace["GENERATORS"][0]
        changed["iterations"][generator] += 1
        try:
            namespace["guard_selected"](changed)
        except ValueError:
            pass
        else:
            raise AssertionError("A changed frozen tree budget was accepted.")
        original_step = namespace["LP_STEP"]
        namespace["LP_STEP"] = 12
        try:
            try:
                namespace["guard_selected"](configuration)
            except ValueError:
                pass
            else:
                raise AssertionError("A changed LP implementation setting was accepted.")
        finally:
            namespace["LP_STEP"] = original_step
        spec = importlib.util.spec_from_file_location("course_metric_reference", Path("kaggle_metric.py"))
        metric = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(metric)
        cases = namespace["train_cases"].iloc[[0, 365, 1826, 2557]]
        rng = np.random.default_rng(17)
        predictions = {g: rng.random((len(cases), namespace["HOURS"])).astype(np.float32)
                       for g in namespace["GENERATORS"]}
        expected = metric.score(cases[["Run No", *namespace["LABELS"]]], namespace["assemble"](cases, predictions))
        assert namespace["official_score"](cases, predictions) == expected
        result["frozen_guard_rejects_tree_and_lp_changes"] = True
        result["inline_metric_matches_course_reference"] = True
    namespace["write_json"](folder / "retained_check.json", result)
    print(result)


if __name__ == "__main__":
    main()
