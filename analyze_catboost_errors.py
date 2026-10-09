"""Audit saved development predictions without fitting models or reading test labels."""

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score


def summarize(actual, predicted):
    actual, predicted = actual.ravel(), predicted.ravel()
    return {
        "count": int(actual.size),
        "auc": float(roc_auc_score(actual, predicted)) if np.unique(actual).size == 2 else np.nan,
        "logloss": float(log_loss(actual, predicted, labels=[0, 1])),
        "brier": float(np.mean((actual - predicted) ** 2)),
        "on_fraction": float(actual.mean()),
        "mean_prediction": float(predicted.mean()),
    }


def analyze_run(root, directory):
    metadata = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    if metadata.get("mode") != "validate" or metadata.get("status") != "complete":
        raise ValueError("Only completed development validation runs are supported.")
    label_path = root / "data/kernel/Unit_commitment_decisions.csv"
    expected_hash = metadata.get("input_sha256", {}).get("labels")
    if expected_hash and hashlib.sha256(label_path.read_bytes()).hexdigest() != expected_hash:
        raise ValueError("Raw labels differ from the validation run's recorded inputs.")
    truth = pd.read_csv(label_path).set_index("Run No")
    if not truth.index.is_unique or truth.index.hasnans:
        raise ValueError("Label IDs must be unique and nonmissing.")
    truth["starttime"] = pd.to_datetime(truth["starttime"], utc=True)
    labels = pd.read_csv(root / "sample_submission.csv", nrows=0).columns.drop("Run No").tolist()
    generators = {}
    for column in labels:
        match = re.fullmatch(r"result_committed_(.+)_t(\d+)", column)
        if match is None:
            raise ValueError(f"Unexpected target: {column}")
        generators.setdefault(match[1], {})[int(match[2])] = column
    if len(generators) != 14 or any(set(v) != set(range(168)) for v in generators.values()):
        raise ValueError("Expected 14 complete 168-hour target sequences.")
    rows = []
    for fold in metadata["development_results"]:
        year = fold["validation_start"][:4]
        predictions = pd.read_csv(directory / f"fold_{year}/validation_predictions.csv").set_index("Run No")
        if not predictions.index.is_unique or predictions.index.hasnans:
            raise ValueError("Prediction IDs must be unique and nonmissing.")
        if predictions.columns.tolist() != labels:
            raise ValueError("Saved prediction columns do not match the template.")
        start, end = (pd.Timestamp(fold[k], tz="UTC") for k in
                      ("validation_start", "validation_end_exclusive"))
        if end > pd.Timestamp("2022-01-01", tz="UTC"):
            raise ValueError("This diagnostic must not inspect the reserved 2022 holdout.")
        expected = truth.index[(truth.starttime >= start) &
                               (truth.starttime + pd.Timedelta(hours=168) <= end)]
        if set(predictions.index) != set(expected):
            raise ValueError("Saved IDs do not match the complete purged development fold.")
        actual_frame = truth.loc[predictions.index]
        actual = actual_frame[labels].to_numpy(dtype=np.uint8)
        predicted = predictions.to_numpy(dtype=np.float64)
        if not np.isin(actual, [0, 1]).all() or not np.isfinite(predicted).all() or \
                (predicted < 0).any() or (predicted > 1).any():
            raise ValueError("Invalid labels or probabilities.")

        def add(group, segment, y, p):
            if y.size:
                rows.append({"run": directory.name, "year": year, "group": group,
                             "segment": segment, **summarize(y, p)})

        add("overall", "all", actual, predicted)
        if not np.isclose(rows[-1]["auc"], fold["micro_auc"], atol=1e-8, rtol=0):
            raise ValueError("Recomputed micro AUC differs from the saved score.")
        y_cube = np.stack([actual_frame[[columns[h] for h in range(168)]].to_numpy()
                           for columns in generators.values()], axis=1)
        p_cube = np.stack([predictions[[columns[h] for h in range(168)]].to_numpy()
                           for columns in generators.values()], axis=1)
        for i, generator in enumerate(generators):
            add("generator", generator, y_cube[:, i], p_cube[:, i])
        for day in range(7):
            add("horizon_day", str(day + 1), y_cube[:, :, day * 24:(day + 1) * 24],
                p_cube[:, :, day * 24:(day + 1) * 24])
        for quarter in range(1, 5):
            mask = actual_frame.starttime.dt.quarter.to_numpy() == quarter
            add("case_start_quarter", str(quarter), y_cube[mask], p_cube[mask])
        # These truth-derived segments are diagnostics ONLY, never model inputs.
        changed = y_cube[:, :, 1:] != y_cube[:, :, :-1]
        for segment, mask in (("switch", changed), ("stable", ~changed)):
            add("transition_at_hour", segment, y_cube[:, :, 1:][mask], p_cube[:, :, 1:][mask])
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = pd.concat([analyze_run(Path.cwd(), run) for run in args.runs], ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    for (run, year), group in result.groupby(["run", "year"]):
        print(run, year)
        print(group.loc[group.group == "generator", ["segment", "auc", "brier"]]
              .sort_values("brier", ascending=False).head(5).to_string(index=False))
        print(group.loc[group.group.isin(["overall", "horizon_day", "transition_at_hour"]),
                        ["group", "segment", "auc", "brier"]].to_string(index=False))
    print(f"Saved {args.output}. Descriptive metrics only: overlapping cases are not independent.")


if __name__ == "__main__":
    main()
