"""Compare fixed blends of saved development predictions without training."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def metadata(directory):
    result = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    if result.get("mode") != "validate" or result.get("status") != "complete":
        raise ValueError("Only completed development validation runs are supported.")
    return result


def predictions(directory, year, labels, ids):
    frame = pd.read_csv(directory / f"fold_{year}/validation_predictions.csv").set_index("Run No")
    if not frame.index.is_unique or frame.index.hasnans or set(frame.index) != set(ids):
        raise ValueError("Prediction IDs do not match the complete development fold.")
    if frame.columns.tolist() != labels:
        raise ValueError("Prediction columns do not match the submission template.")
    values = frame.loc[ids].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Predictions must be finite probabilities in [0, 1].")
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidates", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = metadata(args.reference)
    candidates = [(path, metadata(path)) for path in args.candidates]
    if not reference.get("input_sha256"):
        raise ValueError("Input hashes are required for comparable runs.")
    for _, candidate in candidates:
        if candidate.get("input_sha256") != reference["input_sha256"] or candidate.get("folds") != reference.get("folds"):
            raise ValueError("Compared runs must use the same input hashes and development folds.")
    label_path = Path("data/kernel/Unit_commitment_decisions.csv")
    if hashlib.sha256(label_path.read_bytes()).hexdigest() != reference["input_sha256"]["labels"]:
        raise ValueError("Current labels differ from the recorded validation labels.")
    truth = pd.read_csv(label_path).set_index("Run No")
    if not truth.index.is_unique or truth.index.hasnans:
        raise ValueError("Label IDs must be unique and nonmissing.")
    truth["starttime"] = pd.to_datetime(truth["starttime"], utc=True)
    labels = pd.read_csv("sample_submission.csv", nrows=0).columns.drop("Run No").tolist()
    rows = []
    for fold in reference["development_results"]:
        start, end = (pd.Timestamp(fold[key], tz="UTC") for key in
                      ("validation_start", "validation_end_exclusive"))
        if end > pd.Timestamp("2022-01-01", tz="UTC"):
            raise ValueError("Blend exploration must not inspect the reserved holdout.")
        year = str(start.year)
        ids = truth.index[(truth.starttime >= start) &
                          (truth.starttime + pd.Timedelta(hours=168) <= end)]
        actual = truth.loc[ids, labels].to_numpy(dtype=np.uint8)
        if not np.isin(actual, [0, 1]).all():
            raise ValueError("Expected binary labels.")
        base = predictions(args.reference, year, labels, ids)
        base_auc = float(roc_auc_score(actual, base, average="micro"))
        if not np.isclose(base_auc, fold["micro_auc"], atol=1e-8, rtol=0):
            raise ValueError("Reference score does not match the saved metric.")
        for path, candidate in candidates:
            other = predictions(path, year, labels, ids)
            other_auc = float(roc_auc_score(actual, other, average="micro"))
            saved = next(result for result in candidate["development_results"]
                         if result["validation_start"] == fold["validation_start"])
            if not np.isclose(other_auc, saved["micro_auc"], atol=1e-8, rtol=0):
                raise ValueError("Candidate score does not match the saved metric.")
            for weight in (0.25, 0.5, 0.75):
                blended = (1 - weight) * base + weight * other
                auc = float(roc_auc_score(actual, blended, average="micro"))
                rows.append({"reference": args.reference.name, "candidate": path.name,
                             "year": year, "candidate_weight": weight, "micro_auc": auc,
                             "reference_auc": base_auc, "candidate_auc": other_auc,
                             "delta_vs_reference": auc - base_auc})
    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    table = result.pivot(index=["candidate", "candidate_weight"], columns="year", values="micro_auc")
    print(table.to_string(float_format=lambda value: f"{value:.8f}"))
    print("Fixed coarse weights evaluated on development years only. These are exploratory results,")
    print("not independent confirmation or a Kaggle score. No training or submission generation.")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
