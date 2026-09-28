"""Seasonal frequency baseline: python predict.py

Estimates ON probabilities for each generator and horizon hour, grouped by
the scheduling case's start month. This is an empirical frequency lookup model;
all cases starting in the same month receive the same predictions.
It does not yet use electricity prices, inflows, storage levels, or water values.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
HORIZON = pd.Timedelta(hours=168)


def fit(train, labels):
    """Estimate one ON probability per month for each of the 2352 targets."""
    return train[labels].groupby(train["starttime"].dt.month).mean()


def predict(model, cases):
    months = cases["starttime"].dt.month
    missing = set(months) - set(model.index)
    if missing:
        raise ValueError(f"Missing training data for months {sorted(missing)}")
    return model.loc[months].to_numpy()


def micro_auc(actual, predicted):
    """Compute micro ROC-AUC using rank sums, assigning average ranks to ties."""
    actual = np.asarray(actual).ravel()
    ranks = pd.Series(np.asarray(predicted).ravel()).rank(method="average")
    positive = actual == 1
    n_positive = int(positive.sum())
    n_negative = actual.size - n_positive
    if not n_positive or not n_negative:
        raise ValueError("ROC-AUC requires both ON and OFF labels.")
    return float(
        (ranks[positive].sum() - n_positive * (n_positive + 1) / 2)
        / (n_positive * n_negative)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs" / "submission.csv"
    )
    args = parser.parse_args()

    train = pd.read_csv(
        ROOT / "data/kernel/Unit_commitment_decisions.csv",
        parse_dates=["starttime"],
    )
    test = pd.read_csv(ROOT / "prediction_mapping.csv", parse_dates=["starttime"])
    template = pd.read_csv(ROOT / "sample_submission.csv")
    labels = template.columns.drop("Run No").tolist()
    if len(labels) != 14 * 168 or not set(labels).issubset(train.columns):
        raise ValueError("Expected 2352 prediction columns present in the training labels.")
    if not train[labels].isin([0, 1]).all().all():
        raise ValueError("Training labels must contain only 0 and 1, with no missing values.")
    if test["Run No"].duplicated().any() or template["Run No"].duplicated().any():
        raise ValueError("Test cases must have unique Run No values.")
    if set(test["Run No"]) != set(template["Run No"]):
        raise ValueError("Test mapping and submission template have different Run No values.")
    test = test.set_index("Run No").loc[template["Run No"]].reset_index()

    # Validate on 2022, excluding training horizons that overlap validation.
    split = pd.Timestamp("2022-01-01")
    fitting = train.loc[train["starttime"] + HORIZON <= split]
    validation = train.loc[train["starttime"] >= split]
    validation_predictions = predict(fit(fitting, labels), validation)
    auc = micro_auc(validation[labels].to_numpy(), validation_predictions)
    print(f"Validation: {len(fitting)} training cases, {len(validation)} validation cases")
    print(f"Micro ROC-AUC on 2022: {auc:.6f}")

    # After validation, refit using all available training cases.
    predictions = predict(fit(train, labels), test)
    if not np.isfinite(predictions).all() or not (
        (predictions >= 0) & (predictions <= 1)
    ).all():
        raise ValueError("Predictions must be finite probabilities between 0 and 1.")
    submission = pd.DataFrame(predictions, columns=labels)
    submission.insert(0, "Run No", test["Run No"].to_numpy())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    print(f"Saved {len(submission)} cases with {len(labels)} predictions each:")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
