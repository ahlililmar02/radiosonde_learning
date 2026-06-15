"""
build_calib.py
────────────────
One-time (re-run only if combined_codes_train.csv changes) script that
computes CALIB — the percentile-rank reference arrays for the rule-engine
trace in draft/rule_engine.py — from the TRAIN flights only, exactly as
classifier.ipynb does, and saves it to draft/models_v2/calib.joblib.

This does NOT train or touch the RandomForest model in
draft/models_v2/secondary_classifier_model.joblib — it only persists the
calibration data needed to display percentile ranks for new flights.

Usage:
  uv run python draft/build_calib.py
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from draft.rule_engine import compute_raw_signals


def main():
    rason_df = pd.read_csv('data/rason_complete.csv', low_memory=False)
    train_labels = pd.read_csv('combined_codes_train.csv')
    train_ids = set(train_labels['flight_id'])

    rows = []
    for fid, group in rason_df.groupby('flight_id'):
        if fid not in train_ids:
            continue
        rows.append(compute_raw_signals(group))

    calib_df = pd.DataFrame(rows)
    calib = {
        col: np.sort(calib_df[col].dropna().values)
        for col in calib_df.columns
        if col != 'gps_frozen'
    }

    out_dir = Path('draft/models_v2')
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(calib, out_dir / 'calib.joblib')

    print(f"Calibration sample size (train flights): {len(calib_df)}")
    print("\nPercentile-rank reference table (value at p10/p25/p50/p75/p90):")
    for col, arr in calib.items():
        p = np.percentile(arr, [10, 25, 50, 75, 90])
        print(f"  {col:25s} {np.round(p, 2)}")

    print(f"\nSaved CALIB to {out_dir / 'calib.joblib'}")


if __name__ == '__main__':
    main()
