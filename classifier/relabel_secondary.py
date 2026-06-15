"""
relabel_secondary.py
─────────────────────
Regenerates combined_codes_train.csv / combined_codes_test.csv using the
CAPE-based Deep Convection rule in classifier/rule_engine.py (replacing the old
peak_ascent-percentile rule from classifier.ipynb).

Reuses the EXISTING train/test flight_id split from the current
combined_codes_train.csv / combined_codes_test.csv — only the secondary-cause
labelling logic changes, not the split. CALIB (classifier/models/calib.joblib)
must already include cape_jkg (run `uv run python -m classifier.build_calib` first
if it doesn't).

Usage:
  uv run python -m classifier.relabel_secondary
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from classifier.rule_engine import build_rule_trace


REASON_MAP = {
    0: "Not Specified",
    1: "Balloon Burst",
    2: "Battery Exhausted",
    3: "Ascent Stop",
    4: "Telemetry Interrupted",
    5: "Manual Termination",
    6: "Other",
    11: "Temperature KO",
}

PRIMARY_CODE_MAP = {
    "Balloon Burst":           ("BUR", "100"),
    "Ascent Stop":             ("ASC", "UNK"),
    "Telemetry Interrupted":   ("TEL", "UNK"),
    "Temperature KO":          ("TMP", "UNK"),
    "Battery Exhausted":       ("BAT", "UNK"),
    "Other":                   ("OTH", "UNK"),
    "Not Specified":           ("UNK", "UNK"),
    "Manual Termination":      ("MAN", "UNK"),
}

_SECONDARY_NAME_FIX = {
    "Cloud Layer": "Cloud/Rain Layer",
}

PREMATURE_THRESHOLD_HPA = 30


def classify_secondary_multi(flight_df: pd.DataFrame, calib: dict) -> dict:
    """Same shape as classifier.ipynb's classify_secondary_multi, but derived
    from rule_engine.build_rule_trace so the Deep Convection rule (now
    CAPE-based) stays in sync with the single source of truth."""
    trace = build_rule_trace(flight_df, calib)
    triggered = {t["rule"]: t["score"] for t in trace if t["triggered"]}
    return dict(sorted(triggered.items(), key=lambda x: x[1], reverse=True))


def build_combined_code(primary_label: str, triggered: dict, valid_combos: dict, threshold: float = 0.4) -> dict:
    pre, suf = PRIMARY_CODE_MAP.get(primary_label, ("UNK", "UNK"))
    valid_secondaries = valid_combos.get(primary_label, {})

    candidates = {
        secondary: score for secondary, score in triggered.items()
        if score >= threshold and secondary in valid_secondaries
    }

    if candidates:
        best = max(candidates, key=candidates.get)
        return {
            "secondary":     best,
            "combined_code": valid_secondaries[best],
            "confidence":    candidates[best],
            "valid":         True,
        }

    return {
        "secondary":     "Unknown",
        "combined_code": f"{pre}-UNK-{suf}",
        "confidence":    0.0,
        "valid":         False,
    }


def label_flights(rason_df: pd.DataFrame, flight_ids: set, calib: dict, valid_combos: dict) -> pd.DataFrame:
    rason_df = rason_df.copy()
    rason_df["launch_time"] = pd.to_datetime(rason_df["launch_time"], errors="coerce")
    rason_df["primary_label"] = (
        rason_df["reason_for_termination"]
        .astype(float)
        .map(lambda x: REASON_MAP.get(int(x), "Not Specified") if pd.notna(x) else "Not Specified")
    )

    results = []
    for fid, group in rason_df.groupby("flight_id"):
        if fid not in flight_ids:
            continue

        primary_label = group["primary_label"].iloc[0]
        triggered     = classify_secondary_multi(group, calib)
        code          = build_combined_code(primary_label, triggered, valid_combos)

        burst_pres = group["pressure_hPa"].min()

        results.append({
            "flight_id":      fid,
            "primary":        primary_label,
            "secondary":      code["secondary"],
            "combined_code":  code["combined_code"],
            "confidence":     code["confidence"],
            "valid":          code["valid"],
            "is_premature":   bool(burst_pres > PREMATURE_THRESHOLD_HPA),
            "burst_pres":     burst_pres,
            "launch_time":    group["launch_time"].iloc[0],
            "burst_height_m": group["height_m"].max(),
        })

    return pd.DataFrame(results)


def main():
    rason_df = pd.read_csv("data/rason_complete.csv", low_memory=False)

    old_train = pd.read_csv("combined_codes_train.csv")
    old_test  = pd.read_csv("combined_codes_test.csv")
    train_ids = set(old_train["flight_id"])
    test_ids  = set(old_test["flight_id"])

    calib = joblib.load("classifier/models/calib.joblib")
    if "cape_jkg" not in calib:
        raise RuntimeError("calib.joblib has no cape_jkg — run `uv run python -m classifier.build_calib` first.")

    label_table = pd.read_excel("Label_End_of_radiosonde.xlsx").iloc[:23]
    valid_combos: dict[str, dict[str, str]] = {}
    for _, row in label_table.iterrows():
        primary   = row["Primary Reason"]
        secondary = _SECONDARY_NAME_FIX.get(row["Secondary"], row["Secondary"])
        code      = row["Combined Code"]
        valid_combos.setdefault(primary, {})[secondary] = code

    new_train = label_flights(rason_df, train_ids, calib, valid_combos)
    new_test  = label_flights(rason_df, test_ids, calib, valid_combos)

    new_train.to_csv("combined_codes_train.csv", index=False)
    new_test.to_csv("combined_codes_test.csv", index=False)

    # COMBINED_CODE_MAP: secondary cause -> [(combined_code, count), ...],
    # most frequent first. Computed from (is_premature & valid) rows so it
    # stays in sync whenever this script is rerun on fresh training data.
    both_new_for_map = pd.concat([new_train, new_test])
    valid_new = both_new_for_map[both_new_for_map["is_premature"] & both_new_for_map["valid"]]
    combined_code_map: dict[str, list[tuple[str, int]]] = {}
    for secondary, group in valid_new.groupby("secondary"):
        counts = group["combined_code"].value_counts()
        combined_code_map[secondary] = list(zip(counts.index, counts.values.tolist()))

    joblib.dump(combined_code_map, "classifier/models/combined_code_map.joblib")
    print(f"\nSaved COMBINED_CODE_MAP ({len(combined_code_map)} causes) to classifier/models/combined_code_map.joblib")

    print(f"Train: {len(new_train)} flights, {new_train['valid'].mean():.1%} valid, "
          f"avg confidence {new_train['confidence'].mean():.2f}")
    print(f"Test:  {len(new_test)} flights, {new_test['valid'].mean():.1%} valid, "
          f"avg confidence {new_test['confidence'].mean():.2f}")

    print("\n--- Secondary distribution (is_premature & valid), OLD vs NEW ---")
    both_old = pd.concat([old_train, old_test])
    both_new = pd.concat([new_train, new_test])
    old_dist = both_old[both_old["is_premature"] & both_old["valid"]]["secondary"].value_counts()
    new_dist = both_new[both_new["is_premature"] & both_new["valid"]]["secondary"].value_counts()
    print(pd.DataFrame({"old": old_dist, "new": new_dist}).fillna(0).astype(int))


if __name__ == "__main__":
    main()
