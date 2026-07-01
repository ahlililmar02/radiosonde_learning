"""
secondary_classifier.py
────────────────────────
Inference pipeline for the recalibrated secondary burst-cause taxonomy
(combined_codes_train.csv / combined_codes_test.csv from classifier.ipynb,
masked against Label_End_of_radiosonde.xlsx).

This is a NEW, self-contained pipeline — it does not modify or import
classifier/final_classifier.py. BUFR parsing / cleaning / 30 hPa threshold logic
is verbatim from final_classifier.py so this file can be deployed on its own.
engineer_features / fill_missing_features live in classifier/features.py and are
imported by both this file and train_secondary_model.py — single source of truth.

Steps:
  1. Parse BUFR file -> pressure-level DataFrame
  2. Check 30 hPa threshold -> nominal or premature
  3. Feature engineering (same as classifier/train_secondary_model.py)
  4. Load classifier/models/ model / scaler / label encoder
  5. Predict secondary cause + confidence
  6. Print labelled output with explanation

Usage:
  uv run python classifier/secondary_classifier.py --file data/myfile.bin
  uv run python classifier/secondary_classifier.py --file data/myfile.bin --model_dir classifier/models
  uv run python classifier/secondary_classifier.py --file data/myfile.bin --save_json output.json
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import eccodes
import joblib
import numpy as np
import pandas as pd
import shap

from classifier.rule_engine import build_rule_trace
from classifier.features import engineer_features, fill_missing_features


# ══════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

PREMATURE_THRESHOLD_HPA = 30  # WMO standard

CAUSE_DESCRIPTIONS = {
    'RH Freeze-out': (
        "Burst occurred at a very cold temperature with low dewpoint "
        "depression — moisture on the balloon membrane likely froze out "
        "rapidly near burst, embrittling the latex and causing failure."
    ),
    'RH Saturation': (
        "The flight spent much of its ascent in near-saturated air "
        "(small dewpoint depression throughout). Persistent moisture "
        "loading and icing on the membrane is the likely stressor."
    ),
    'Cloud/Rain Layer': (
        "A deep, near-saturated moist layer was crossed during ascent, "
        "coinciding with a drop in ascent rate. The balloon likely picked "
        "up liquid water / ice mass passing through a cloud or rain layer, "
        "adding weight and aerodynamic drag until it failed."
    ),
    'Strong Shear': (
        "Wind speed or directional shear was unusually high for this "
        "flight. Strong mechanical stress from rapid changes in wind "
        "speed across a layer likely tore or deformed the balloon."
    ),
    'Directional Shear': (
        "A large change in wind direction occurred without a "
        "corresponding increase in wind speed. The balloon was twisted "
        "or sheared directionally, stressing the membrane and payload "
        "train."
    ),
    'Deep Convection': (
        "An extreme spike in ascent rate was detected near burst, "
        "consistent with the balloon being caught in a strong convective "
        "updraft. The rapid acceleration overstressed the membrane."
    ),
    'Slow Ascent': (
        "The flight's ascent rate was unusually slow for most of its "
        "duration (likely under-inflation). Prolonged exposure time "
        "increased solar heating / cooling cycles and UV degradation of "
        "the membrane before it eventually failed."
    ),
    'Cold Point Tropopause': (
        "Burst occurred close to the cold-point tropopause altitude. "
        "The extreme low temperatures and structural transition near the "
        "tropopause are the likely cause of membrane failure."
    ),
    'Pressure Reversal': (
        "The pressure record shows the balloon descending or oscillating "
        "before termination, rather than a clean monotonic ascent to "
        "burst. This points to a partial failure (slow leak, tangled "
        "train) followed by descent rather than a single sharp burst."
    ),
    'GPS Fail': (
        "Latitude/longitude telemetry froze or went missing near the end "
        "of the flight. This points to a GPS / telemetry hardware fault "
        "rather than a purely atmospheric burst cause."
    ),
    'Unknown': (
        "The flight is premature, but its feature profile does not match "
        "any of the known secondary-cause patterns with confidence. "
        "Manual review of the sounding is recommended."
    ),
    'nominal': (
        "The balloon reached or exceeded the 30 hPa target ceiling. "
        "This is a successful flight — no premature burst detected."
    ),
}


# ══════════════════════════════════════════════════════════════════════════════
# 2. BUFR PARSER (verbatim from classifier/final_classifier.py)
# ══════════════════════════════════════════════════════════════════════════════

def extract_radiosonde_data(file_path: str) -> pd.DataFrame:
    """
    Parse a BUFR radiosonde file into a pressure-level DataFrame.
    Returns an empty DataFrame if parsing fails.
    """

    with open(file_path, 'rb') as f:
        msgid = eccodes.codes_bufr_new_from_file(f)
        if msgid is None:
            return pd.DataFrame()

        eccodes.codes_set(msgid, 'unpack', 1)

        profile_keys = {
            'pressure': 'pressure_Pa',
            'airTemperature': 'temp_K',
            'dewpointTemperature': 'dewpoint_K',
            'windDirection': 'wind_dir_deg',
            'windSpeed': 'wind_speed_mps',
            'nonCoordinateGeopotentialHeight': 'height_m',
        }

        launch_time = None
        try:
            year   = eccodes.codes_get(msgid, 'year')
            month  = eccodes.codes_get(msgid, 'month')
            day    = eccodes.codes_get(msgid, 'day')
            hour   = eccodes.codes_get(msgid, 'hour')
            minute = eccodes.codes_get(msgid, 'minute')
            launch_time = datetime(year, month, day, hour, minute)
        except Exception:
            pass

        data = {}
        lengths = []

        for bufr_key, df_name in profile_keys.items():
            try:
                val = eccodes.codes_get_array(msgid, bufr_key)
                data[df_name] = val
                lengths.append(len(val))
            except eccodes.KeyValueNotFoundError:
                continue

        if not lengths:
            eccodes.codes_release(msgid)
            return pd.DataFrame()

        max_len = max(lengths)

        clean_data = {}

        for k, v in data.items():
            v = np.asarray(v, dtype=float)

            if len(v) < max_len:
                v = np.pad(v, (0, max_len - len(v)), constant_values=np.nan)
            elif len(v) > max_len:
                v = v[:max_len]

            clean_data[k] = v

        station_id = None

        try:
            wigos = eccodes.codes_get(msgid, 'wigosStationIdentifier')
            if wigos not in [None, '', 'missing']:
                station_id = int(str(wigos).split('-')[-1])
        except Exception:
            pass

        if station_id is None:
            try:
                block = eccodes.codes_get(msgid, 'blockNumber')
                station = eccodes.codes_get(msgid, 'stationNumber')
                if block is not None and station is not None:
                    station_id = int(f"{int(block):02d}{int(station):03d}")
            except Exception:
                pass

        eccodes.codes_release(msgid)

        df = pd.DataFrame(clean_data)

        if 'temp_K' in df.columns:
            df['temp_C'] = df['temp_K'] - 273.15

        if 'dewpoint_K' in df.columns:
            df['dewpoint_C'] = df['dewpoint_K'] - 273.15

        if 'pressure_Pa' in df.columns:
            df['pressure_hPa'] = df['pressure_Pa'] / 100.0

        df['launch_time'] = launch_time
        df['station_id'] = station_id
        df['source_file'] = Path(file_path).name

        return df


# ══════════════════════════════════════════════════════════════════════════════
# 3. PREMATURE BURST DETECTION (verbatim from classifier/final_classifier.py)
# ══════════════════════════════════════════════════════════════════════════════

def detect_premature_burst(df: pd.DataFrame) -> dict:
    df['pressure_hPa'] = pd.to_numeric(df['pressure_hPa'], errors='coerce')
    df['pressure_hPa'] = df['pressure_hPa'].replace(
        [-9999, -999, 9999, 99999, 2e20], np.nan
    )

    valid = df['pressure_hPa'].dropna()
    if valid.empty:
        return {
            'is_premature'    : None,
            'burst_pres_hpa'  : None,
            'burst_alt_m'     : None,
            'n_levels'        : 0,
            'error'           : 'no valid pressure values',
        }

    burst_pres = float(valid.min())
    burst_alt  = float(df['height_m'].max()) if 'height_m' in df.columns else None

    return {
        'is_premature'   : burst_pres > PREMATURE_THRESHOLD_HPA,
        'burst_pres_hpa' : round(burst_pres, 2),
        'burst_alt_m'    : round(burst_alt, 1) if burst_alt is not None else None,
        'n_levels'       : len(df),
        'threshold_hpa'  : PREMATURE_THRESHOLD_HPA,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. DATA CLEANING (verbatim from classifier/final_classifier.py)
# ══════════════════════════════════════════════════════════════════════════════

def clean_profile(df: pd.DataFrame) -> pd.DataFrame:
    MISSING = [-9999, -999, 9999, 99999, 2e20, 1e20]

    num_cols = ['temp_C', 'dewpoint_C', 'wind_speed_mps',
                'height_m', 'pressure_hPa']

    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].replace(MISSING, np.nan)
            df.loc[df[col].abs() > 1e10, col] = np.nan

    if 'ascent_rate_mps' not in df.columns:
        df = df.sort_values('pressure_hPa', ascending=False).reset_index(drop=True)
        df['ascent_rate_mps'] = df['height_m'].diff().abs() / 1.0

    df = df.sort_values('pressure_hPa', ascending=False).reset_index(drop=True)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 5b. LABEL CODES (Label_End_of_radiosonde.xlsx, see classifier.ipynb cell 3/5)
# ══════════════════════════════════════════════════════════════════════════════

# 3-letter code for each secondary cause (classifier.ipynb SECONDARY_CODE_MAP).
SECONDARY_CODE_MAP = {
    'RH Freeze-out':         'FRZ',
    'Cloud/Rain Layer':      'CLD',
    'Deep Convection':       'CNB',
    'Strong Shear':          'SHR',
    'Directional Shear':     'DSH',
    'Pressure Reversal':     'PRS',
    'Slow Ascent':           'ASN',
    'RH Saturation':         'SAT',
    'Cold Point Tropopause': 'CPT',
    'GPS Fail':              'GPS',
}

# Combined codes (Primary-Secondary-Suffix) from Label_End_of_radiosonde.xlsx,
# restricted to the (is_premature & valid) flights this model was trained on
# (combined_codes_train.csv + combined_codes_test.csv). "Primary" (Balloon
# Burst / Ascent Stop / etc.) is ground-truth termination metadata that isn't
# derivable from the BUFR profile, so a predicted secondary cause can map to
# more than one combined code — listed here as (code, training count), most
# frequent first.
#
# Generated by classifier/relabel_secondary.py -> classifier/models/combined_code_map.joblib
# every time the training labels are regenerated, so it stays in sync with
# monthly retrains. Hardcoded values below are only a fallback for older
# model bundles that predate the generated artifact.
COMBINED_CODE_MAP_FALLBACK = {
    'RH Freeze-out':         [('BUR-FRZ-100', 60), ('ASC-FRZ-UNK', 10)],
    'Cloud/Rain Layer':      [('BUR-CLD-100', 25)],
    'Deep Convection':       [('BUR-CNB-100', 25), ('ASC-CNB-UNK', 5)],
    'Strong Shear':          [('BUR-SHR-100', 111), ('ASC-SHR-UNK', 46), ('UNK-SHR-UNK', 5)],
    'Pressure Reversal':     [('BUR-PRS-UNK', 35)],
    'Slow Ascent':           [('ASC-ASN-UNK', 33)],
    'Cold Point Tropopause': [('TMP-CPT-UNK', 34)],
}


# ══════════════════════════════════════════════════════════════════════════════
# 6. EXPLANATION GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

# Feature thresholds for evidence extraction — informed by the
# classify_secondary_multi() rule definitions in classifier.ipynb,
# translated to the engineer_features() feature set.
EVIDENCE_THRESHOLDS = {
    'RH Freeze-out': {
        'temp_at_burst_C'     : ('low',   -40,  'burst temperature was extremely cold ({:.1f}°C)'),
        'moist_dd_mean'       : ('low',   3.0,  'dewpoint depression near burst was small ({:.1f}°C) — near-saturated'),
    },
    'RH Saturation': {
        'moist_dd_mean'       : ('low',   3.0,  'dewpoint depression averaged only {:.1f}°C — column was near-saturated'),
        'pw_proxy'            : ('high',  3e5,  'column moisture proxy was high ({:.0f}) — humid profile throughout'),
    },
    'Cloud/Rain Layer': {
        'moist_depth_m'       : ('high',  1000, 'a moist/cloud layer {:.0f} m deep was crossed'),
        'moist_index'         : ('high',  3e4,  'moisture loading index was elevated ({:.0f})'),
    },
    'Strong Shear': {
        # max_shear thresholds calibrated from train-set percentiles
        # (classifier/models/calib.joblib): p75=7.5, p50=5.96 m/s per 100m.
        # Old fixed values (0.04 / 0.02) were ~100x too low and triggered
        # on virtually every flight.
        'max_shear'           : ('high',  7.5,  'maximum wind shear was high ({:.3f} m/s per 100 m, top quartile)'),
        'max_wind_speed_mps'  : ('high',  20,   'maximum wind speed was high ({:.1f} m/s)'),
    },
    'Directional Shear': {
        'max_wind_speed_mps'  : ('low',   15,   'wind speed stayed moderate ({:.1f} m/s) despite the shear event'),
        'max_shear'           : ('high',  6.0,  'wind shear was elevated ({:.3f} m/s per 100 m, above median)'),
    },
    'Deep Convection': {
        'cape_jkg'            : ('high',  1000, 'surface-based CAPE was {:.0f} J/kg — moderate-to-strong convective potential'),
        'lcl_agl_m'           : ('low',   600,  'lifted condensation level was low ({:.0f} m AGL) — easy cloud formation'),
    },
    'Slow Ascent': {
        'ascent_rate_mean'    : ('low',   3.0,  'mean ascent rate was slow ({:.2f} m/s)'),
        'time_to_burst_mins'  : ('high',  100,  'flight duration before burst was long ({:.0f} min)'),
    },
    'Cold Point Tropopause': {
        'cpt_dist'            : ('low',   500,  'burst occurred close to the cold-point tropopause altitude ({:.0f} m away)'),
        'tropopause_temp_C'   : ('low',   -75,  'tropopause temperature was extremely cold ({:.1f}°C)'),
    },
}


def generate_explanation(
    cause: str,
    confidence: float,
    features: dict,
    all_probs: dict,
    burst_info: dict,
) -> str:
    lines = []
    lines.append("=" * 62)
    lines.append("  RADIOSONDE SECONDARY BURST CAUSE CLASSIFICATION (v2)")
    lines.append("=" * 62)
    lines.append(f"  Predicted cause  : {cause}")
    lines.append(f"  Confidence       : {confidence:.1f}%")
    lines.append(f"  Burst pressure   : {burst_info['burst_pres_hpa']} hPa  "
                 f"(threshold: {burst_info['threshold_hpa']} hPa)")
    if burst_info.get('burst_alt_m'):
        lines.append(f"  Burst altitude   : {burst_info['burst_alt_m']:,.0f} m")
    lines.append(f"  Sounding levels  : {burst_info['n_levels']}")
    lines.append(f"  Premature burst  : {'YES' if burst_info['is_premature'] else 'NO (reached nominal ceiling)'}")
    lines.append("-" * 62)

    if not burst_info['is_premature']:
        lines.append("  NOTE: This flight reached the 30 hPa nominal ceiling — it did")
        lines.append("  NOT burst prematurely. The model (trained on premature flights")
        lines.append("  only) is reporting which failure-pattern this flight's")
        lines.append("  end-of-flight profile most closely resembles, not an actual")
        lines.append("  cause of failure.")
        lines.append("-" * 62)

    lines.append("  EVIDENCE FROM SOUNDING PROFILE:")
    lines.append("")
    thresholds = EVIDENCE_THRESHOLDS.get(cause, {})
    found = []
    for feat_name, (direction, threshold, template) in thresholds.items():
        val = features.get(feat_name)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        triggered = (
            (direction == 'high' and val > threshold) or
            (direction == 'low'  and val < threshold)
        )
        if triggered:
            try:
                found.append("  - " + template.format(val))
            except Exception:
                found.append(f"  - {feat_name} = {val:.3f}")

    if found:
        lines.extend(found)
    else:
        lines.append("  - Pattern matched cluster profile from training data")
        lines.append("    (individual thresholds not exceeded but overall")
        lines.append("    feature combination matches this cause)")

    lines.append("")
    lines.append("  CAUSE DESCRIPTION:")
    desc = CAUSE_DESCRIPTIONS.get(cause, "No description available.")
    words = desc.split()
    line_buf, wrapped = [], []
    for w in words:
        if sum(len(x) + 1 for x in line_buf) + len(w) > 56:
            wrapped.append("  " + " ".join(line_buf))
            line_buf = [w]
        else:
            line_buf.append(w)
    if line_buf:
        wrapped.append("  " + " ".join(line_buf))
    lines.extend(wrapped)

    lines.append("")
    lines.append("  ALTERNATIVE CAUSES:")
    sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
    shown = 0
    for c, p in sorted_probs:
        if c == cause:
            continue
        if p >= 5.0:
            lines.append(f"  - {c:<30} {p:.1f}%")
            shown += 1
    if shown == 0:
        lines.append("  - No alternative above 5% confidence")

    lines.append("=" * 62)
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# 7. MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(file_path: str, model_dir: str = "classifier/models") -> dict:

    print(f"\nProcessing: {Path(file_path).name}")
    print("-" * 62)

    print("  [1/6] Parsing BUFR file...")
    df = extract_radiosonde_data(file_path)

    if df.empty:
        print("  ERROR: Could not parse BUFR file or no data found.")
        return {"file": file_path, "error": "parse_failed"}

    print(f"        {len(df)} pressure levels extracted")

    print("  [2/6] Cleaning profile data...")
    df = clean_profile(df)

    print("  [3/6] Checking burst pressure threshold (30 hPa)...")
    burst_info = detect_premature_burst(df)

    if burst_info.get('error'):
        print(f"  ERROR: {burst_info['error']}")
        return {"file": file_path, "error": burst_info['error']}

    print(f"        Burst pressure : {burst_info['burst_pres_hpa']} hPa")
    print(f"        Premature      : {'YES' if burst_info['is_premature'] else 'NO'}")

    # Note: the model is trained only on premature flights (its
    # "secondary cause" labels describe failure signatures near burst),
    # but it is applied here to ALL flights regardless of is_premature —
    # for nominal flights this is the model's best guess at which
    # failure-pattern the end-of-flight profile resembles, not a
    # statement that the flight actually failed. is_premature is
    # returned as a display field so the UI can frame the result
    # accordingly.

    print("  [4/6] Engineering features from vertical profile...")
    features_raw = engineer_features(df)

    if features_raw is None:
        print("  ERROR: Too few valid levels for feature engineering.")
        return {"file": file_path, "error": "insufficient_levels"}

    # Rule-engine trace (analysis overlay only — does not affect the
    # ML prediction). Uses the frozen percentile distributions from
    # classifier/models/calib.joblib (built once from train flights).
    rule_trace = None
    calib_path = Path(model_dir) / "calib.joblib"
    if calib_path.exists():
        calib = joblib.load(calib_path)
        rule_trace = build_rule_trace(df, calib)

    print("  [5/6] Loading saved model and scaler...")
    model_path   = Path(model_dir) / "secondary_classifier_model.joblib"
    scaler_path  = Path(model_dir) / "secondary_classifier_scaler.joblib"
    encoder_path = Path(model_dir) / "secondary_label_encoder.joblib"
    cols_path    = Path(model_dir) / "feature_cols.joblib"

    for p in [model_path, scaler_path, encoder_path, cols_path]:
        if not p.exists():
            print(f"  ERROR: Model file not found: {p}")
            print("         Run classifier/train_secondary_model.py first to generate model files.")
            return {"file": file_path, "error": f"missing_model_file: {p.name}"}

    model        = joblib.load(model_path)
    scaler       = joblib.load(scaler_path)
    le           = joblib.load(encoder_path)
    feature_cols = joblib.load(cols_path)

    features = fill_missing_features(features_raw, feature_cols)

    X = np.array([[features.get(c, 0.0) for c in feature_cols]])
    X_scaled = scaler.transform(X)

    print("  [6/6] Running classifier...")
    pred_enc    = model.predict(X_scaled)[0]
    proba       = model.predict_proba(X_scaled)[0]
    cause       = le.inverse_transform([pred_enc])[0]
    confidence  = float(proba.max()) * 100
    all_probs   = {
        cls: round(float(p) * 100, 1)
        for cls, p in zip(le.classes_, proba)
    }

    explanation = generate_explanation(
        cause, confidence, features, all_probs, burst_info
    )

    # SHAP per-prediction evidence: which features actually drove THIS
    # prediction (vs the fixed EVIDENCE_THRESHOLDS checklist, which only
    # checks 1-2 features per class and can be empty even when that class
    # is predicted). Built fresh per request — TreeExplainer is cheap
    # (<0.1s) for this model size.
    explainer  = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(X_scaled)
    contrib    = shap_vals[0, :, pred_enc]
    top_idx    = np.argsort(-np.abs(contrib))[:5]
    shap_evidence = [
        {
            "feature": feature_cols[i],
            "value": round(float(features[feature_cols[i]]), 4),
            "shap_value": round(float(contrib[i]), 4),
            "direction": "supports" if contrib[i] > 0 else "opposes",
        }
        for i in top_idx
    ]

    # Label codes for the predicted secondary cause (Label_End_of_radiosonde.xlsx).
    # combined_code_map.joblib is regenerated each retrain by relabel_secondary.py;
    # fall back to the frozen snapshot for older model bundles that lack it.
    combined_code_map_path = Path(model_dir) / "combined_code_map.joblib"
    combined_code_map = (
        joblib.load(combined_code_map_path) if combined_code_map_path.exists()
        else COMBINED_CODE_MAP_FALLBACK
    )

    secondary_code = SECONDARY_CODE_MAP.get(cause)
    combos = combined_code_map.get(cause, [])
    total = sum(n for _, n in combos)
    combined_code_candidates = [
        {"code": code, "share_pct": round(100 * n / total, 1)}
        for code, n in combos
    ] if total else []

    result = {
        "file"             : Path(file_path).name,
        "launch_time"      : str(df['launch_time'].iloc[0])
                             if 'launch_time' in df.columns else None,
        "station_id"       : str(df['station_id'].iloc[0])
                             if 'station_id' in df.columns else None,
        "is_premature"     : burst_info['is_premature'],
        "burst_pres_hpa"   : burst_info['burst_pres_hpa'],
        "burst_alt_m"      : burst_info['burst_alt_m'],
        "n_levels"         : burst_info['n_levels'],
        "predicted_cause"  : cause,
        "confidence_pct"   : round(confidence, 1),
        "all_probabilities": all_probs,
        "secondary_code"           : secondary_code,
        "combined_code"            : combined_code_candidates[0]["code"] if combined_code_candidates else None,
        "combined_code_candidates" : combined_code_candidates,
        "key_features"     : [e["feature"] for e in shap_evidence if e["direction"] == "supports"],
        "shap_evidence"    : shap_evidence,
        "features_used"    : {k: round(float(v), 4)
                              for k, v in features.items()
                              if k in feature_cols},
        "rule_trace"       : rule_trace,
        "explanation"      : explanation,
        "processed_at"     : datetime.utcnow().isoformat() + "Z",
    }

    print(explanation)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 8. ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Radiosonde BUFR secondary burst-cause classifier (v2)"
    )
    parser.add_argument("--file", type=str, required=True,
                        help="Path to BUFR radiosonde file")
    parser.add_argument("--model_dir", type=str, default="classifier/models",
                        help="Directory containing saved model files (default: classifier/models/)")
    parser.add_argument("--save_json", type=str, default=None,
                        help="Optional path to save result as JSON e.g. result.json")
    args = parser.parse_args()

    if not Path(args.file).exists():
        print(f"ERROR: File not found: {args.file}")
        sys.exit(1)

    result = run_pipeline(args.file, args.model_dir)

    if args.save_json:
        save_result = {k: v for k, v in result.items() if k != 'features_used'}
        with open(args.save_json, 'w') as f:
            json.dump(save_result, f, indent=2, default=str)
        print(f"\nResult saved to: {args.save_json}")


if __name__ == "__main__":
    main()
