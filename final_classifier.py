"""
bufr_burst_inference.py
───────────────────────
Full inference pipeline for a single radiosonde BUFR file.

Steps:
  1. Parse BUFR file → pressure-level DataFrame
  2. Check 30 hPa threshold → nominal or premature
  3. Feature engineering (same as training)
  4. Load saved model / scaler / label encoder
  5. Predict cause + confidence
  6. Print labelled output with explanation

Usage:
  python bufr_burst_inference.py --file data/A_IUSG51WION121200_C_WIIX_20251012120000.bin
  python bufr_burst_inference.py --file data/myfile.bin --model_dir models/
  python bufr_burst_inference.py --file data/myfile.bin --save_json output.json
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import eccodes
import joblib
import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

PREMATURE_THRESHOLD_HPA = 30   # WMO standard

CAUSE_DESCRIPTIONS = {
    'launch_failure': (
        "The balloon burst very early with no atmospheric stressor present. "
        "Burst pressure was extremely high (low altitude). Likely caused by "
        "balloon damage before or at launch, under-inflation, or equipment defect. "
        "Recommend checking ground handling logs for this flight."
    ),
    'deep_moisture_icing': (
        "A deep supercooled liquid water layer was detected during ascent. "
        "The icing layer extended over a significant altitude range with near-saturated "
        "conditions (heavy precipitable water and moisture). Ice accumulation on the balloon membrane "
        "over this extended layer caused progressive stress until burst."
    ),
    'severe_icing': (
        "A highly concentrated supercooled zone was detected. Although the icing layer "
        "was thin event, the icing intensity was the highest "
        "recorded — suggesting a dense supercooled cloud layer. Rapid ice accumulation "
        "caused membrane failure."
    ),
    'high_updraft': (
        "Large ascent rate spikes were detected combined with unstable atmospheric layers"
        "and high wind speeds. The balloon flew through turbulent air with strong updrafts and dynamic instability,"
        " likely associated with deep convection. Mechanical stress from rapid acceleration "
        "and deceleration caused premature burst."
    ),
    'atmospheric_instability': (
        "Dynamic atmospheric instability was detected at high altitude near the "
        "tropopause. Richardson number minimum was found in the upper troposphere, "
        "combined with elevated shear. The balloon survived the lower atmosphere but was "
        "stressed by wind shear (Clear Air Turbulence) near the jet stream or tropopause."
    ),
    'dry_layer': (
        "The atmosphere was anomalously dry (high dewpoint depression) with elevated "
        "upper-level shear and unstable layers. This is consistent with clear-air "
        "turbulence in dry conditions — unusual for Indonesia."
        "Another possibility is that the balloon ascends through a very dry, cold layer, builds up friction, and popped by static electricity."
    ),
    'boundary_layer_shear': (
        "Strong wind shear was detected in the lower layer with rapid changes "
        "in wind speed or direction."
        "The resulting mechanical stress and structural deformation of the latex membrane "
        "caused it to tear well before reaching upper altitudes."
    ),
    'sensor_failure': (
        "Cold point detected at such a low altitude "
        "suggests a sensor failure or data error rather than a physical burst cause. "
        "check the raw data and metadata for this flight for signs of sensor issues or data corruption."
    ),
    'nominal': (
        "The balloon reached or exceeded the 30 hPa target ceiling. "
        "This is a successful flight — no premature burst detected."
    ),
    'unknown_premature': (
        "The flight shows premature burst but no single atmospheric cause "
        "could be identified with confidence. The feature profile does not "
        "match any known burst cause pattern clearly."
    ),
}


# ══════════════════════════════════════════════════════════════════════════════
# 2. BUFR PARSER  (from your existing code)
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
            'pressure'                      : 'pressure_Pa',
            'airTemperature'                : 'temp_K',
            'dewpointTemperature'           : 'dewpoint_K',
            'windDirection'                 : 'wind_dir_deg',
            'windSpeed'                     : 'wind_speed_mps',
            'nonCoordinateGeopotentialHeight': 'height_m',
        }

        # Try to get launch time from BUFR header
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

        # Try to get station info
        station_id = None
        try:
            station_id = eccodes.codes_get(msgid, 'stationNumber')
        except Exception:
            try:
                station_id = eccodes.codes_get(msgid, 'stationOrSiteName')
            except Exception:
                pass

        data    = {}
        lengths = []

        for bufr_key, df_name in profile_keys.items():
            try:
                val = eccodes.codes_get_array(msgid, bufr_key)
                data[df_name] = val
                lengths.append(len(val))
            except eccodes.KeyValueNotFoundError:
                continue

        eccodes.codes_release(msgid)

        if not lengths:
            return pd.DataFrame()

        max_len = max(lengths)

        import numpy as np

        clean_data = {}

        
        for k, v in data.items():
            v = np.asarray(v, dtype=float)  # ✅ FORCE FLOAT HERE

            if len(v) < max_len:
                print(f"  [pad] {k}: {len(v)} → {max_len}")
                v = np.pad(v, (0, max_len - len(v)), constant_values=np.nan)
            elif len(v) > max_len:
                print(f"  [trim] {k}: {len(v)} → {max_len}")
                v = v[:max_len]

            clean_data[k] = v

        df = pd.DataFrame(clean_data)
        print("\n=== HEAD ===")
        print(df.head(10))
        print("\n=== DATAFRAME INFO ===")
        print("Columns:", df.columns.tolist())
        print("Shape:", df.shape)
        
        # Unit conversions
        if 'temp_K'      in df.columns:
            df['temp_C']      = df['temp_K']      - 273.15
        if 'dewpoint_K'  in df.columns:
            df['dewpoint_C']  = df['dewpoint_K']  - 273.15
        if 'pressure_Pa' in df.columns:
            df['pressure_hPa'] = df['pressure_Pa'] / 100.0

        # Add metadata columns
        df['launch_time'] = launch_time
        df['station_id']  = station_id
        df['source_file'] = Path(file_path).name

        return df


# ══════════════════════════════════════════════════════════════════════════════
# 3. PREMATURE BURST DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_premature_burst(df: pd.DataFrame) -> dict:
    """
    Apply the 30 hPa WMO threshold.
    Returns a dict with burst info and is_premature flag.
    """
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
# 4. DATA CLEANING
# ══════════════════════════════════════════════════════════════════════════════

def clean_profile(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean sentinel values and sort surface → top.
    """
    MISSING = [-9999, -999, 9999, 99999, 2e20, 1e20]

    num_cols = ['temp_C', 'dewpoint_C', 'wind_speed_mps',
                'height_m', 'pressure_hPa']

    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].replace(MISSING, np.nan)
            # eccodes fills missing with 1e36 — catch that too
            df.loc[df[col].abs() > 1e10, col] = np.nan

    # Add ascent_rate_mps if not present (compute from height)
    if 'ascent_rate_mps' not in df.columns:
        df = df.sort_values('pressure_hPa', ascending=False).reset_index(drop=True)
        df['ascent_rate_mps'] = df['height_m'].diff().abs() / 1.0  # assume ~1s per level
        # This is approximate — if time column exists use that instead

    df = df.sort_values('pressure_hPa', ascending=False).reset_index(drop=True)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 5. FEATURE ENGINEERING  (identical to training)
# ══════════════════════════════════════════════════════════════════════════════

def engineer_features(flight_df: pd.DataFrame) -> dict | None:
    """
    Collapse the full vertical profile into one feature vector.
    Must be identical to the function used during training.
    """
    f = flight_df.copy()

    if len(f) < 3:
        return None

    f['dz']    = f['height_m'].diff().abs()
    f['dspd']  = f['wind_speed_mps'].diff().abs()
    f['dtemp'] = f['temp_C'].diff()
    dz_safe    = f['dz'].replace(0, np.nan)

    # Wind shear (m/s per 100m)
    f['shear'] = np.where(f['dz'] > 10, f['dspd'] / dz_safe * 100, np.nan)

    # Richardson number
    g, Gamma_d = 9.81, 9.8 / 1000
    T_K = f['temp_C'] + 273.15
    N2  = (g / T_K) * (f['dtemp'] / dz_safe + Gamma_d)
    S2  = (f['dspd'] / dz_safe) ** 2
    f['Ri'] = N2 / S2.replace(0, np.nan)

    # Dewpoint depression
    f['dd'] = f['temp_C'] - f['dewpoint_C']
    f['rh_approx'] = (100 - 5 * f['dd']).clip(0, 100)

    # Icing flag: sub-zero, near-saturated
    f['icing'] = (
        f['temp_C'].between(-20, 0) &
        (f['dd'].abs() < 3) &
        f['dewpoint_C'].notna()
    )

    # Heavy moisture flag: warm, near-saturated
    f['heavy_moisture'] = (
        (f['temp_C'] > 0) &
        (f['dd'] < 2) &
        f['dewpoint_C'].notna()
    )

    # Ascent rate stats
    ar      = f['ascent_rate_mps'].dropna() if 'ascent_rate_mps' in f else pd.Series(dtype=float)
    ar_mean = ar.mean() if len(ar) > 0 else np.nan
    ar_std  = ar.std()  if len(ar) > 1 else np.nan

    burst_alt = f['height_m'].max()
    near_burst = (
        f[f['height_m'] >= (burst_alt - 2000)]['ascent_rate_mps'].dropna()
        if 'ascent_rate_mps' in f else pd.Series(dtype=float)
    )

    spike_mask = (
        f['ascent_rate_mps'] > (ar_mean + 2 * ar_std)
        if (len(ar) > 0 and ar_std and ar_std > 0)
        else pd.Series([False] * len(f), index=f.index)
    )

    burst_pres    = f['pressure_hPa'].min()
    burst_idx     = f['pressure_hPa'].idxmin()
    temp_at_burst = f.loc[burst_idx, 'temp_C']

    shear_idx      = f['shear'].idxmax() if f['shear'].notna().any() else None
    max_shear      = f['shear'].max()
    shear_alt      = f.loc[shear_idx, 'height_m'] if shear_idx is not None else np.nan
    shear_to_burst = (burst_alt - shear_alt) if pd.notna(shear_alt) else np.nan

    ri_idx = f['Ri'].idxmin() if f['Ri'].notna().any() else None
    min_Ri = f['Ri'].min()
    ri_alt = f.loc[ri_idx, 'height_m'] if ri_idx is not None else np.nan
    boundary_layer_shear = f.loc[f['height_m'] <= 2000, 'shear'].max()

    icing_levels = f[f['icing']]
    icing_depth  = len(icing_levels) * 50
    icing_index  = (
        icing_levels['temp_C'].abs() * icing_levels['dd'].abs()
    ).mean() if len(icing_levels) > 0 else 0.0

    moist_levels  = f[f['heavy_moisture']]
    moist_depth   = len(moist_levels) * 50
    moist_index   = (
        moist_levels['rh_approx'] * moist_levels['dz'].fillna(50)
    ).sum() if len(moist_levels) > 0 else 0.0
    moist_dd_mean = moist_levels['dd'].mean() if len(moist_levels) > 0 else np.nan
    pw_proxy      = (f['rh_approx'] * f['dz'].fillna(50)).sum()

    min_temp_idx = f['temp_C'].idxmin() if f['temp_C'].notna().any() else None
    min_temp     = f['temp_C'].min()
    min_temp_alt = f.loc[min_temp_idx, 'height_m'] if min_temp_idx is not None else np.nan
   
    time_to_burst_mins = len(f) * (1 / 60) if 'time_s' not in f.columns else (f['time_s'].max() - f['time_s'].min()) / 60

    return {
        'burst_pres_hpa'        : burst_pres,
        'burst_alt_m'           : burst_alt,
        'max_shear'             : max_shear,
        'shear_alt_m'           : shear_alt,
        'shear_to_burst_m'      : shear_to_burst,
        'bulk_shear_lower'      : f.loc[f['height_m'] <= 6000, 'shear'].mean(),
        'bulk_shear_upper'      : f.loc[f['height_m'] >  6000, 'shear'].mean(),
        'max_wind_speed_mps'    : f['wind_speed_mps'].max(),
        'ascent_rate_mean'      : ar_mean,
        'ascent_rate_std'       : ar_std,
        'ascent_rate_var_burst' : near_burst.std() if len(near_burst) > 1 else np.nan,
        'ascent_rate_max_spike' : f.loc[spike_mask, 'ascent_rate_mps'].max()
                                  if spike_mask.any() else 0.0,
        'n_turbulent_spikes'    : int(spike_mask.sum()),
        'boundary_layer_shear'  : boundary_layer_shear if pd.notna(boundary_layer_shear) else 0.0,
        'min_richardson'        : min_Ri,
        'ri_alt_m'              : ri_alt,
        'n_unstable_layers'     : int((f['Ri'] < 0.25).sum()),
        'icing_depth_m'         : icing_depth,
        'icing_index'           : icing_index,
        'moist_depth_m'         : moist_depth,
        'moist_index'           : moist_index,
        'moist_dd_mean'         : moist_dd_mean,
        'pw_proxy'              : pw_proxy,
        'tropopause_temp_C'     : min_temp,
        'tropopause_alt_m'      : min_temp_alt,
        'temp_at_burst_C'       : temp_at_burst,
        'time_to_burst_mins'    : time_to_burst_mins,
    }


def fill_missing_features(feat: dict, feature_cols: list) -> dict:
    """Fill NaNs with safe defaults — same strategy as training."""
    defaults = {
        'icing_index'           : 0.0,
        'bulk_shear_upper'      : 0.0,
        'ascent_rate_max_spike' : 0.0,
        'n_turbulent_spikes'    : 0,
        'moist_depth_m'         : 0.0,
        'moist_index'           : 0.0,
    }
    for k, v in defaults.items():
        if feat.get(k) is None or (isinstance(feat.get(k), float) and np.isnan(feat[k])):
            feat[k] = v

    if feat.get('temp_at_burst_C') is None or (
        isinstance(feat.get('temp_at_burst_C'), float) and
        np.isnan(feat['temp_at_burst_C'])
    ):
        feat['temp_at_burst_C'] = feat.get('min_temp_C', 0.0)

    if feat.get('moist_dd_mean') is None or (
        isinstance(feat.get('moist_dd_mean'), float) and
        np.isnan(feat['moist_dd_mean'])
    ):
        feat['moist_dd_mean'] = 5.0  # neutral value

    # Zero-fill any remaining NaNs
    for col in feature_cols:
        v = feat.get(col)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            feat[col] = 0.0

    return feat


# ══════════════════════════════════════════════════════════════════════════════
# 6. EXPLANATION GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

# Feature thresholds for evidence extraction — tuned from training data
EVIDENCE_THRESHOLDS = {
    'launch_failure': {
        'burst_pres_hpa'      : ('high',  150,  'burst pressure was very high ({:.0f} hPa) — balloon never left lower troposphere'),
        'temp_at_burst_C'     : ('high',  -20,  'burst temperature was warm ({:.1f}°C) — still in mid-troposphere at burst'),
        'n_turbulent_spikes'  : ('low',   10,   'turbulent spikes were low ({:.0f}) — atmosphere was calm'),
        'max_wind_speed_mps'  : ('low',   12,   'wind speed was low ({:.1f} m/s) — no wind stress'),
    },
    'deep_moisture_icing': {
        'icing_depth_m'       : ('high',  3000, 'supercooled layer was {:.0f} m deep — extensive ice formation'),
        'pw_proxy'            : ('high',  1e6,  'column moisture was high — moist atmospheric profile'),
        'moist_dd_mean'       : ('low',   3.0,  'dewpoint depression was small ({:.1f}°C) — near-saturated conditions'),
    },
    'severe_icing': {
        'icing_index'         : ('high',  10,   'icing index was high ({:.1f}) — intense supercooled zone'),
        'icing_depth_m'       : ('high',  1000, 'supercooled layer present ({:.0f} m)'),
    },
    'high_updraft': {
        'ascent_rate_max_spike': ('high', 3.0,  'maximum ascent rate spike was {:.1f} m/s — strong updraft impact'),
        'ascent_rate_std'     : ('high',  1.0,  'ascent rate variability was high (std={:.2f} m/s)'),
        'min_richardson'      : ('low',   0.5,  'Richardson number was low ({:.2f}) — dynamic instability'),
        'max_wind_speed_mps'  : ('high',  15,   'wind speed was elevated ({:.1f} m/s)'),
    },
    'atmospheric_instability': {
        'ri_alt_m'            : ('high',  10000,'Richardson minimum found at high altitude ({:.0f} m)'),
        'max_shear'           : ('high',  0.05, 'wind shear was elevated ({:.3f} m/s per 100m)'),
        'n_unstable_layers'   : ('high',  3,    '{:.0f} dynamically unstable layers detected (Ri < 0.25)'),
    },
    'dry_layer': {
        'moist_dd_mean'       : ('high',  5.0,  'dewpoint depression was large ({:.1f}°C) — very dry atmosphere'),
        'moist_index'         : ('low',   1e5,  'moisture burden was very low — anomalously dry for Indonesia'),
        'bulk_shear_upper'    : ('high',  0.04, 'upper-level shear elevated ({:.3f} m/s per 100m)'),
    },
    'lower_wind_shear': {
        'max_shear'           : ('high',  0.05, 'maximum wind shear was {:.3f} m/s per 100m'),
        'shear_alt_m'         : ('high',  10000,'shear found low in the column ({:.0f} m)'),
    },
    'sensor_failure': {
        'tropopause_alt_m'      : ('low',   5000, 'cold point detected at very low altitude ({:.0f} m) — likely sensor error'),
        'height_m'              : ('high',   5000, 'maximum altitude reached was high ({:.0f} m) while detected tropopause is low — inconsistent with physical burst'),
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
    lines.append("  RADIOSONDE BURST CAUSE CLASSIFICATION")
    lines.append("=" * 62)
    lines.append(f"  Predicted cause  : {cause.replace('_',' ').upper()}")
    lines.append(f"  Confidence       : {confidence:.1f}%")
    lines.append(f"  Burst pressure   : {burst_info['burst_pres_hpa']} hPa  "
                 f"(threshold: {burst_info['threshold_hpa']} hPa)")
    if burst_info.get('burst_alt_m'):
        lines.append(f"  Burst altitude   : {burst_info['burst_alt_m']:,.0f} m")
    lines.append(f"  Sounding levels  : {burst_info['n_levels']}")
    lines.append("-" * 62)

    # Evidence
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
                found.append("  • " + template.format(val))
            except Exception:
                found.append(f"  • {feat_name} = {val:.3f}")

    if found:
        lines.extend(found)
    else:
        lines.append("  • Pattern matched cluster profile from training data")
        lines.append("    (individual thresholds not exceeded but overall")
        lines.append("    feature combination matches this cause)")

    # Cause description
    lines.append("")
    lines.append("  CAUSE DESCRIPTION:")
    desc = CAUSE_DESCRIPTIONS.get(cause, "No description available.")
    # Word-wrap at 60 chars
    words = desc.split()
    line_buf, wrapped = [], []
    for w in words:
        if sum(len(x)+1 for x in line_buf) + len(w) > 56:
            wrapped.append("  " + " ".join(line_buf))
            line_buf = [w]
        else:
            line_buf.append(w)
    if line_buf:
        wrapped.append("  " + " ".join(line_buf))
    lines.extend(wrapped)

    # Alternative causes
    lines.append("")
    lines.append("  ALTERNATIVE CAUSES:")
    sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
    shown = 0
    for c, p in sorted_probs:
        if c == cause:
            continue
        if p >= 5.0:
            lines.append(f"  • {c.replace('_',' '):<30} {p:.1f}%")
            shown += 1
    if shown == 0:
        lines.append("  • No alternative above 5% confidence")

    lines.append("=" * 62)
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# 7. MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(file_path: str, model_dir: str = "models") -> dict:

    print(f"\nProcessing: {Path(file_path).name}")
    print("-" * 62)

    # ── Step 1: Parse BUFR ────────────────────────────────────────────────────
    print("  [1/6] Parsing BUFR file...")
    df = extract_radiosonde_data(file_path)

    if df.empty:
        print("  ERROR: Could not parse BUFR file or no data found.")
        return {"file": file_path, "error": "parse_failed"}

    print(f"        {len(df)} pressure levels extracted")

    # ── Step 2: Clean ─────────────────────────────────────────────────────────
    print("  [2/6] Cleaning profile data...")
    df = clean_profile(df)

    # ── Step 3: 30 hPa threshold ──────────────────────────────────────────────
    print("  [3/6] Checking burst pressure threshold (30 hPa)...")
    burst_info = detect_premature_burst(df)

    if burst_info.get('error'):
        print(f"  ERROR: {burst_info['error']}")
        return {"file": file_path, "error": burst_info['error']}

    print(f"        Burst pressure : {burst_info['burst_pres_hpa']} hPa")
    print(f"        Premature      : {'YES' if burst_info['is_premature'] else 'NO'}")

    # ── If nominal, no classifier needed ──────────────────────────────────────
    if not burst_info['is_premature']:
        result = {
            "file"            : Path(file_path).name,
            "launch_time"     : str(df['launch_time'].iloc[0])
                                if 'launch_time' in df.columns else None,
            "station_id"      : str(df['station_id'].iloc[0])
                                if 'station_id' in df.columns else None,
            "is_premature"    : False,
            "burst_pres_hpa"  : burst_info['burst_pres_hpa'],
            "burst_alt_m"     : burst_info['burst_alt_m'],
            "predicted_cause" : "nominal",
            "confidence_pct"  : 100.0,
            "explanation"     : generate_explanation(
                "nominal", 100.0, {}, {"nominal": 100.0}, burst_info
            ),
        }
        print(result["explanation"])
        return result

    # ── Step 4: Feature engineering ───────────────────────────────────────────
    print("  [4/6] Engineering features from vertical profile...")
    features_raw = engineer_features(df)

    if features_raw is None:
        print("  ERROR: Too few valid levels for feature engineering.")
        return {"file": file_path, "error": "insufficient_levels"}

    # ── Step 5: Load model ────────────────────────────────────────────────────
    print("  [5/6] Loading saved model and scaler...")
    model_path   = Path(model_dir) / "burst_classifier_model.joblib"
    scaler_path  = Path(model_dir) / "burst_classifier_scaler.joblib"
    encoder_path = Path(model_dir) / "burst_label_encoder.joblib"
    cols_path    = Path(model_dir) / "feature_cols.joblib"

    for p in [model_path, scaler_path, encoder_path, cols_path]:
        if not p.exists():
            print(f"  ERROR: Model file not found: {p}")
            print("         Run training script first to generate model files.")
            return {"file": file_path, "error": f"missing_model_file: {p.name}"}

    model        = joblib.load(model_path)
    scaler       = joblib.load(scaler_path)
    le           = joblib.load(encoder_path)
    feature_cols = joblib.load(cols_path)

    # Fill missing features using same strategy as training
    features = fill_missing_features(features_raw, feature_cols)

    # Build feature vector in exact column order
    X = np.array([[features.get(c, 0.0) for c in feature_cols]])
    X_scaled = scaler.transform(X)

    # ── Step 6: Predict ───────────────────────────────────────────────────────
    print("  [6/6] Running classifier...")
    pred_enc    = model.predict(X_scaled)[0]
    proba       = model.predict_proba(X_scaled)[0]
    cause       = le.inverse_transform([pred_enc])[0]
    confidence  = float(proba.max()) * 100
    all_probs   = {
        cls: round(float(p) * 100, 1)
        for cls, p in zip(le.classes_, proba)
    }

    # ── Generate explanation ──────────────────────────────────────────────────
    explanation = generate_explanation(
        cause, confidence, features, all_probs, burst_info
    )

    result = {
        "file"             : Path(file_path).name,
        "launch_time"      : str(df['launch_time'].iloc[0])
                             if 'launch_time' in df.columns else None,
        "station_id"       : str(df['station_id'].iloc[0])
                             if 'station_id' in df.columns else None,
        "is_premature"     : True,
        "burst_pres_hpa"   : burst_info['burst_pres_hpa'],
        "burst_alt_m"      : burst_info['burst_alt_m'],
        "n_levels"         : burst_info['n_levels'],
        "predicted_cause"  : cause,
        "confidence_pct"   : round(confidence, 1),
        "all_probabilities": all_probs,
        "features_used"    : {k: round(float(v), 4)
                              for k, v in features.items()
                              if k in feature_cols},
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
        description="Radiosonde BUFR burst cause classifier"
    )
    parser.add_argument(
        "--file",
        type    = str,
        required= True,
        help    = "Path to BUFR radiosonde file"
    )
    parser.add_argument(
        "--model_dir",
        type    = str,
        default = "models",
        help    = "Directory containing saved model files (default: models/)"
    )
    parser.add_argument(
        "--save_json",
        type    = str,
        default = None,
        help    = "Optional path to save result as JSON e.g. result.json"
    )
    args = parser.parse_args()

    if not Path(args.file).exists():
        print(f"ERROR: File not found: {args.file}")
        sys.exit(1)

    result = run_pipeline(args.file, args.model_dir)

    if args.save_json:
        save_result = {k: v for k, v in result.items()
                       if k != 'features_used'}
        with open(args.save_json, 'w') as f:
            json.dump(save_result, f, indent=2, default=str)
        print(f"\nResult saved to: {args.save_json}")


if __name__ == "__main__":
    # Allow direct call with hardcoded file for quick testing
    if len(sys.argv) == 1:
        # No args — run on the file from your code
        result = run_pipeline(
            "data/A_IUSG51WION111200_C_WIIX_20251011120000.bin",
            model_dir="models"
        )
    else:
        main()