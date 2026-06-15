"""
train_secondary_model.py
─────────────────────────
Trains the secondary burst-cause classifier on the recalibrated
percentile-rank labels produced by classifier.ipynb
(combined_codes_train.csv / combined_codes_test.csv).

Does NOT touch draft/final_classifier.py or draft/models/ — this script
builds a separate, self-contained model bundle in draft/models_v2/ for
the new secondary-cause taxonomy (RH Freeze-out, Strong Shear, Cold Point
Tropopause, Pressure Reversal, Slow Ascent, Cloud/Rain Layer, Deep
Convection, ...).

Usage:
  uv run python draft/train_secondary_model.py
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV

from draft.rule_engine import compute_cape_cin


# ══════════════════════════════════════════════════════════════════════════════
# 1. FEATURE ENGINEERING (kept identical to draft/final_classifier.py
#    engineer_features / fill_missing_features so the same code can run
#    on a flight profile parsed from a BUFR file at inference time)
# ══════════════════════════════════════════════════════════════════════════════

MISSING = [-9999, -999, 9999, 99999, 2e20, 1e20]

PROFILE_COLS = ['temp_C', 'dewpoint_C', 'wind_speed_mps', 'height_m',
                'pressure_hPa', 'ascent_rate_mps']


def clean_profile(flight_df: pd.DataFrame) -> pd.DataFrame:
    f = flight_df.copy()
    for col in PROFILE_COLS:
        if col in f.columns:
            f[col] = pd.to_numeric(f[col], errors='coerce')
            f[col] = f[col].replace(MISSING, np.nan)
    return f.sort_values('pressure_hPa', ascending=False).reset_index(drop=True)


def engineer_features(flight_df: pd.DataFrame) -> dict | None:
    f = flight_df.copy()

    if len(f) < 3:
        return None

    f['dz']    = f['height_m'].diff().abs()
    f['dspd']  = f['wind_speed_mps'].diff().abs()
    f['dtemp'] = f['temp_C'].diff()
    dz_safe    = f['dz'].replace(0, np.nan)

    f['shear'] = np.where(f['dz'] > 10, f['dspd'] / dz_safe * 100, np.nan)

    g, Gamma_d = 9.81, 9.8 / 1000
    T_K = f['temp_C'] + 273.15
    N2  = (g / T_K) * (f['dtemp'] / dz_safe + Gamma_d)
    S2  = (f['dspd'] / dz_safe) ** 2
    f['Ri'] = N2 / S2.replace(0, np.nan)

    f['dd'] = f['temp_C'] - f['dewpoint_C']
    f['rh_approx'] = (100 - 5 * f['dd']).clip(0, 100)

    f['icing'] = (
        f['temp_C'].between(-20, 0) &
        (f['dd'].abs() < 3) &
        f['dewpoint_C'].notna()
    )

    f['heavy_moisture'] = (
        (f['temp_C'] > 0) &
        (f['dd'] < 2) &
        f['dewpoint_C'].notna()
    )

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

    # ── Rule-engine raw signals (draft/rule_engine.py::compute_raw_signals),
    #    exposed as model features so the classifier sees the same signals
    #    the secondary-cause labels were derived from. gps_frozen is
    #    excluded — BUFR inference has no per-level lat/lon. ─────────────────
    last = f.iloc[-1]
    dewdep_burst = last['temp_C'] - last['dewpoint_C']

    warm = f[f['temp_C'] > -10]
    min_dewdep_warm = (warm['temp_C'] - warm['dewpoint_C']).min() if not warm.empty else np.nan

    n_high_rh = int((f['dd'] < 2).sum())

    tail30 = f.tail(30)
    ascent_drop = tail30['ascent_rate_mps'].diff().min()
    peak_ascent = tail30['ascent_rate_mps'].max()
    med_ascent  = f['ascent_rate_mps'].median()

    top_500 = f[f['height_m'] > burst_alt - 500]
    if len(top_500) > 1:
        dir_delta = top_500['wind_dir_deg'].diff().abs().max()
        spd_max   = top_500['wind_speed_mps'].max()
    else:
        dir_delta = np.nan
        spd_max   = np.nan

    # Cold-point tropopause distance: burst altitude minus the altitude of
    # the coldest temperature above 10 km (matches rule_engine cpt_dist —
    # NOT the same as shear_to_burst_m above).
    strato = f[f['height_m'] > 10000]
    cpt_dist = (burst_alt - strato.loc[strato['temp_C'].idxmin(), 'height_m']) if not strato.empty else np.nan

    tail100 = f.tail(100)
    pos_diffs = tail100['pressure_hPa'].diff()
    pos_diffs = pos_diffs[pos_diffs > 0]
    pressure_increase_total = pos_diffs.sum() if len(pos_diffs) else 0.0

    # Surface-based CAPE/CIN/LCL (MetPy parcel theory, rule_engine.compute_cape_cin) —
    # scientifically grounded convective-potential signals, replacing the
    # glitch-prone ascent-rate-spike proxy for Deep Convection.
    cape_cin = compute_cape_cin(f)

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
        'min_richardson'        : np.clip(min_Ri, -5, 5) if pd.notna(min_Ri) else np.nan,
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
        'dewdep_burst'          : dewdep_burst,
        'min_dewdep_warm'       : min_dewdep_warm,
        'n_high_rh'             : n_high_rh,
        'ascent_drop'           : ascent_drop,
        'peak_ascent'           : peak_ascent,
        'med_ascent'            : med_ascent,
        'dir_delta'             : dir_delta,
        'spd_max'               : spd_max,
        'cpt_dist'              : cpt_dist,
        'pressure_increase_total': pressure_increase_total,
        'cape_jkg'              : cape_cin['cape_jkg'],
        'cin_jkg'               : cape_cin['cin_jkg'],
        'lcl_agl_m'             : cape_cin['lcl_agl_m'],
    }


FEATURE_COLS = [
    'burst_pres_hpa', 'burst_alt_m',
    'max_shear', 'shear_alt_m', 'shear_to_burst_m',
    'bulk_shear_lower', 'bulk_shear_upper', 'max_wind_speed_mps',
    'ascent_rate_mean', 'ascent_rate_std', 'ascent_rate_var_burst',
    'ascent_rate_max_spike', 'n_turbulent_spikes', 'boundary_layer_shear',
    'min_richardson', 'ri_alt_m', 'n_unstable_layers',
    'icing_depth_m', 'icing_index',
    'moist_depth_m', 'moist_index', 'moist_dd_mean', 'pw_proxy',
    'tropopause_temp_C', 'tropopause_alt_m', 'temp_at_burst_C',
    'time_to_burst_mins',
    # Rule-engine raw signals (see compute_raw_signals in rule_engine.py) —
    # added so the model has direct access to the signals that drove the
    # weak labels, and so cpt_dist gives the model a correctly-defined
    # cold-point-tropopause feature (see Cold Point Tropopause caveat).
    'dewdep_burst', 'min_dewdep_warm', 'n_high_rh', 'ascent_drop',
    'peak_ascent', 'med_ascent', 'dir_delta', 'spd_max', 'cpt_dist',
    'pressure_increase_total',
    # Surface-based CAPE/CIN/LCL (MetPy parcel theory) — scientifically
    # grounded convective-potential signals (see Deep Convection rule).
    'cape_jkg', 'cin_jkg', 'lcl_agl_m',
]


def fill_missing_features(feat: dict, feature_cols: list) -> dict:
    """Fill NaNs with safe defaults — same strategy as final_classifier.py."""
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
        feat['temp_at_burst_C'] = feat.get('tropopause_temp_C', 0.0)

    if feat.get('moist_dd_mean') is None or (
        isinstance(feat.get('moist_dd_mean'), float) and
        np.isnan(feat['moist_dd_mean'])
    ):
        feat['moist_dd_mean'] = 5.0  # neutral value

    # cpt_dist is NaN when the flight never reached 10 km — treat as "far
    # from the cold point" rather than the generic 0.0 ("right at it").
    if feat.get('cpt_dist') is None or (
        isinstance(feat.get('cpt_dist'), float) and np.isnan(feat['cpt_dist'])
    ):
        feat['cpt_dist'] = 20000.0

    # min_dewdep_warm is NaN when there are no levels warmer than -10C —
    # treat as "not saturated" rather than 0.0 ("fully saturated").
    if feat.get('min_dewdep_warm') is None or (
        isinstance(feat.get('min_dewdep_warm'), float) and np.isnan(feat['min_dewdep_warm'])
    ):
        feat['min_dewdep_warm'] = 50.0

    for col in feature_cols:
        v = feat.get(col)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            feat[col] = 0.0

    return feat


# ══════════════════════════════════════════════════════════════════════════════
# 2. BUILD FEATURE TABLE FROM rason_complete.csv + combined_codes_*.csv
# ══════════════════════════════════════════════════════════════════════════════

def build_feature_table(rason_df: pd.DataFrame, labels_df: pd.DataFrame):
    """
    Restrict training to flights that are (a) premature and (b) have a
    valid secondary label from the xlsx-masked rule engine. The rule
    engine's secondary causes describe failure signatures near burst —
    including nominal (non-premature) flights adds label noise (their
    "secondary cause" doesn't represent why they failed, since they
    didn't) and measurably hurts test accuracy (0.66 -> 0.55).

    The trained model is still applied to ALL flights at inference time
    (draft/secondary_classifier.py) — is_premature is shown as a display
    field there, not used to skip classification.
    """
    label_lookup = (
        labels_df[labels_df['is_premature'] & labels_df['valid']]
        .set_index('flight_id')['secondary']
        .to_dict()
    )

    X_rows, y, flight_ids = [], [], []
    skipped = 0

    for fid, group in rason_df.groupby('flight_id'):
        if fid not in label_lookup:
            continue

        f = clean_profile(group)
        feat = engineer_features(f)
        if feat is None:
            skipped += 1
            continue

        feat = fill_missing_features(feat, FEATURE_COLS)
        X_rows.append([feat.get(c, 0.0) for c in FEATURE_COLS])
        y.append(label_lookup[fid])
        flight_ids.append(fid)

    X = np.array(X_rows, dtype=float)
    return X, np.array(y), flight_ids, skipped


# ══════════════════════════════════════════════════════════════════════════════
# 3. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    rason_df = pd.read_csv('data/rason_complete.csv', low_memory=False)
    train_labels = pd.read_csv('combined_codes_train.csv')
    test_labels  = pd.read_csv('combined_codes_test.csv')

    X_train, y_train, train_ids, train_skipped = build_feature_table(rason_df, train_labels)
    X_test,  y_test,  test_ids,  test_skipped  = build_feature_table(rason_df, test_labels)

    print(f"Train samples: {len(y_train)} (skipped {train_skipped})")
    print(f"Test samples : {len(y_test)} (skipped {test_skipped})")
    print("\nTrain class distribution:")
    print(pd.Series(y_train).value_counts())

    # ── Scale ────────────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # ── Encode labels ────────────────────────────────────────────────────────
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc  = le.transform(y_test)

    # ── Cross-validation on the training set ────────────────────────────────
    # 319 train / 70 test is small enough that a single split has high
    # variance. 5-fold stratified CV over the training set gives a more
    # stable estimate of how the model+features generalize.
    cv_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_acc = cross_val_score(cv_model, X_train_scaled, y_train_enc, cv=skf, scoring='accuracy')
    cv_f1  = cross_val_score(cv_model, X_train_scaled, y_train_enc, cv=skf, scoring='f1_macro')
    print(f"\n5-fold CV on train set ({len(y_train)} flights, {len(FEATURE_COLS)} features):")
    print(f"  Accuracy : {cv_acc.mean():.3f} +/- {cv_acc.std():.3f}  {np.round(cv_acc, 3)}")
    print(f"  Macro F1 : {cv_f1.mean():.3f} +/- {cv_f1.std():.3f}  {np.round(cv_f1, 3)}")

    # ── Hyperparameter search (5-fold CV, optimizing macro F1) ──────────────
    param_grid = {
        'n_estimators'     : [200, 300, 500],
        'max_depth'        : [6, 8, 10, None],
        'min_samples_leaf' : [1, 2, 4],
        'max_features'     : ['sqrt', 0.5],
    }
    search = GridSearchCV(
        RandomForestClassifier(class_weight='balanced', random_state=42),
        param_grid,
        scoring='f1_macro',
        cv=skf,
        n_jobs=-1,
    )
    search.fit(X_train_scaled, y_train_enc)
    print(f"\nGrid search best params: {search.best_params_}")
    print(f"Grid search best CV macro F1: {search.best_score_:.3f}")
    print(f"(baseline CV macro F1 was {cv_f1.mean():.3f} with "
          f"n_estimators=300, max_depth=8, min_samples_leaf=2, max_features='sqrt')")

    # ── Train final model with best hyperparameters ──────────────────────────
    model = search.best_estimator_

    # ── Evaluate ─────────────────────────────────────────────────────────────
    y_pred = model.predict(X_test_scaled)
    print("\nHeld-out test set classification report:")
    print(classification_report(
        y_test_enc, y_pred, target_names=le.classes_, zero_division=0
    ))

    # ── Save model bundle ────────────────────────────────────────────────────
    out_dir = Path('draft/models_v2')
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, out_dir / 'secondary_classifier_model.joblib')
    joblib.dump(scaler, out_dir / 'secondary_classifier_scaler.joblib')
    joblib.dump(le, out_dir / 'secondary_label_encoder.joblib')
    joblib.dump(FEATURE_COLS, out_dir / 'feature_cols.joblib')

    print(f"\nSaved model bundle to {out_dir}/")


if __name__ == '__main__':
    main()
