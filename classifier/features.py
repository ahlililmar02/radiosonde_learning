"""
features.py
───────────
Single source of truth for feature engineering used by both
train_secondary_model.py (training) and secondary_classifier.py (inference).

Keeping these functions here prevents silent divergence between the feature
vector the model was trained on and the one computed at inference time.
"""

import numpy as np
import pandas as pd

from classifier.rule_engine import compute_cape_cin


MISSING = [-9999, -999, 9999, 99999, 2e20, 1e20]

PROFILE_COLS = ['temp_C', 'dewpoint_C', 'wind_speed_mps', 'height_m',
                'pressure_hPa', 'ascent_rate_mps']

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
    'dewdep_burst', 'min_dewdep_warm', 'n_high_rh', 'ascent_drop',
    'peak_ascent', 'med_ascent', 'dir_delta', 'spd_max', 'cpt_dist',
    'pressure_increase_total',
    'cape_jkg', 'cin_jkg', 'lcl_agl_m',
]


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

    strato = f[f['height_m'] > 10000]
    cpt_dist = (burst_alt - strato.loc[strato['temp_C'].idxmin(), 'height_m']) if not strato.empty else np.nan

    tail100 = f.tail(100)
    pos_diffs = tail100['pressure_hPa'].diff()
    pos_diffs = pos_diffs[pos_diffs > 0]
    pressure_increase_total = pos_diffs.sum() if len(pos_diffs) else 0.0

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


def fill_missing_features(feat: dict, feature_cols: list) -> dict:
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
        feat['moist_dd_mean'] = 5.0

    if feat.get('cpt_dist') is None or (
        isinstance(feat.get('cpt_dist'), float) and np.isnan(feat['cpt_dist'])
    ):
        feat['cpt_dist'] = 20000.0

    if feat.get('min_dewdep_warm') is None or (
        isinstance(feat.get('min_dewdep_warm'), float) and np.isnan(feat['min_dewdep_warm'])
    ):
        feat['min_dewdep_warm'] = 50.0

    for col in feature_cols:
        v = feat.get(col)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            feat[col] = 0.0

    return feat
