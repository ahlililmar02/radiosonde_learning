import joblib
import numpy as np
import pandas as pd 
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import umap

"""Classifier module for radiosonde flight classification.
This module loads the pre-trained classification pipeline and defines the main function
to classify a new radiosonde flight based on its vertical profile data.
The classification process includes:
1. Feature engineering: Compute derived features from the raw sounding data.    
2. Data cleaning: Handle missing values and outliers in the engineered features.
3. Feature selection: Select the subset of features used by the model.
4. Data transformation: Apply scaling and dimensionality reduction to match the training data format.
5. Prediction: Use the Gaussian Mixture Model to predict the flight class and confidence score.
The module is designed to be used in a production environment where new radiosonde data can be classified on demand.

Assuming the following structure for the input DataFrame 
`df`:
Index(['time_s', 'status_flag', 'pressure_hPa', 'height_m', 'lat_disp',
       'lon_disp', 'temp_C', 'dewpoint_C', 'wind_dir_deg', 'wind_speed_mps',
       'ascent_rate_mps', 'latitude', 'longitude'],
      dtype='str')
      
example :
time_s  status_flag  pressure_hPa  height_m  lat_disp  lon_disp  \
1578472     0.0     145408.0        1010.6       7.0   0.00000   0.00000   
1578473     2.0          0.0        1008.4      26.0   0.00000   0.00005   
1578474     4.0          0.0        1006.8      40.0   0.00003   0.00004   
1578475     6.0          0.0        1005.4      53.0   0.00000   0.00002   
1578476     8.0          0.0        1004.3      62.0  -0.00006   0.00011   

         temp_C  dewpoint_C  wind_dir_deg  wind_speed_mps  ascent_rate_mps  \
1578472   27.00       26.83         292.0             1.5              NaN   
1578473   26.62       25.62         291.0             1.7              9.5   
1578474   26.34       25.34         291.0             1.8              7.0   
1578475   26.21       25.32         292.0             1.9              6.5   
1578476   26.09       25.31         294.0             2.0              4.5   

         latitude  longitude  
1578472   3.32705  117.57047  
1578473   3.32705  117.57052  
1578474   3.32708  117.57051  
1578475   3.32705  117.57049  
1578476   3.32699  117.57058
"""

def engineer_features(flight_df):
    f = flight_df.sort_values('pressure_hPa', ascending=False).reset_index(drop=True)

    if len(f) < 3:
        return None

    for col in ['temp_C', 'dewpoint_C', 'wind_speed_mps',
                'height_m', 'ascent_rate_mps']:
        f[col] = pd.to_numeric(f[col], errors='coerce')
        f[col] = f[col].replace([-9999, -999, 9999, 99999], np.nan)

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

    # ── dewpoint depression (spread) — key moisture diagnostic ───────────
    # Small spread = near-saturated air = balloon skin gets wet
    f['dd'] = f['temp_C'] - f['dewpoint_C']  # dewpoint depression (°C)
    # Relative humidity proxy from dewpoint depression
    # Magnus approximation: RH ≈ 100 - 5 * dd (valid for dd < 50°C)
    f['rh_approx'] = (100 - 5 * f['dd']).clip(0, 100)

    # ── ICING flag: supercooled liquid water zone ─────────────────────────
    # Requires sub-zero temperature AND near-saturation
    # Reference: GRUAN 2025 — ice formation on balloon membrane
    f['icing'] = (
        f['temp_C'].between(-20, 0) &
        (f['dd'].abs() < 3) &
        f['dewpoint_C'].notna()
    )

    # ── HEAVY MOISTURE flag: warm saturated layers ────────────────────────
    # Distinct from icing — positive temperature, near-saturated
    # Mechanism: liquid water collecting on latex skin weakens membrane
    # Reference: latex balloon science — prolonged moisture exposure
    # reduces tensile strength (BalloonHQ, Balloon Science 101)
    # Indonesia-specific: deep warm moist layer from maritime convection
    f['heavy_moisture'] = (
        f['temp_C'] > 0 &           # warm — NOT icing territory
        (f['dd'] < 2) &             # near-saturated (RH > ~90%)
        f['dewpoint_C'].notna()
    )

    # ascent rate stats
    ar         = f['ascent_rate_mps'].dropna()
    ar_mean    = ar.mean()
    ar_std     = ar.std() if len(ar) > 1 else np.nan
    burst_alt  = f['height_m'].max()
    near_burst = f[f['height_m'] >= (burst_alt - 2000)]['ascent_rate_mps'].dropna()

    spike_mask = (
        f['ascent_rate_mps'] > (ar_mean + 2 * ar_std)
        if (ar_std and ar_std > 0)
        else pd.Series([False] * len(f))
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

    # icing stats
    icing_levels = f[f['icing']]
    icing_depth  = len(icing_levels) * 50
    icing_index  = (
        icing_levels['temp_C'].abs() *
        icing_levels['dd'].abs()
    ).mean() if len(icing_levels) > 0 else 0.0

    # ── heavy moisture stats ──────────────────────────────────────────────
    moist_levels = f[f['heavy_moisture']]

    # depth of continuous warm saturated layer (m)
    moist_depth  = len(moist_levels) * 50

    # moisture burden index: integrate RH over the saturated depth
    # higher = more cumulative latex stress from moisture exposure
    moist_index  = (
        moist_levels['rh_approx'] *
        (moist_levels['dz'].fillna(50))       # layer thickness weight
    ).sum() if len(moist_levels) > 0 else 0.0

    # mean dewpoint depression in the saturated layer
    # smaller = more saturated = more skin wetting
    moist_dd_mean = moist_levels['dd'].mean() if len(moist_levels) > 0 else np.nan

    # column-integrated precipitable water proxy
    # integral of RH-approx over full profile (not just saturated layers)
    pw_proxy = (f['rh_approx'] * f['dz'].fillna(50)).sum()

    min_temp_idx = f['temp_C'].idxmin() if f['temp_C'].notna().any() else None
    min_temp     = f['temp_C'].min()
    min_temp_alt = f.loc[min_temp_idx, 'height_m'] if min_temp_idx is not None else np.nan

    return {
        'n_levels'              : len(f),
        'burst_pres_hpa'        : burst_pres,
        'burst_alt_m'           : burst_alt,

        # wind shear
        'max_shear'             : max_shear,
        'shear_alt_m'           : shear_alt,
        'shear_to_burst_m'      : shear_to_burst,
        'bulk_shear_lower'      : f.loc[f['height_m'] <= 6000, 'shear'].mean(),
        'bulk_shear_upper'      : f.loc[f['height_m'] >  6000, 'shear'].mean(),
        'max_wind_speed_mps'    : f['wind_speed_mps'].max(),

        # turbulence
        'ascent_rate_mean'      : ar_mean,
        'ascent_rate_std'       : ar_std,
        'ascent_rate_var_burst' : near_burst.std() if len(near_burst) > 1 else np.nan,
        'ascent_rate_max_spike' : f.loc[spike_mask, 'ascent_rate_mps'].max()
                                  if spike_mask.any() else 0.0,
        'n_turbulent_spikes'    : int(spike_mask.sum()),

        # Richardson
        'min_richardson'        : min_Ri,
        'ri_alt_m'              : ri_alt,
        'n_unstable_layers'     : int((f['Ri'] < 0.25).sum()),

        # icing (sub-zero, near-saturated)
        'icing_depth_m'         : icing_depth,
        'icing_index'           : icing_index,

        # heavy moisture (warm, near-saturated) 
        'moist_depth_m'         : moist_depth,
        'moist_index'           : moist_index,
        'moist_dd_mean'         : moist_dd_mean,
        'pw_proxy'              : pw_proxy,

        # cold tropopause
        'min_temp_C'            : min_temp,
        'min_temp_alt_m'        : min_temp_alt,
        'temp_at_burst_C'       : temp_at_burst,
    }


def clean_features(df, medians):

    df = df.copy()

    df['icing_index'] = df['icing_index'].fillna(0)
    df['bulk_shear_upper'] = df['bulk_shear_upper'].fillna(0)
    df['ascent_rate_max_spike'] = df['ascent_rate_max_spike'].fillna(0)
    df['n_turbulent_spikes'] = df['n_turbulent_spikes'].fillna(0)

    df['temp_at_burst_C'] = df['temp_at_burst_C'].fillna(df['min_temp_C'])

    for col in medians.index:
        if col in df.columns:
            df[col] = df[col].fillna(medians[col])

    return df


def classify_prepared_data(df, scaler, reducer_5d, gmm, feature_cols, medians):
    REQUIRED_COLUMNS = [
        'pressure_hPa', 'height_m', 'temp_C',
        'dewpoint_C', 'wind_speed_mps', 'ascent_rate_mps'
    ]

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if len(df) < 10:
        raise ValueError("Input profile too short for reliable classification")

    # --- Step 1: Feature engineering ---
    val_raw_dict = engineer_features(df)

    if val_raw_dict is None:
        raise ValueError("Feature engineering failed (insufficient data)")

    # --- Step 2: Cleaning ---
    val_feat_clean = clean_features(pd.DataFrame([val_raw_dict]), medians)

    # --- Step 3: Select features ---
    X_cleaned = val_feat_clean[feature_cols].values

    # --- Step 4: Transform ---
    X_scaled = scaler.transform(X_cleaned)
    X_5d = reducer_5d.transform(X_scaled)

    # --- Step 5: Predict ---
    pred_idx = gmm.predict(X_5d)[0]
    conf = np.max(gmm.predict_proba(X_5d))

    if conf < 0.6:
        print("⚠️ Warning: Low confidence prediction")

    return pred_idx, conf, val_feat_clean

label_map = {
    0: "Mechanical (Shear)",                   # Cluster 0
    1: "Dynamic Instability",                # Cluster 1
    2: "Early Launch Failure",    # Cluster 2
    3: "Deep Moist Convection",     # Cluster 3
    4: "Dry Cooling",   # Cluster 4
    5: "Microphysical Icing",                       # Cluster 5
    6: "Shallow Cold Layer"
}


df = pd.read_csv("example_flight_data.csv")

# Basic validation
if df.empty:
    raise ValueError("Input dataframe is empty")

# --- Determine flight outcome ---
max_alt = df['height_m'].max()
min_pres = df['pressure_hPa'].min()

print(f"Max Altitude: {max_alt:.1f} m")
print(f"Min Pressure: {min_pres:.1f} hPa")

TARGET_PRESSURE = 30

# --- Decision ---
if (min_pres <= TARGET_PRESSURE) :
    print("✅ Flight reached target altitude and pressure (NORMAL)")
else:
    print("⚠️ Premature termination detected → running classification")

    # Load model ONLY if needed
    pipeline = joblib.load("flight_classifier.pkl")

    scaler = pipeline["scaler"]
    reducer_5d = pipeline["umap"]
    gmm = pipeline["gmm"]
    feature_cols = pipeline["feature_cols"]
    medians = pipeline["medians"]

    # Run your classifier
    pred_idx, conf, val_feat_clean = classify_prepared_data(
    df, scaler, reducer_5d, gmm, feature_cols, medians)

    print(f"Predicted cluster: {pred_idx}")
    print(f"Confidence: {conf:.2f}")

if pred_idx == 0:
    print("Explanation: MECHANICAL FAILURE (SHEAR-INDUCED RUPTURE)")
    print("Analysis: Balloon encountered strong vertical wind shear near burst altitude.")
    print("Key Signals:")
    print(f"  - Shear-to-burst distance: {val_feat_clean['shear_to_burst_m'].values[0]:.1f} m (shear close to burst)")
    print(f"  - Max shear: {val_feat_clean['max_shear'].values[0]:.3f} s⁻¹ (elevated)")
    print("Interpretation: Balloon likely ruptured due to mechanical stress from sharp wind gradients.")

elif pred_idx == 1:
    print("Explanation: DYNAMIC INSTABILITY (ALOFT TURBULENCE)")
    print("Analysis: Balloon reached higher altitude and encountered instability layers.")
    print("Key Signals:")
    print(f"  - Richardson instability altitude: {val_feat_clean['ri_alt_m'].values[0]:.0f} m")
    print(f"  - Burst pressure: {val_feat_clean['burst_pres_hpa'].values[0]:.1f} hPa (low → high altitude)")
    print("Interpretation: Failure likely caused by turbulence or dynamic instability in the upper atmosphere.")

elif pred_idx == 2:
    print("Explanation: EARLY LAUNCH / LOW-ALTITUDE FAILURE")
    print("Analysis: Balloon failed prematurely at low altitude under high pressure conditions.")
    print("Key Signals:")
    print(f"  - Burst altitude: {val_feat_clean['burst_alt_m'].values[0]:.1f} m")
    print(f"  - Burst pressure: {val_feat_clean['burst_pres_hpa'].values[0]:.1f} hPa")
    print(f"  - Temperature at burst: {val_feat_clean['temp_at_burst_C'].values[0]:.1f} °C")
    print(f"  - Max wind speed: {val_feat_clean['max_wind_speed_mps'].values[0]:.1f} m/s")
    print("Interpretation: Failure likely due to launch stress or material weakness, not atmospheric forcing.")

elif pred_idx == 3:
    print("Explanation: DEEP MOIST CONVECTION (MOISTURE-DRIVEN FAILURE)")
    print("Analysis: Balloon ascended through a deep moist atmospheric column.")
    print("Key Signals:")
    print(f"  - Icing depth: {val_feat_clean['icing_depth_m'].values[0]:.0f} m")
    print(f"  - Precipitable water proxy: {val_feat_clean['pw_proxy'].values[0]:.0f} (relative indicator)")
    print("Interpretation: Failure likely linked to strong moisture loading, cloud processes, or embedded convection.")

elif pred_idx == 4:
    print("Explanation: DRY COLD REGIME (NON-ICING THERMODYNAMIC STRESS)")
    print("Analysis: Balloon operated in a dry atmosphere with low moisture and no icing support.")
    print("Key Signals:")
    print(f"  - Icing index: {val_feat_clean['icing_index'].values[0]:.2f} (low)")
    print(f"  - Moisture index: {val_feat_clean['moist_index'].values[0]:.0f} (low)")
    print(f"  - Mean dewpoint depression: {val_feat_clean['moist_dd_mean'].values[0]:.2f} °C")
    print("Interpretation: Failure likely due to cold, dry conditions causing material brittleness.")

elif pred_idx == 5:
    print("Explanation: MICROPHYSICAL ICING (SUPERCOOLED WATER)")
    print("Analysis: Balloon encountered localized icing conditions.")
    print("Key Signals:")
    print(f"  - Icing index: {val_feat_clean['icing_index'].values[0]:.2f} (high)")
    print("Interpretation: Failure likely due to ice accumulation on the balloon surface.")

elif pred_idx == 6:
    print("Explanation: SHALLOW COLD LAYER (EARLY THERMODYNAMIC STRESS)")
    print("Analysis: Coldest layer occurs unusually low in the atmosphere.")
    print("Key Signals:")
    print(f"  - Min temperature altitude: {val_feat_clean['min_temp_alt_m'].values[0]:.1f} m")
    print(f"  - Min temperature: {val_feat_clean['min_temp_C'].values[0]:.1f} °C")
    print("Interpretation: Balloon experienced early cold stress before reaching typical high-altitude conditions.")

print("-" * 45)

print(f"--- Flight Vertical Stats ---")
print(f"Burst Altitude : {val_feat_clean['burst_alt_m'].values[0]:.2f} m")
print(f"Burst Pressure : {val_feat_clean['burst_pres_hpa'].values[0]:.2f} hPa")