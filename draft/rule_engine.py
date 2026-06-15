"""
rule_engine.py
───────────────
Ports the percentile-rank labelling rules from classifier.ipynb
(compute_raw_signals / percentile_rank / classify_secondary_multi) so that
draft/secondary_classifier.py can show, for a single uploaded flight, *how*
the rule engine would have labelled it — purely as an analysis/explanation
overlay alongside the trained RandomForest prediction.

This module does NOT train anything. CALIB (the percentile-rank reference
arrays) is computed ONCE from the training flights by
draft/build_calib.py and saved to draft/models_v2/calib.joblib; this module
just loads that file and applies the frozen rules to a new flight.
"""

import numpy as np
import pandas as pd
import metpy.calc as mpcalc
from metpy.units import units


RANK_HI = 0.75
RANK_EXTREME = 0.90
RANK_LO = 0.25
RANK_LO_EXTREME = 0.10


def compute_cape_cin(flight_df: pd.DataFrame) -> dict:
    """Surface-based CAPE/CIN (J/kg) and LCL height (m above ground) via
    MetPy parcel-theory (mpcalc.surface_based_cape_cin). Returns NaNs if the
    profile is too short or the calculation fails (e.g. non-monotonic
    pressure, all-NaN levels)."""
    sub = flight_df[["pressure_hPa", "temp_C", "dewpoint_C", "height_m"]].dropna()
    sub = sub.sort_values("pressure_hPa", ascending=False).drop_duplicates(subset="pressure_hPa")

    if len(sub) < 4:
        return dict(cape_jkg=np.nan, cin_jkg=np.nan, lcl_agl_m=np.nan)

    try:
        p  = sub["pressure_hPa"].values * units.hPa
        T  = sub["temp_C"].values * units.degC
        Td = sub["dewpoint_C"].values * units.degC

        cape, cin = mpcalc.surface_based_cape_cin(p, T, Td)

        lcl_p, _ = mpcalc.lcl(p[0], T[0], Td[0])
        lcl_h = np.interp(lcl_p.magnitude, sub["pressure_hPa"].values[::-1], sub["height_m"].values[::-1])
        lcl_agl = float(lcl_h - sub["height_m"].iloc[0])

        return dict(
            cape_jkg=float(cape.magnitude),
            cin_jkg=float(cin.magnitude),
            lcl_agl_m=lcl_agl,
        )
    except Exception:
        return dict(cape_jkg=np.nan, cin_jkg=np.nan, lcl_agl_m=np.nan)


def compute_raw_signals(flight_df: pd.DataFrame) -> dict:
    """Per-flight raw signals used by the secondary-cause rules."""
    tail    = flight_df.tail(30)
    burst   = flight_df.iloc[-1]
    max_h   = flight_df["height_m"].max()
    top_500 = flight_df[flight_df["height_m"] > max_h - 500]

    dewdep_burst = burst["temp_C"] - burst["dewpoint_C"]

    warm = flight_df[flight_df["temp_C"] > -10]
    min_dewdep_warm = (warm["temp_C"] - warm["dewpoint_C"]).min() if not warm.empty else np.nan

    high_rh = flight_df[(flight_df["temp_C"] - flight_df["dewpoint_C"]) < 2]
    n_high_rh = len(high_rh)

    ascent_drop = tail["ascent_rate_mps"].diff().min()

    # Many BUFR profiles only have a single reported level within the last
    # 500m before burst (sparse reporting near termination), which leaves
    # spd_max/dir_delta undefined for ~half of flights. Fall back to the
    # last 5 levels of the whole profile in that case.
    near_burst = top_500 if len(top_500) > 1 else flight_df.tail(5)

    if len(near_burst) > 1:
        dir_delta = near_burst["wind_dir_deg"].diff().abs().max()
        spd_max   = near_burst["wind_speed_mps"].max()
    else:
        dir_delta = np.nan
        spd_max   = np.nan

    peak_ascent = tail["ascent_rate_mps"].max()
    med_ascent  = flight_df["ascent_rate_mps"].median()

    strato = flight_df[flight_df["height_m"] > 10000]
    if not strato.empty:
        min_temp_h = strato.loc[strato["temp_C"].idxmin(), "height_m"]
        cpt_dist = max_h - min_temp_h
    else:
        cpt_dist = np.nan

    tail100 = flight_df.tail(100)
    pos_diffs = tail100["pressure_hPa"].diff()
    pos_diffs = pos_diffs[pos_diffs > 0]
    pressure_increase_total = pos_diffs.sum() if len(pos_diffs) else 0.0

    if "latitude" in flight_df.columns:
        gps_frozen = (tail["latitude"].nunique() <= 1) or tail["latitude"].isna().all()
    else:
        gps_frozen = None  # not available (e.g. BUFR has no per-level lat/lon)

    cape_cin = compute_cape_cin(flight_df)

    # max_shear: same definition as engineer_features() in
    # train_secondary_model.py / secondary_classifier.py — wind-speed
    # change per 100m, over level gaps > 10m.
    dz   = flight_df["height_m"].diff().abs()
    dspd = flight_df["wind_speed_mps"].diff().abs()
    dz_safe = dz.replace(0, np.nan)
    shear = np.where(dz > 10, dspd / dz_safe * 100, np.nan)
    max_shear = np.nanmax(shear) if np.any(~np.isnan(shear)) else np.nan

    return dict(
        burst_temp=burst["temp_C"],
        dewdep_burst=dewdep_burst,
        min_dewdep_warm=min_dewdep_warm,
        n_high_rh=n_high_rh,
        ascent_drop=ascent_drop,
        dir_delta=dir_delta,
        spd_max=spd_max,
        peak_ascent=peak_ascent,
        med_ascent=med_ascent,
        cpt_dist=cpt_dist,
        pressure_increase_total=pressure_increase_total,
        gps_frozen=gps_frozen,
        cape_jkg=cape_cin["cape_jkg"],
        cin_jkg=cape_cin["cin_jkg"],
        lcl_agl_m=cape_cin["lcl_agl_m"],
        max_shear=max_shear,
    )


def percentile_rank(value, signal_name: str, calib: dict) -> float:
    """Fraction of the calibration population <= value. NaN-safe."""
    if pd.isna(value) or signal_name not in calib:
        return np.nan
    arr = calib[signal_name]
    if len(arr) == 0:
        return np.nan
    return float(np.searchsorted(arr, value, side="right") / len(arr))


def threshold_value(signal_name: str, rank: float, calib: dict) -> float:
    """Real-world value at a given percentile rank of the training
    distribution for signal_name (e.g. rank=0.75 -> the value at the 75th
    percentile). Used to show rule thresholds in real units instead of
    abstract percentile ranks."""
    if signal_name not in calib:
        return np.nan
    arr = calib[signal_name]
    if len(arr) == 0:
        return np.nan
    return float(np.percentile(arr, rank * 100))


def build_rule_trace(flight_df: pd.DataFrame, calib: dict) -> list[dict]:
    """
    Evaluate every secondary-cause rule against this flight and return a
    trace row per rule: raw signal value, percentile rank vs the training
    distribution (where applicable), the threshold used, whether it
    triggered, and the resulting score.

    This mirrors classify_secondary_multi() from classifier.ipynb but
    returns the full trace (including non-triggered rules) for display.
    """
    sig = compute_raw_signals(flight_df)
    trace = []

    # 1. RH Freeze-out — fixed physical constant, not percentile-based
    burst_temp = sig["burst_temp"]
    dewdep_burst = sig["dewdep_burst"]
    triggered = pd.notna(burst_temp) and burst_temp < -40
    score = float(np.clip(1.0 - dewdep_burst / 10, 0.0, 1.0)) if triggered else 0.0
    trace.append({
        "rule": "RH Freeze-out",
        "signal": "burst_temp",
        "value": _safe(burst_temp),
        "rank": None,
        "rule_desc": "burst temperature < -40 deg C",
        "triggered": bool(triggered),
        "score": round(score, 2),
    })

    # 2. RH Saturation — bottom quartile of min dewpoint depression (warm levels)
    rank = percentile_rank(sig["min_dewdep_warm"], "min_dewdep_warm", calib)
    triggered = pd.notna(rank) and rank <= RANK_LO
    dewdep_thr = threshold_value("min_dewdep_warm", RANK_LO, calib)
    trace.append({
        "rule": "RH Saturation",
        "signal": "min_dewdep_warm",
        "value": _safe(sig["min_dewdep_warm"]),
        "rank": _safe(rank),
        "rule_desc": f"min dewpoint depression (warm levels) <= {dewdep_thr:.2f} degC",
        "triggered": bool(triggered),
        "score": round(1.0 - rank, 2) if triggered else 0.0,
    })

    # 3. Cloud/Rain Layer — top quartile high-RH levels + bottom quartile ascent drop
    rh_rank = percentile_rank(sig["n_high_rh"], "n_high_rh", calib)
    drop_rank = percentile_rank(sig["ascent_drop"], "ascent_drop", calib)
    triggered = (
        pd.notna(rh_rank) and rh_rank >= RANK_HI
        and pd.notna(drop_rank) and drop_rank <= RANK_LO
    )
    n_high_rh_thr = threshold_value("n_high_rh", RANK_HI, calib)
    ascent_drop_thr = threshold_value("ascent_drop", RANK_LO, calib)
    trace.append({
        "rule": "Cloud/Rain Layer",
        "signal": "n_high_rh & ascent_drop",
        "value": f"{_safe(sig['n_high_rh'])} levels, drop={_safe(sig['ascent_drop'])}",
        "rank": f"rh_rank={_safe(rh_rank)}, drop_rank={_safe(drop_rank)}",
        "rule_desc": f"high-humidity levels >= {n_high_rh_thr:.0f} AND ascent rate drop <= {ascent_drop_thr:.2f} m/s",
        "triggered": bool(triggered),
        "score": round(rh_rank, 2) if triggered else 0.0,
    })

    # 4. Strong Shear — top quartile wind speed OR top decile direction change
    spd_rank = percentile_rank(sig["spd_max"], "spd_max", calib)
    dir_rank = percentile_rank(sig["dir_delta"], "dir_delta", calib)
    triggered = (
        (pd.notna(spd_rank) and spd_rank >= RANK_HI)
        or (pd.notna(dir_rank) and dir_rank >= RANK_EXTREME)
    )
    score = max(spd_rank if pd.notna(spd_rank) else 0.0,
                 dir_rank if pd.notna(dir_rank) else 0.0)
    spd_max_hi_thr = threshold_value("spd_max", RANK_HI, calib)
    dir_delta_extreme_thr = threshold_value("dir_delta", RANK_EXTREME, calib)
    trace.append({
        "rule": "Strong Shear",
        "signal": "spd_max & dir_delta",
        "value": f"spd_max={_safe(sig['spd_max'])}, dir_delta={_safe(sig['dir_delta'])}",
        "rank": f"spd_rank={_safe(spd_rank)}, dir_rank={_safe(dir_rank)}",
        "rule_desc": f"max wind speed near burst >= {spd_max_hi_thr:.1f} m/s OR direction change >= {dir_delta_extreme_thr:.0f} deg",
        "triggered": bool(triggered),
        "score": round(score, 2) if triggered else 0.0,
    })

    # 5. Directional Shear — top decile direction change WITHOUT high speed
    triggered = (
        pd.notna(dir_rank) and dir_rank >= RANK_EXTREME
        and pd.notna(spd_rank) and spd_rank < 0.5
    )
    spd_max_med_thr = threshold_value("spd_max", 0.5, calib)
    trace.append({
        "rule": "Directional Shear",
        "signal": "dir_delta & spd_max",
        "value": f"dir_delta={_safe(sig['dir_delta'])}, spd_max={_safe(sig['spd_max'])}",
        "rank": f"dir_rank={_safe(dir_rank)}, spd_rank={_safe(spd_rank)}",
        "rule_desc": f"direction change >= {dir_delta_extreme_thr:.0f} deg AND max wind speed near burst < {spd_max_med_thr:.1f} m/s",
        "triggered": bool(triggered),
        "score": round(dir_rank, 2) if triggered else 0.0,
    })

    # 6. Deep Convection — top decile CAPE (surface-based, MetPy parcel theory)
    cape_rank = percentile_rank(sig["cape_jkg"], "cape_jkg", calib)
    triggered = pd.notna(cape_rank) and cape_rank >= RANK_EXTREME
    cape_thr = threshold_value("cape_jkg", RANK_EXTREME, calib)
    trace.append({
        "rule": "Deep Convection",
        "signal": "cape_jkg",
        "value": _safe(sig["cape_jkg"]),
        "rank": _safe(cape_rank),
        "rule_desc": f"surface-based CAPE >= {cape_thr:.0f} J/kg",
        "triggered": bool(triggered),
        "score": round(cape_rank, 2) if triggered else 0.0,
    })

    # 7. Slow Ascent — bottom decile median ascent rate
    med_rank = percentile_rank(sig["med_ascent"], "med_ascent", calib)
    triggered = pd.notna(med_rank) and med_rank <= RANK_LO_EXTREME
    med_ascent_thr = threshold_value("med_ascent", RANK_LO_EXTREME, calib)
    trace.append({
        "rule": "Slow Ascent",
        "signal": "med_ascent",
        "value": _safe(sig["med_ascent"]),
        "rank": _safe(med_rank),
        "rule_desc": f"median ascent rate <= {med_ascent_thr:.2f} m/s",
        "triggered": bool(triggered),
        "score": round(1.0 - med_rank, 2) if triggered else 0.0,
    })

    # 8. Cold Point Tropopause — flight ended close to the cold-point altitude
    cpt_rank = percentile_rank(sig["cpt_dist"], "cpt_dist", calib)
    triggered = pd.notna(cpt_rank) and cpt_rank <= RANK_LO
    cpt_dist_thr = threshold_value("cpt_dist", RANK_LO, calib)
    trace.append({
        "rule": "Cold Point Tropopause",
        "signal": "cpt_dist",
        "value": _safe(sig["cpt_dist"]),
        "rank": _safe(cpt_rank),
        "rule_desc": f"distance to cold-point tropopause <= {cpt_dist_thr:.0f} m",
        "triggered": bool(triggered),
        "score": round(1.0 - cpt_rank, 2) if triggered else 0.0,
    })

    # 9. Pressure Reversal — top quartile cumulative pressure increase near termination
    pr_rank = percentile_rank(sig["pressure_increase_total"], "pressure_increase_total", calib)
    triggered = pd.notna(pr_rank) and pr_rank >= RANK_HI
    pressure_increase_thr = threshold_value("pressure_increase_total", RANK_HI, calib)
    trace.append({
        "rule": "Pressure Reversal",
        "signal": "pressure_increase_total",
        "value": _safe(sig["pressure_increase_total"]),
        "rank": _safe(pr_rank),
        "rule_desc": f"cumulative pressure increase near termination >= {pressure_increase_thr:.2f} hPa",
        "triggered": bool(triggered),
        "score": round(pr_rank, 2) if triggered else 0.0,
    })

    # 10. GPS Fail — lat/lon frozen or missing in the tail
    gps_frozen = sig["gps_frozen"]
    trace.append({
        "rule": "GPS Fail",
        "signal": "gps_frozen",
        "value": "n/a (no per-level lat/lon in BUFR)" if gps_frozen is None else bool(gps_frozen),
        "rank": None,
        "rule_desc": "latitude frozen or missing over last 30 levels",
        "triggered": bool(gps_frozen) if gps_frozen is not None else False,
        "score": 0.9 if gps_frozen else 0.0,
    })

    return trace


def _safe(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (np.floating, np.integer)):
        v = v.item()
    if isinstance(v, float):
        return round(v, 3)
    return v
