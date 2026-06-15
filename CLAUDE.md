# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A machine-learning web app that classifies *why* a radiosonde (weather balloon) burst
prematurely, based on its vertical atmospheric profile. Given a sounding file, the
pipeline checks whether the balloon reached the WMO 30 hPa "nominal" ceiling; if it
burst earlier ("premature"), a trained RandomForest classifier predicts the most
likely secondary cause (strong/directional shear, RH freeze-out, deep convection,
slow ascent, cold-point tropopause, etc.), explains the prediction with SHAP, and
cross-checks it against an independent percentile-rank rule engine.

This `v2` branch contains only the files needed to run and retrain the **v2 secondary
classifier** — the v1 primary classifier (`draft/final_classifier.py`,
`draft/models/`, `templates/index.html`, the OPTICS/UMAP/XGBoost notebooks) has been
removed.

## Environment & Commands

- Python 3.13, managed with `uv` (`pyproject.toml` + `uv.lock`, both tracked).
  `uv sync` installs the environment.
- `eccodes` (ECMWF BUFR decoding library) is required for parsing `.bfr` radiosonde
  files.
- Run the Flask app: `uv run python app.py` (serves at http://127.0.0.1:5000, `/`
  and `/v2` both render the v2 UI, debug mode on).
- Run the inference pipeline directly on a BUFR file:
  ```
  uv run python -m draft.secondary_classifier --file data/<file>.bfr [--model_dir draft/models_v2] [--save_json out.json]
  ```
- No automated test suite exists.

## Architecture

### Inference path (what `app.py` calls)
- `app.py` — Flask app. `/` and `/v2` render `templates/index_v2.html`;
  `/analyze_v2` accepts an uploaded radiosonde file, saves it to `uploads/`, calls
  `draft.secondary_classifier.run_pipeline()`, returns JSON, then deletes the upload.
- `draft/secondary_classifier.py` — self-contained end-to-end pipeline (its own
  `extract_radiosonde_data` / `clean_profile` / `engineer_features` /
  `fill_missing_features`, duplicated from the training scripts so inference doesn't
  depend on them at runtime), callable as a script or via
  `run_pipeline(file_path, model_dir="draft/models_v2")`:
  1. Parse the BUFR file, clean the profile, derive `ascent_rate_mps`.
  2. `detect_premature_burst` — applies the **30 hPa WMO threshold**
     (`PREMATURE_THRESHOLD_HPA`). If the balloon reached ≤30 hPa, the flight is
     "nominal" and the classifier is skipped.
  3. `engineer_features` — collapses the vertical profile into the 40-feature vector
     (`draft/models_v2/feature_cols.joblib`) used at training time. **Must stay
     identical to `engineer_features()` in `draft/train_secondary_model.py`** — any
     drift here silently breaks predictions.
  4. `fill_missing_features` — fills NaNs with training-time defaults.
  5. Loads `draft/models_v2/secondary_classifier_model.joblib` +
     `..._scaler.joblib` + `..._label_encoder.joblib` + `feature_cols.joblib`, scales
     the feature vector, predicts a cause, and computes class probabilities.
  6. `shap.TreeExplainer` computes per-prediction SHAP values — the top 5
     contributing features (with direction "supports"/"opposes" the predicted class)
     are returned as `shap_evidence`, replacing the old fixed
     `EVIDENCE_THRESHOLDS`-only explanation.
  7. `generate_explanation` — turns the prediction into a readable report using
     `CAUSE_DESCRIPTIONS` and `EVIDENCE_THRESHOLDS` (a small fixed checklist, display
     only — not the model's actual decision basis).
  8. `build_rule_trace()` (from `draft/rule_engine.py`) is run as a separate analysis
     overlay — 10 percentile-rank heuristic rules over raw profile signals — and
     returned as `rule_trace`. It does **not** feed back into the ML prediction; the
     two can legitimately disagree.
  9. The predicted cause is mapped to a `combined_code` via
     `draft/models_v2/combined_code_map.joblib` (falls back to
     `COMBINED_CODE_MAP_FALLBACK` in `secondary_classifier.py` if the artifact is
     missing).

### `draft/rule_engine.py` — shared rule/signal logic
- `compute_raw_signals()` — per-flight raw signals (max shear, wind shear near burst,
  CAPE/CIN/LCL via MetPy parcel theory, cold-point tropopause distance, moisture
  signals, etc.).
- `percentile_rank()` / `threshold_value()` — map a raw value to/from its percentile
  in the training distribution (`draft/models_v2/calib.joblib`).
- `build_rule_trace()` — evaluates all 10 secondary-cause rules, used both by the
  inference-time analysis overlay and by `relabel_secondary.py` for training labels.

### Training / retraining pipeline
Run in this order whenever `data/rason_complete.csv` is refreshed:
1. `uv run python -m draft.build_calib` — rebuilds `draft/models_v2/calib.joblib`
   (percentile-rank reference distributions for `rule_engine.py`, computed from
   `data/rason_complete.csv`).
2. `uv run python -m draft.relabel_secondary` — re-runs `build_rule_trace()` over
   every flight to regenerate `combined_codes_train.csv` / `combined_codes_test.csv`
   (reusing the existing train/test flight-id split) and
   `draft/models_v2/combined_code_map.joblib` (secondary cause → combined-code
   frequency table, used by `secondary_classifier.py` at inference time).
3. `uv run python -m draft.train_secondary_model` — trains the RandomForest
   (`class_weight='balanced'`, GridSearchCV) on `data/rason_complete.csv` +
   `combined_codes_train.csv`/`combined_codes_test.csv`, writing
   `draft/models_v2/secondary_classifier_model.joblib`, `..._scaler.joblib`,
   `..._label_encoder.joblib`, and `feature_cols.joblib`.

### Data
- `data/rason_complete.csv` — flattened flight level/metadata dataset (not tracked in
  git — 875MB; regenerate locally or obtain separately). Required for all three
  training scripts above.
- `Label_End_of_radiosonde.xlsx` — the primary/secondary → combined-code lookup table
  used by `relabel_secondary.py`.
- `combined_codes_train.csv` / `combined_codes_test.csv` — the flight-id train/test
  split with secondary-cause labels; regenerated in place by `relabel_secondary.py`.

### Key invariant
The 30 hPa nominal/premature threshold (`PREMATURE_THRESHOLD_HPA`), the
`engineer_features()` feature set/order, and the `max_shear`/`spd_max`/`dir_delta`
definitions in `rule_engine.compute_raw_signals()` must stay in sync across
`draft/secondary_classifier.py`, `draft/train_secondary_model.py`, and
`draft/rule_engine.py`. Changing one without the others will desync
`draft/models_v2/*.joblib` from the inference and rule-trace code.
