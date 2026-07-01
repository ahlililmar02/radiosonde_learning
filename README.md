# Radiosonde Burst Classification v2

## Overview

A machine-learning web app that classifies *why* a radiosonde (weather balloon) burst
prematurely, based on its vertical atmospheric profile. Users upload a radiosonde
BUFR file; the app checks whether the balloon reached the WMO 30 hPa "nominal"
ceiling, and if it burst earlier, predicts the most likely secondary cause (e.g.
strong/directional wind shear, RH freeze-out, deep convection, slow ascent, cold-point
tropopause).

## Features

- Upload a radiosonde `.bfr` file via the web interface
- End-to-end pipeline: parse -> engineer features -> scale -> RandomForest prediction
- Per-prediction SHAP evidence ("why this prediction") alongside a fixed evidence
  checklist
- An independent rule-based trace overlay for cross-checking the ML prediction

## Requirements

- Python 3.13
- `uv` (recommended) or `pip`

## Setup

```
uv sync
uv run python app.py
```

Then open http://127.0.0.1:5000/.

### Without uv

If `uv` isn't available, use a plain venv:

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

## CLI usage

Run the inference pipeline directly on a BUFR file without starting the web app:

```
uv run python -m classifier.secondary_classifier --file data/<file>.bfr
uv run python -m classifier.secondary_classifier --file data/<file>.bfr --save_json result.json
```

## Retraining

Requires `data/rason_complete.csv` (not tracked in git — 875 MB, obtain separately).
Run in order whenever the training data changes:

```
uv run python -m classifier.build_calib
uv run python -m classifier.relabel_secondary
uv run python -m classifier.train_secondary_model
```
