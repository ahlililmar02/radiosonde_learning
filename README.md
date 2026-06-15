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

## Setup

```
uv sync
uv run python app.py
```

Then open http://127.0.0.1:5000/.

### Without uv

If `uv` isn't available, use `requirements.txt` with a plain venv:

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

## Retraining

Retraining pipeline (run in order whenever the training data changes):
`classifier.build_calib` -> `classifier.relabel_secondary` -> `classifier.train_secondary_model`.
