"""
train_secondary_model.py
─────────────────────────
Trains the secondary burst-cause classifier on the recalibrated
percentile-rank labels produced by classifier.ipynb
(combined_codes_train.csv / combined_codes_test.csv).

Does NOT touch classifier/final_classifier.py or classifier/models/ — this script
builds a separate, self-contained model bundle in classifier/models/ for
the new secondary-cause taxonomy (RH Freeze-out, Strong Shear, Cold Point
Tropopause, Pressure Reversal, Slow Ascent, Cloud/Rain Layer, Deep
Convection, ...).

Usage:
  uv run python classifier/train_secondary_model.py
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV

from classifier.features import engineer_features, fill_missing_features, FEATURE_COLS, MISSING, PROFILE_COLS


def clean_profile(flight_df: pd.DataFrame) -> pd.DataFrame:
    f = flight_df.copy()
    for col in PROFILE_COLS:
        if col in f.columns:
            f[col] = pd.to_numeric(f[col], errors='coerce')
            f[col] = f[col].replace(MISSING, np.nan)
    return f.sort_values('pressure_hPa', ascending=False).reset_index(drop=True)


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
    (classifier/secondary_classifier.py) — is_premature is shown as a display
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
    out_dir = Path('classifier/models')
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, out_dir / 'secondary_classifier_model.joblib')
    joblib.dump(scaler, out_dir / 'secondary_classifier_scaler.joblib')
    joblib.dump(le, out_dir / 'secondary_label_encoder.joblib')
    joblib.dump(FEATURE_COLS, out_dir / 'feature_cols.joblib')

    print(f"\nSaved model bundle to {out_dir}/")


if __name__ == '__main__':
    main()
