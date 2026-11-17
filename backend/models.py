"""Trained models: file names, loading, the probability calibrator and the clinical alert metrics."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from backend.config import Config


def artifact_path(cfg: Config, model: str, feature_set: str, horizon: int, seed: int) -> Path:
    """trained_models/{model}_{set}_{horizon}h_s{seed}.joblib"""
    return cfg.path("trained_models") / f"{model}_{feature_set}_{horizon}h_s{seed}.joblib"


def calib_path(art_path: Path) -> Path:
    return art_path.with_suffix(".calib.joblib")


def load_artifact(path: Path) -> dict:
    """Dict with the fitted sklearn/XGBoost pipeline under "model" and its column order under "feature_cols"."""
    return joblib.load(path)


def load_calibrator(art_path: Path) -> dict | None:
    p = calib_path(art_path)
    return joblib.load(p) if p.exists() else None


@dataclass
class Calibrator:
    """Monotone map from raw model probability to calibrated probability."""
    method: str
    model: object

    def predict(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        if self.method == "isotonic":
            return self.model.predict(p)
        z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
        return self.model.predict_proba(z[:, None])[:, 1]


def clinical_metrics(df_pred: pd.DataFrame, onset: pd.Series, threshold: float) -> dict:
    """Patient-level alert metrics at one threshold.

    df_pred: columns patient_id, hour, p (one row per scored hour, pre-onset only).
    onset:   patient_id -> onset hour (missing for non-septic patients).
    An alert is an hour with p >= threshold. Lead time = onset - first alert hour, over the
    septic patients alerted before onset, so read it together with the detected fraction.
    """
    d = df_pred[["patient_id", "hour", "p"]].copy()
    d["onset"] = d["patient_id"].map(onset)
    d["alert"] = d["p"] >= threshold
    septic = d["onset"].notna()
    first = d[d.alert & septic].groupby("patient_id")["hour"].min()
    lead = onset.reindex(first.index) - first
    n_septic = d.loc[septic, "patient_id"].nunique()
    patient_days = len(d) / 24.0
    true_alerts = int((d.alert & septic).sum())
    false_alerts = int((d.alert & ~septic).sum())
    return {
        "median_lead_time_hours": float(lead.median()) if len(lead) else np.nan,
        "patients_detected_fraction": len(first) / n_septic if n_septic else np.nan,
        "alerts_per_patient_day": float(d.alert.sum() / patient_days) if patient_days else np.nan,
        "false_alarms_per_patient_day": float(false_alerts / patient_days) if patient_days else np.nan,
        "false_alarms_per_true_alarm": false_alerts / true_alerts if true_alerts else np.nan,
        "nonseptic_patients_alerted_fraction": float(d[~septic & d.alert].patient_id.nunique()
                                                     / max(d[~septic].patient_id.nunique(), 1)),
        "threshold": float(threshold),
    }
