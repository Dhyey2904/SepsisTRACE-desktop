"""Everything the GUI asks for: demo patients, stored and live predictions, SHAP shares, threshold metrics.

Every live prediction goes through features.build_features, the same feature code the models were trained with.
"""
from __future__ import annotations

import functools
import json

import numpy as np
import pandas as pd

from backend import config as C
from backend.features import build_features
from backend.models import artifact_path, clinical_metrics, load_artifact, load_calibrator

CFG = C.load()
MODELS = list(CFG.models.names)
FEATURE_SETS = list(CFG.models.feature_sets)
HORIZONS = list(CFG.data.horizons)
WHATIF_VARS = ["HR", "MAP", "Temp", "Lactate"]
PANEL_VITALS = ["HR", "MAP", "Temp", "Resp"]
DATA_FILES = ["demo_patients", "demo_predictions", "shap_cache", "test_predictions", "thresholds"]


def missing_files() -> list[str]:
    """Data or model files the app needs that are not on disk (empty list = ready to run)."""
    missing = [str(CFG.path(k).relative_to(C.ROOT)) for k in DATA_FILES if not CFG.path(k).exists()]
    if not any(CFG.path("trained_models").glob("*.joblib")):
        missing.append(f"{CFG.paths.trained_models}/*.joblib")
    return missing


def combo(model: str, feature_set: str, horizon: int) -> str:
    return f"{model}|{feature_set}|{horizon}"


# ==== loading (each file is read once, then kept in memory) =====================================

@functools.lru_cache(maxsize=1)
def load_demo() -> pd.DataFrame:
    """Hourly rows of the 30 demo patients (pre-onset), with category and onset."""
    return pd.read_parquet(CFG.path("demo_patients"))


@functools.lru_cache(maxsize=1)
def load_predictions() -> pd.DataFrame:
    return pd.read_parquet(CFG.path("demo_predictions"))


@functools.lru_cache(maxsize=1)
def load_test_predictions() -> pd.DataFrame:
    return pd.read_parquet(CFG.path("test_predictions"))


@functools.lru_cache(maxsize=1)
def load_thresholds() -> dict[str, float]:
    return json.loads(CFG.path("thresholds").read_text())


@functools.lru_cache(maxsize=1)
def load_shap_cache() -> pd.DataFrame:
    return pd.read_parquet(CFG.path("shap_cache"))


@functools.lru_cache(maxsize=1)
def load_test_onsets() -> pd.Series | None:
    """patient_id -> onset hour for septic test patients; None if the file is missing."""
    f = CFG.path("test_onsets")
    return pd.Series(json.loads(f.read_text()), dtype=float) if f.exists() else None


@functools.lru_cache(maxsize=48)
def load_model(model: str, feature_set: str, horizon: int) -> tuple[dict, object | None]:
    """(artifact dict, calibrator or None)."""
    path = artifact_path(CFG, model, feature_set, horizon, CFG.seed)
    cal = load_calibrator(path)
    return load_artifact(path), (cal["calibrator"] if cal else None)


# ==== patients and predictions ==================================================================

def patient(patient_id: str) -> pd.DataFrame:
    d = load_demo()
    return d[d.patient_id == patient_id].sort_values("hour").reset_index(drop=True)


def precomputed(patient_id: str, model: str, feature_set: str, horizon: int) -> pd.Series:
    """Stored calibrated predictions: hour -> p."""
    p = load_predictions()
    sel = p[(p.patient_id == patient_id) & (p.model == model) & (p.feature_set == feature_set) & (p.horizon == horizon)]
    return sel.set_index("hour")["p"].sort_index()


def predict_stay(patient_df: pd.DataFrame, model: str, feature_set: str, horizon: int,
                 calibrated: bool = True) -> pd.Series:
    """Live per-hour risk for one stay: hour -> p (hours >= min_history_hours)."""
    art, cal = load_model(model, feature_set, horizon)
    f = build_features(patient_df, CFG, feature_set)
    p = art["model"].predict_proba(f[art["feature_cols"]].to_numpy(np.float32))[:, 1]
    if calibrated and cal is not None:
        p = cal.predict(p)
    return pd.Series(p, index=f["hour"].to_numpy(), name="p")


def predict_with_hidden(patient_df: pd.DataFrame, hide_vars: list[str], model: str, feature_set: str,
                        horizon: int, calibrated: bool = True) -> pd.Series:
    """As predict_stay, with the listed variables never measured (all NaN)."""
    p = patient_df.copy()
    if hide_vars:
        p[list(hide_vars)] = np.nan
    return predict_stay(p, model, feature_set, horizon, calibrated)


def predict_with_edits(patient_df: pd.DataFrame, edits: dict[str, float], hour: int, model: str,
                       feature_set: str, horizon: int, calibrated: bool = True) -> float:
    """Risk at `hour` after overwriting values at that hour (an edit to NaN = not measured)."""
    p = patient_df[patient_df.hour <= hour].copy()
    for var, val in edits.items():
        p.loc[p.hour == hour, var] = val
    s = predict_stay(p, model, feature_set, horizon, calibrated)
    return float(s.loc[hour]) if hour in s.index else float("nan")


# ==== SHAP explanations (precomputed for XGBoost V+M) ===========================================

def _shap_row(patient_id: str, hour: int, model: str, horizon: int) -> pd.Series | None:
    c = load_shap_cache()
    sel = c[(c.patient_id == patient_id) & (c.hour == hour) & (c.model == model) & (c.horizon == horizon)]
    return None if sel.empty else sel.iloc[0]


def contribution_for(patient_id: str, hour: int, model: str = "xgb", horizon: int = 6) -> tuple[float, float] | None:
    """(share_V, share_M) of |SHAP| for this hour; None if not cached."""
    r = _shap_row(patient_id, hour, model, horizon)
    return None if r is None else (float(r.share_V), float(r.share_M))


def top_features_for(patient_id: str, hour: int, model: str = "xgb", horizon: int = 6) -> pd.DataFrame:
    r = _shap_row(patient_id, hour, model, horizon)
    rows = [] if r is None else [(r[f"top{k}_feature"], r[f"top{k}_phi"]) for k in range(1, 6)]
    return pd.DataFrame(rows, columns=["feature", "phi"])


# ==== threshold console =========================================================================

def default_threshold(model: str, feature_set: str, horizon: int) -> float:
    """Best-F1 threshold chosen on the validation split."""
    return float(load_thresholds()[combo(model, feature_set, horizon)])


def threshold_console(model: str, feature_set: str, horizon: int, threshold: float) -> dict:
    """Test-split operating point: hour-level confusion matrix plus clinical metrics.
    `onsets_available` is False (and the patient-level metrics NaN) if test_onsets.json is missing."""
    tp = load_test_predictions()
    p, y = tp[combo(model, feature_set, horizon)].to_numpy(), tp[f"y_{horizon}"].to_numpy()
    alert = p >= threshold
    cm = {"TP": int((alert & (y == 1)).sum()), "FP": int((alert & (y == 0)).sum()),
          "FN": int((~alert & (y == 1)).sum()), "TN": int((~alert & (y == 0)).sum())}
    onsets = load_test_onsets()
    if onsets is None:
        clin = dict.fromkeys(["patients_detected_fraction", "false_alarms_per_patient_day",
                              "median_lead_time_hours"], np.nan)
    else:
        pred = tp[["patient_id", "hour"]].assign(p=p)
        clin = clinical_metrics(pred, onsets.reindex(tp.patient_id.unique()).dropna(), threshold)
    sens = cm["TP"] / max(cm["TP"] + cm["FN"], 1)
    return {"confusion": cm, "hour_sensitivity": sens, "onsets_available": onsets is not None, **clin}
