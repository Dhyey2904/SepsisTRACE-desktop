"""Features for one patient: causal 12-hour windows and the V / M / VM feature sets.

V = values, M = measurement pattern only, S = statics (included in every set).
Each hour t only ever uses rows up to t, so no future data leaks into a prediction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from backend.config import VARIABLE_COLUMNS, Config

V_STATS = ["mean", "std", "min", "max", "last", "slope"]
M_PER_VAR = ["count", "dt_log"]
M_GLOBAL = ["meas_rate", "meas_rate_delta", "hours_available", "n_vars_ever"]


# ==== column names ==============================================================================

def value_vars(cfg: Config) -> list[str]:
    return list(cfg.data.vitals) + list(cfg.data.labs_keep)


def feature_columns(cfg: Config, feature_set: str) -> list[str]:
    cols = list(cfg.data.statics)
    if "V" in feature_set:
        cols += [f"{v}_{s}" for v in value_vars(cfg) for s in V_STATS]
    if "M" in feature_set:
        cols += [f"{v}_{s}" for v in VARIABLE_COLUMNS for s in M_PER_VAR] + M_GLOBAL
    return cols


# ==== windows ===================================================================================

def causal_windows(X: np.ndarray, window_hours: int) -> np.ndarray:
    """(T, k) -> (T, W, k): element [t, j] is row t - (W-1) + j, NaN before the stay starts."""
    T, k = X.shape
    pad = np.full((window_hours - 1, k), np.nan, dtype=np.float64)
    Xp = np.vstack([pad, X.astype(np.float64, copy=False)])
    return sliding_window_view(Xp, window_hours, axis=0).transpose(0, 2, 1)


def trailing_sum(A: np.ndarray, length: int, lag: int = 0) -> np.ndarray:
    """(T, k) -> (T, k): sum of rows t-lag-length+1 .. t-lag (rows before 0 count as 0)."""
    cs = np.vstack([np.zeros((1, A.shape[1])), np.cumsum(A, axis=0)])
    T = A.shape[0]
    hi = np.arange(T) - lag + 1
    lo = hi - length
    return cs[np.clip(hi, 0, T)] - cs[np.clip(lo, 0, T)]


def _nan_stats(win: np.ndarray) -> tuple[np.ndarray, ...]:
    """win (T, W, k) -> n, mean, std(ddof=1), min, max over axis 1 ignoring NaN, without warnings."""
    ok = ~np.isnan(win)
    n = ok.sum(axis=1)
    z = np.where(ok, win, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = z.sum(axis=1) / n
        dev = np.where(ok, win - mean[:, None, :], 0.0)
        std = np.sqrt((dev ** 2).sum(axis=1) / (n - 1))
    std[n < 2] = np.nan
    mn = np.where(ok, win, np.inf).min(axis=1)
    mx = np.where(ok, win, -np.inf).max(axis=1)
    mn[n == 0] = np.nan
    mx[n == 0] = np.nan
    mean[n == 0] = np.nan
    return n, mean, std, mn, mx


# ==== feature blocks ============================================================================

def block_V(X: np.ndarray, window_hours: int) -> np.ndarray:
    """Values block. X (T, 16) raw values -> (T, 16*6).

    Per variable: mean, std, min, max, last over the window after forward-filling
    *within the window*; slope = least-squares slope of value on hour over observed points.
    """
    W = window_hours
    raw = causal_windows(X, W)                                  # (T, W, k)
    obs = ~np.isnan(raw)
    pos = np.arange(W)[None, :, None]
    idx = np.maximum.accumulate(np.where(obs, pos, -1), axis=1)
    ff = np.take_along_axis(raw, np.clip(idx, 0, None), axis=1)
    ff[idx < 0] = np.nan
    _, mean, std, mn, mx = _nan_stats(ff)
    last = ff[:, -1, :]

    n_o = obs.sum(axis=1)
    x = np.broadcast_to(pos, raw.shape).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        xm = np.where(obs, x, 0).sum(axis=1) / n_o
        ym = np.where(obs, raw, 0).sum(axis=1) / n_o
        dx = np.where(obs, x - xm[:, None, :], 0)
        dy = np.where(obs, raw - ym[:, None, :], 0)
        sxx = (dx ** 2).sum(axis=1)
        slope = (dx * dy).sum(axis=1) / sxx
    slope[(n_o < 2) | (sxx == 0)] = np.nan
    out = np.stack([mean, std, mn, mx, last, slope], axis=2)    # (T, k, 6)
    return out.reshape(len(X), -1)


def block_M(X: np.ndarray, hours: np.ndarray, window_hours: int, half: int) -> np.ndarray:
    """Measurement block. X (T, 40) full history, hours (T,) 1-based -> (T, 40*2 + 4).
    Uses only *whether* each value was measured, never the value itself."""
    W = window_hours
    T, k = X.shape
    obs = (~np.isnan(X)).astype(np.float64)
    count = trailing_sum(obs, W)                                # (T, k)
    row = np.arange(T)
    last = np.maximum.accumulate(np.where(obs > 0, row[:, None], -1), axis=0)
    ever = last >= 0
    hrs = hours.astype(np.float64)[:, None]
    last_hour = np.where(ever, np.take(hours, np.clip(last, 0, None)).astype(np.float64), 0.0)
    dt_log = np.log1p(hrs - last_hour)                         # never observed -> hours since admission
    per_var = np.stack([count, dt_log], axis=2).reshape(T, -1)

    meas_rate = count.sum(axis=1) / (W * k)
    recent = trailing_sum(obs, half).sum(axis=1) / (half * k)
    prev = trailing_sum(obs, half, lag=half).sum(axis=1) / (half * k)
    hours_available = np.minimum(hours, W).astype(np.float64)
    n_vars_ever = ever.sum(axis=1).astype(np.float64)
    glob = np.column_stack([meas_rate, recent - prev, hours_available, n_vars_ever])
    return np.hstack([per_var, glob])


def block_S(S: np.ndarray) -> np.ndarray:
    """Statics at t: last observed value up to t (forward fill over the stay)."""
    return pd.DataFrame(S).ffill().to_numpy(dtype=np.float64)


# ==== assembly ==================================================================================

def build_features(patient_df: pd.DataFrame, cfg: Config, feature_set: str) -> pd.DataFrame:
    """One row per hour t >= min_history_hours with columns patient_id, hour, <features>.

    patient_df: one patient's hourly rows, consecutive hours starting at 1.
    """
    p = patient_df.sort_values("hour")
    hours = p["hour"].to_numpy()
    assert (np.diff(hours) == 1).all(), "hours must be consecutive"
    W = cfg.data.window_hours
    parts = [block_S(p[list(cfg.data.statics)].to_numpy(np.float64))]
    if "V" in feature_set:
        parts.append(block_V(p[value_vars(cfg)].to_numpy(np.float64), W))
    if "M" in feature_set:
        parts.append(block_M(p[VARIABLE_COLUMNS].to_numpy(np.float64), hours, W, cfg.data.rate_half_window))
    X = np.hstack(parts).astype(np.float32)
    out = pd.DataFrame(X, columns=feature_columns(cfg, feature_set))
    out.insert(0, "hour", hours.astype(np.int16))
    out.insert(0, "patient_id", str(p["patient_id"].iloc[0]))
    out = out[out["hour"] >= cfg.data.min_history_hours]
    return out.reset_index(drop=True)
