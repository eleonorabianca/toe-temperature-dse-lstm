"""Evaluation utilities: scaling, prediction, threshold times, plots, CSV trial runner."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf


def scale_series_only(X: np.ndarray, scaler_series: MinMaxScaler, series_col_idx: int = 0) -> np.ndarray:
    """Scale only X[..., series_col_idx]. Accepts (T,F), (1,T,F), or (N,T,F)."""
    X_arr = np.array(X, copy=True)
    single = False
    if X_arr.ndim == 2:
        X_arr = X_arr[np.newaxis, :, :]
        single = True
    if X_arr.ndim != 3:
        raise ValueError(f"Unexpected X shape: {X_arr.shape}")

    series_flat = X_arr[:, :, series_col_idx].reshape(-1, 1)
    series_scaled = scaler_series.transform(series_flat).reshape(X_arr[:, :, series_col_idx].shape)

    X_scaled = X_arr.copy()
    X_scaled[:, :, series_col_idx] = series_scaled
    return X_scaled[0] if single else X_scaled


def predict_sequence(
    model: tf.keras.Model,
    X_seq: np.ndarray,
    scaler_series: MinMaxScaler,
    scaler_output: MinMaxScaler,
    series_col_idx: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict big toe time-series for a single sequence and inverse-transform to °C."""
    X_scaled = scale_series_only(X_seq, scaler_series, series_col_idx)
    X_input = X_scaled[np.newaxis, :, :] if X_scaled.ndim == 2 else X_scaled
    preds_scaled = model.predict(X_input)  # (batch, T, 1)
    preds_scaled = np.asarray(preds_scaled)

    if preds_scaled.ndim == 3 and preds_scaled.shape[0] == 1:
        preds_scaled_flat = preds_scaled[0, :, 0]
    else:
        preds_scaled_flat = preds_scaled.reshape(-1)

    preds_real = scaler_output.inverse_transform(preds_scaled_flat.reshape(-1, 1)).reshape(-1)
    return preds_real, preds_scaled_flat


def time_vector(total_minutes: float = 70.0, timesteps: int = 420) -> np.ndarray:
    return np.linspace(0.0, total_minutes, timesteps)


def time_to_threshold(values: np.ndarray, threshold: float = 15.0, time_vec: Optional[np.ndarray] = None) -> float:
    v = np.asarray(values).reshape(-1)
    t = time_vector(70.0, v.size) if time_vec is None else np.asarray(time_vec)
    below = np.where(v <= threshold)[0]
    if below.size:
        return float(t[below[0]])
    idx = int(np.argmin(np.abs(v - threshold)))
    return float(t[idx])


def plot_threshold_bars(results: Sequence[Tuple[str, float, float]], ylim: Tuple[float, float] = (0.0, 70.0)):
    labels = [r[0] for r in results]
    pred = [r[1] for r in results]
    meas = [r[2] for r in results]

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w/2, pred, w, label="Predicted", color="red")
    ax.bar(x + w/2, meas, w, label="Measured", color="black")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Time to reach 15°C (min)")
    ax.set_ylim(*ylim)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    plt.tight_layout()
    plt.show()


def run_trial_from_csv(
    csv_path: Union[str, Path],
    model: tf.keras.Model,
    scaler_series: MinMaxScaler,
    scaler_output: MinMaxScaler,
    physical_activity: int = 2,
    ambient_temp: float = -13.0,
    rct: float = 1.0,
    total_minutes: float = 70.0,
) -> Tuple[np.ndarray, float]:
    """Run inference for a single external SKIN series stored in a CSV file."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path, header=None)
    series = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
    if np.any(np.isnan(series)):
        series = np.nan_to_num(series, nan=np.nanmean(series))

    T = series.shape[0]
    cond = np.full((T, 3), [physical_activity, ambient_temp, rct], dtype=float)
    X_trial = np.hstack((series.reshape(-1, 1), cond))  # (T,4)

    preds_real, _ = predict_sequence(model, X_trial, scaler_series, scaler_output)
    tvec = time_vector(total_minutes=total_minutes, timesteps=T)
    t_thresh = time_to_threshold(preds_real, threshold=15.0, time_vec=tvec)
    return preds_real, t_thresh
