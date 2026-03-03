"""Training pipeline: split, scaling, custom loss and training wrapper."""

from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import joblib
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

from src.models.toe_temp_lstm import build_toe_temp_lstm

ArrayLike = Union[np.ndarray, List[np.ndarray]]


def pick_one_test_per_folder(
    X: ArrayLike,
    Y: ArrayLike,
    meta_list: List[Dict[str, Any]],
    folders: Dict[str, Dict[str, Any]],
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int]]:
    """Pick one random test sequence per folder name."""
    random.seed(random_seed)
    np.random.seed(random_seed)

    folder_to_indices: Dict[str, List[int]] = {}
    for idx, meta in enumerate(meta_list):
        folder_name = meta.get("original_folder") or meta.get("anonymized_folder")
        folder_to_indices.setdefault(folder_name, []).append(idx)

    test_indices: List[int] = []
    for folder_name in folders.keys():
        candidates = folder_to_indices.get(folder_name, [])
        if not candidates:
            print(f"Warning: no candidates found for folder '{folder_name}', skipping.")
            continue
        test_indices.append(random.choice(candidates))

    all_indices = list(range(len(meta_list)))
    train_indices = [i for i in all_indices if i not in test_indices]

    def pick(arr: ArrayLike, idxs: List[int]) -> np.ndarray:
        if isinstance(arr, np.ndarray):
            return arr[idxs]
        return np.asarray([arr[i] for i in idxs], dtype=object)

    X_test = pick(X, test_indices)
    Y_test = pick(Y, test_indices)
    X_train = pick(X, train_indices)
    Y_train = pick(Y, train_indices)

    return X_train, Y_train, X_test, Y_test, test_indices


def clip_by_percentile(arr: np.ndarray, upper_pct: float = 90.0, lower_pct: float = 0.0) -> np.ndarray:
    """Clip an array to global percentiles to reduce extreme outliers."""
    if arr.size == 0:
        return arr
    lo = np.percentile(arr, lower_pct)
    hi = np.percentile(arr, upper_pct)
    return np.clip(arr, lo, hi)


def scale_series_and_output(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    series_col_idx: int = 0,
) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler, MinMaxScaler]:
    """Scale only the series column in X and scale Y (target)."""
    if not isinstance(X_train, np.ndarray) or X_train.ndim != 3:
        raise ValueError("scale_series_and_output expects padded numpy arrays with shape (N, T, F).")

    scaler_series = MinMaxScaler()
    series_flat = X_train[:, :, series_col_idx].reshape(-1, 1)
    series_scaled = scaler_series.fit_transform(series_flat).reshape(X_train[:, :, series_col_idx].shape)

    X_scaled = np.array(X_train, copy=True)
    X_scaled[:, :, series_col_idx] = series_scaled

    scaler_output = MinMaxScaler()
    Y_scaled = scaler_output.fit_transform(Y_train.reshape(-1, 1)).reshape(Y_train.shape)

    return X_scaled, Y_scaled, scaler_series, scaler_output


def compute_crossing_time_minutes(y_batch: tf.Tensor, threshold: float, total_time_minutes: float = 70.0) -> tf.Tensor:
    """First time (minutes) where y <= threshold, else total_time_minutes."""
    y = tf.convert_to_tensor(y_batch, dtype=tf.float32)
    if y.shape.rank == 3 and y.shape[-1] == 1:
        y = tf.squeeze(y, axis=-1)
    if y.shape.rank == 1:
        y = tf.expand_dims(y, axis=0)

    timesteps = tf.shape(y)[1]
    dt = tf.cast(total_time_minutes, tf.float32) / tf.cast(timesteps, tf.float32)

    crossed = tf.less_equal(y, tf.cast(threshold, tf.float32))
    indices = tf.cast(tf.range(timesteps), tf.float32)[tf.newaxis, :]
    sentinel = tf.cast(timesteps, tf.float32)

    masked = tf.where(crossed, indices, sentinel * tf.ones_like(indices))
    first_idx = tf.reduce_min(masked, axis=1)

    no_cross = tf.equal(first_idx, sentinel)
    t_minutes = first_idx * dt
    return tf.where(no_cross, tf.cast(total_time_minutes, tf.float32), t_minutes)


def threshold_component_loss(y_true: tf.Tensor, y_pred: tf.Tensor, threshold: float, total_time_minutes: float = 70.0) -> tf.Tensor:
    t_true = compute_crossing_time_minutes(y_true, threshold, total_time_minutes)
    t_pred = compute_crossing_time_minutes(y_pred, threshold, total_time_minutes)
    return tf.reduce_mean(tf.square(t_true - t_pred))


def make_custom_loss(alpha: float, threshold_normalized: float, total_time_minutes: float = 70.0):
    """Loss = MSE(y) + alpha * MSE(t_cross). All computed in scaled output space."""
    def loss_fn(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=[1, 2])
        mse = tf.reduce_mean(mse)
        thr = threshold_component_loss(y_true, y_pred, threshold_normalized, total_time_minutes)
        return mse + alpha * thr
    return loss_fn


def train_model_pipeline(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    scaler_series: MinMaxScaler,
    scaler_output: MinMaxScaler,
    output_dir: Union[str, Path] = "logs",
    epochs: int = 800,
    batch_size: int = 10,
    alpha: float = 0.4,
    threshold_value_real: float = 15.0,
    total_time_minutes: float = 70.0,
) -> Tuple[tf.keras.Model, Dict[str, Any]]:
    """Compile, fit, and save model + scalers + history."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    threshold_normalized = float(scaler_output.transform([[threshold_value_real]])[0, 0])
    loss_fn = make_custom_loss(alpha, threshold_normalized, total_time_minutes)

    model = build_toe_temp_lstm(
        input_timesteps=X_train.shape[1],
        input_features=X_train.shape[2],
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=loss_fn,
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = output_dir / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.CSVLogger(str(run_dir / "training_log.csv")),
        tf.keras.callbacks.TensorBoard(log_dir=str(run_dir / "tensorboard")),
        tf.keras.callbacks.ModelCheckpoint(
            str(run_dir / "best.weights.h5"),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            mode="min",
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            mode="min",
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=50,
            restore_best_weights=True,
            mode="min",
        ),
    ]

    history = model.fit(
        X_train,
        Y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(str(run_dir / "model.keras"))
    joblib.dump(scaler_output, str(run_dir / "scaler_output.save"))
    joblib.dump(scaler_series, str(run_dir / "scaler_series.save"))

    with open(run_dir / "history.json", "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f)

    return model, {"history": history, "run_dir": run_dir}
