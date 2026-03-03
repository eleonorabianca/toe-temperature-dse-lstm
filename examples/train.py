"""Example end-to-end training script.

Edit BASE_DATA_PATH to point to your local/private dataset directory.
Do NOT commit data to the public repository.
"""

from pathlib import Path
import numpy as np

from src.utils.env_setup import configure_cuda, set_seed
from src.data.load_data import load_experiment_sequences
from src.train.training import (
    pick_one_test_per_folder,
    clip_by_percentile,
    scale_series_and_output,
    train_model_pipeline,
)
from src.eval.evaluate import time_vector, time_to_threshold, plot_threshold_bars
import joblib
import tensorflow as tf


# --- user settings ---
BASE_DATA_PATH = Path("/path/to/your/private/data")  # <-- change locally
USE_CPU = False
SEED = 42

folders = {
    "still_10": {"activity": 0, "ambient_temp": -10, "RcT": 1.31},
    "still_17": {"activity": 0, "ambient_temp": -17, "RcT": 1.31},
    "moving_10": {"activity": 1, "ambient_temp": -10, "RcT": 1.31},
    "moving_17": {"activity": 1, "ambient_temp": -17, "RcT": 1.31},
    "alpinism_17": {"activity": 2, "ambient_temp": -15, "RcT": 1.31},
    "alpinism_20": {"activity": 2, "ambient_temp": -20, "RcT": 1.86},
    "movingrib_13": {"activity": 1, "ambient_temp": -13, "RcT": 1.27},
    "movingrush_13": {"activity": 1, "ambient_temp": -13, "RcT": 1.23},
    "stillrib_13": {"activity": 0, "ambient_temp": -13, "RcT": 1.27},
    "stillrush_13": {"activity": 0, "ambient_temp": -13, "RcT": 1.23},
}

def main():
    configure_cuda(use_cpu=USE_CPU)
    set_seed(SEED)

    X, Y, meta = load_experiment_sequences(BASE_DATA_PATH, folders, anonymize=True, pad_to_maxlen=True)
    print("Loaded:", X.shape, Y.shape)

    X_train, Y_train, X_test, Y_test, test_idx = pick_one_test_per_folder(X, Y, meta, folders, random_seed=SEED)

    X_train = clip_by_percentile(X_train, upper_pct=90.0)
    Y_train = clip_by_percentile(Y_train, upper_pct=90.0)

    X_scaled, Y_scaled, scaler_series, scaler_output = scale_series_and_output(X_train, Y_train, series_col_idx=0)
    if Y_scaled.ndim == 2:
        Y_scaled = np.expand_dims(Y_scaled, axis=-1)

    model, info = train_model_pipeline(
        X_scaled, Y_scaled, scaler_series, scaler_output,
        output_dir="logs", epochs=200, batch_size=8,
        alpha=0.4, threshold_value_real=15.0
    )

    # Quick threshold evaluation on test samples
    tvec = time_vector(70.0, X.shape[1])
    results = []
    for idx in test_idx:
        xseq = X[idx]
        y_meas = Y[idx].reshape(-1)

        # scale xseq series only (same as training) and predict
        xseq_scaled = np.array(xseq, copy=True)
        s = scaler_series.transform(xseq_scaled[:, 0].reshape(-1, 1)).reshape(-1)
        xseq_scaled[:, 0] = s

        y_pred_scaled = model.predict(xseq_scaled[np.newaxis, :, :]).reshape(-1, 1)
        y_pred = scaler_output.inverse_transform(y_pred_scaled).reshape(-1)

        pred_t = time_to_threshold(y_pred, threshold=15.0, time_vec=tvec)
        meas_t = time_to_threshold(y_meas, threshold=15.0, time_vec=tvec)
        results.append((meta[idx]["original_folder"], float(pred_t), float(meas_t)))

    plot_threshold_bars(results)
    print(results)

if __name__ == "__main__":
    main()
