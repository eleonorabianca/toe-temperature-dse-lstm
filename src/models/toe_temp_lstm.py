"""Model architectures for toe temperature prediction."""

from __future__ import annotations

from typing import Sequence, Tuple

import tensorflow as tf


def build_toe_temp_lstm(
    input_timesteps: int,
    input_features: int,
    lstm_units: Sequence[int] = (384, 256),
    dense_units: int = 80,
    dropout_rate: float = 0.05,
    name: str = "ToeTempLSTM",
) -> tf.keras.Model:
    """Build the LSTM model used in the paper.

    Architecture (default):
        Input: (T, 4)
        LSTM(384, return_sequences=True)
        LSTM(256, return_sequences=True)
        Dropout(0.05)
        Dense(80, tanh)
        Dense(1, linear)

    Args:
        input_timesteps: Number of timesteps (e.g. 420).
        input_features: Number of features (e.g. 4).
        lstm_units: Units per LSTM layer.
        dense_units: Units in intermediate dense layer.
        dropout_rate: Dropout rate after LSTM stack.
        name: Model name.

    Returns:
        A compiled-uncompiled Keras Model (compile in training code).
    """
    inputs = tf.keras.Input(shape=(input_timesteps, input_features), name="inputs")
    x = inputs
    for i, units in enumerate(lstm_units):
        x = tf.keras.layers.LSTM(
            units,
            activation="tanh",
            return_sequences=True,
            name=f"lstm_{i}",
        )(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="dropout")(x)
    x = tf.keras.layers.Dense(dense_units, activation="tanh", name="dense_hidden")(x)
    outputs = tf.keras.layers.Dense(1, activation="linear", name="output")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)
