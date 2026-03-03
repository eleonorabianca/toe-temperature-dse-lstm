# Paper-LSTM (clean public code)

This repository contains cleaned, modular code for training and evaluating an LSTM model that predicts big toe temperature time-series.

## Data
Raw experimental data are **not included** for privacy reasons.
The loader expects folders like:

```
<BASE_DATA_PATH>/
  still_10/
    dfs_*.xlsx
  moving_10/
    dfs_*.xlsx
  ...
```

Each `.xlsx` must include columns:
- `SKIN`
- `BIG TOE`

## Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train (example)
Edit `BASE_DATA_PATH` in `examples/train.py`, then run:
```bash
python examples/train.py
```

Outputs (models, scalers, logs) are saved under `logs/<timestamp>/`.
