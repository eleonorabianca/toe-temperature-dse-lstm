# LSTM-Based Prediction of Big Toe Skin Temperature  
## Safety-Oriented Modeling for Cold Exposure Assessment

This repository provides the full training and evaluation pipeline for an LSTM-based model designed to predict big toe skin temperature during controlled cold exposure experiments.

The model integrates experimental time-series data, environmental conditions, and footwear thermal insulation parameters to predict:

- The full time-series of big toe temperature
- The time required to reach a safety-relevant threshold (15°C)

This code is released to ensure transparency and reproducibility of the methodology described in the associated publication.

---

## Scientific Context

Toe skin temperature is a critical parameter in cold injury prevention and thermal comfort assessment.  
A threshold of **15°C** is commonly considered a safety-relevant limit for cold exposure.

To align model optimization with safety assessment, the loss function includes a dedicated component that penalizes errors in predicted threshold-crossing time.

---

## Model Architecture

Input features (per timestep):

1. Skin temperature  
2. Activity level  
3. Ambient temperature (°C)  
4. Footwear thermal insulation (RcT)

Neural network structure:

- LSTM layer (384 units, return sequences)
- LSTM layer (256 units, return sequences)
- Dropout (0.05)
- Dense (80 units, tanh)
- Dense (1 unit, linear output)

Loss function:

Loss function:

$$
\text{Loss} = MSE_{\text{series}} + \alpha \cdot MSE_{\text{threshold}}
$$

Where:

- $MSE_{\text{series}}$ = reconstruction error of the full temperature time-series  
- $MSE_{\text{threshold}}$ = error in predicted time-to-15°C  
- $\alpha$ = weighting factor (default = 0.4) \(\alpha\) = weighting factor (default = 0.4)

---

## Repository Structure
paper-lstm/

├── src/
│ ├── utils/
│ ├── data/
│ ├── train/
│ └── eval/
├── examples/
│ └── train.py
├── requirements.txt
├── LICENSE
└── README.md

## Installation

It is recommended to use a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

## Training

Edit the base_path variable here to point to the directory where the data is stored.

'''
examples/train.py
''' 

Each Excel file should contain the following columns:

- 'SKIN'
- 'BIG TOE'

Then run:

```
python examples/train.py
```

Outputs include:

- Trained model (.keras)
- Input/output scalers
- Training logs (CSV)
- TensorBoard logs
- Threshold comparison plots

## License

This project is released under the MIT License.
