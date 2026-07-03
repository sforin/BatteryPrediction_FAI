# BatteryPrediction_FAI

Early capacity prediction for Ni-MH battery cells using classical ML and deep learning models. The goal is to predict the final discharge capacity from the first 60–150 minutes of the discharge curve, enabling early pass/fail decisions without waiting for the full 14-hour test.

## Requirements

- Python 3.9+
- pip

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd BatteryPrediction_FAI
   ```

2. (Optional) Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS / Linux
   source .venv/bin/activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Notebook

Launch Jupyter and open the notebook:

```bash
jupyter notebook notebook.ipynb
```

Or with JupyterLab:

```bash
jupyter lab notebook.ipynb
```

Run all cells from top to bottom. All results are fully reproducible (`RANDOM_SEED = 42`).

## Project Structure

```
BatteryPrediction_FAI/
├── notebook.ipynb       # Main notebook
├── requirements.txt     # Python dependencies
└── data/
    └── raw_csvs/        # Raw battery discharge CSV files
```

## Dataset

Each CSV file in `data/raw_csvs/` represents a single galvanostatic discharge test for one Ni-MH cell, with columns for timestamp (`Tempo`), voltage (`Tensione (V)`), and cumulative capacity (`Capacity (Ah)`).