# Chlamydia-Forecasting_US
```
# Probabilistic Ensemble Forecasting of Chlamydia Trachomatis Incidence Across US States

This repository contains all code associated with the manuscript:
"A probabilistic ensemble for weekly forecasting of Chlamydia
trachomatis incidence in the USA" (submitted to PLOS Computational Biology).

---

## Overview

Weekly chlamydia incidence data from the CDC National Notifiable Diseases
Surveillance System (NNDSS) are used to train five base models — LightGBM,
XGBoost, CatBoost, LSTM, and GRU — under a biweekly rolling-window evaluation
protocol. Model predictions are aggregated into a weighted ensemble using
inverse Weighted Interval Score (WIS) weights derived from accumulated
out-of-sample performance.

---

## Repository Structure

| File | Description |
|------|-------------|
| `data_analysis.py` | Downloads raw NNDSS chlamydia data from the CDC API, computes cumulative case totals and zero-incidence week counts per state, and generates the bar plots used for state selection |
| `data_preprocessing.ipynb` | Reshapes raw data from wide to long format, computes cases per 100,000 population, engineers all features (lags, rolling statistics, expanding window stats, EWMA, cyclical encodings, holiday flag), and saves `preprocessed_No_OHE.csv` and `preprocessed_OHE.csv` |
| `dl_data_utils.py` | PyTorch dataset classes and data pipeline utilities for the recurrent models: sliding window construction, LabelEncoder for state identity, StandardScaler fitted on training data only, and `build_dataloaders` |
| `model_utils.py` | Shared utility functions: Huber loss, rolling window split generator, Optuna tuning functions for LightGBM/XGBoost/CatBoost, interval score, Weighted Interval Score, and inverse-WIS weight computation |
| `train_lgbm.py` | Trains LightGBM across all biweekly rolling cutoffs with 500-trial Optuna hyperparameter search; saves predictions to `outputs/lgbm_test_predictions.csv` |
| `train_catboost.py` | Trains CatBoost across all biweekly rolling cutoffs with 500-trial Optuna hyperparameter search; saves predictions to `outputs/catboost_test_predictions.csv` |
| `train_xgboost.py` | Trains XGBoost across all biweekly rolling cutoffs with 500-trial Optuna hyperparameter search; saves predictions to `outputs/xgboost_test_predictions.csv` |
| `train_lstm_rolling.py` | Trains LSTM across all biweekly rolling cutoffs (lookback=52 weeks, forecast window=2 weeks, hidden size=32, AdamW optimizer, early stopping patience=20); saves predictions to `outputs/lstm_test_predictions.csv` |
| `train_gru_rolling.py` | Identical pipeline to LSTM but using a GRU layer; saves predictions to `outputs/gru_test_predictions.csv` |
| `ensemble_predict.py` | Loads all five model prediction CSVs, computes state-specific inverse-WIS weights from accumulated out-of-sample test windows, and outputs weighted ensemble forecasts to `outputs/ensemble_test_predictions.csv` |

---

## Execution Order

Run scripts in the following order:

```
1. data_analysis.py
2. data_preprocessing.ipynb
3. train_lgbm.py
4. train_catboost.py
5. train_xgboost.py
6. train_lstm_rolling.py
7. train_gru_rolling.py
8. ensemble_predict.py
```

Steps 3–7 are independent of each other and can be run in parallel if
compute resources allow.

---

## Data

Chlamydia case counts are downloaded automatically from the CDC NNDSS
public API (data.cdc.gov) by `data_analysis.py`. No raw data files need
to be downloaded manually. Population estimates use 2023 US Census Bureau
figures, which are hard-coded in `data_preprocessing.ipynb`.

25 US states are included after screening for zero-incidence reporting
artifacts and minimum cumulative case volume (see manuscript for details).

---

## Requirements

```
Python >= 3.9
lightgbm
xgboost
catboost
torch
optuna
pandas
numpy
scikit-learn
matplotlib
requests
jupyter
```

Install all dependencies with:

```
pip install lightgbm xgboost catboost torch optuna pandas numpy scikit-learn matplotlib requests jupyter
```

---

## Outputs

All model prediction files are saved to the `outputs/` directory, which
is created automatically on first run. The ensemble script expects all
five model CSVs to be present before running.

---

## Citation

[Citation details to be added upon publication]

---

## License

Code is released under the MIT License.
```
