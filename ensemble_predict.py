import numpy as np
import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path('outputs')

MODEL_CSVS = {
    'lgbm': OUTPUT_DIR / 'lgbm_test_predictions.csv',
    'catboost': OUTPUT_DIR / 'catboost_test_predictions.csv',
    'xgboost': OUTPUT_DIR / 'xgboost_test_predictions.csv',
    'lstm': OUTPUT_DIR / 'lstm_test_predictions.csv',
    'gru': OUTPUT_DIR / 'gru_test_predictions.csv',
}


def ensure_dirs(*paths):
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def inverse_wis_weights(wis_scores, eps=1e-8):
    wis_scores = np.asarray(wis_scores, dtype=float)
    safe = np.maximum(wis_scores, eps)
    weights = 1 / safe
    return weights / weights.sum()


def save_predictions(predictions, path):
    ensure_dirs(Path(path).parent)
    predictions.to_csv(path, index=False)


def load_prediction_csvs(model_csvs):
    dfs = []
    for model_name, path in model_csvs.items():
        if path.exists():
            df = pd.read_csv(path, parse_dates=['date', 'cutoff_date'])
            df['model'] = model_name
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def add_test_set_wis_weights(predictions_df):
    predictions_df = predictions_df.copy()

    history = predictions_df[
        (predictions_df['split'] == 'test')
    ][['state', 'model', 'wis']].copy()
    history['wis'] = pd.to_numeric(history['wis'], errors='coerce')
    history = history[np.isfinite(history['wis'])]

    mean_wis_by_state_model = (
        history.groupby(['state', 'model'])['wis']
        .mean()
        .rename('weight_wis')
    )
    return predictions_df.join(
        mean_wis_by_state_model,
        on=['state', 'model'],
    )


def aggregate_ensemble(rows):
    def finite_mean(values):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        return float(values.mean()) if len(values) else np.nan

    def weighted_row(group):
        predictions = group['prediction'].astype(float).to_numpy()
        weight_wis = group['weight_wis'].astype(float).to_numpy() if 'weight_wis' in group else None
        mean_pred = predictions.mean()
        if weight_wis is not None and np.isfinite(weight_wis).all():
            weights = inverse_wis_weights(weight_wis)
            weighted_pred = float((predictions * weights).sum())
            weighted_lower = float((group['lower_95'].astype(float) * weights).sum())
            weighted_upper = float((group['upper_95'].astype(float) * weights).sum())
        else:
            weighted_pred = float(mean_pred)
            weighted_lower = float(group['lower_95'].astype(float).mean())
            weighted_upper = float(group['upper_95'].astype(float).mean())
        return pd.Series({
            'cases_per_100k': group['cases_per_100k'].iloc[0],
            'prediction': mean_pred,
            'prediction_weighted': weighted_pred,
            'lower_95': weighted_lower,
            'upper_95': weighted_upper,
            'model_count': len(group),
            'wis_mean': group['wis'].mean() if 'wis' in group else pd.NA,
            'weight_wis_mean': finite_mean(weight_wis) if weight_wis is not None else pd.NA,
        })

    ensemble = rows.groupby(['date', 'state'], as_index=False).apply(weighted_row)
    ensemble['date'] = pd.to_datetime(ensemble['date'])
    return ensemble


def compute_ensemble(predictions_df):
    test_df = predictions_df[predictions_df['split'] == 'test'].copy()
    return aggregate_ensemble(test_df)


def compute_full_history_ensemble(predictions_df):
    test_df = predictions_df[predictions_df['split'] == 'test'].copy()
    first_test_date = test_df['date'].min()
    train_df = predictions_df[
        (predictions_df['split'] == 'train')
        & (predictions_df['date'] < first_test_date)
    ].copy()
    first_cutoff_by_model = train_df.groupby('model')['cutoff_date'].transform('min')
    train_df = train_df[train_df['cutoff_date'] == first_cutoff_by_model]
    return aggregate_ensemble(pd.concat([train_df, test_df], ignore_index=True))


def main():
    ensure_dirs(OUTPUT_DIR)
    predictions_df = load_prediction_csvs(MODEL_CSVS)
    if predictions_df.empty:
        raise FileNotFoundError('No model prediction CSVs found in outputs/')
    predictions_df = add_test_set_wis_weights(predictions_df)

    ensemble_df = compute_ensemble(predictions_df)
    save_predictions(ensemble_df, OUTPUT_DIR / 'ensemble_test_predictions.csv')
    full_history_df = compute_full_history_ensemble(predictions_df)
    save_predictions(full_history_df, OUTPUT_DIR / 'ensemble_test_wis_by_state_date.csv')
    print('Saved ensemble test predictions to outputs/ensemble_test_predictions.csv')
    print('Saved full-history ensemble to outputs/ensemble_test_wis_by_state_date.csv')


if __name__ == '__main__':
    main()
