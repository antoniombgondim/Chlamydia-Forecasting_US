import pandas as pd
from pathlib import Path
from sklearn.metrics import root_mean_squared_error
from model_utils import (
    ensure_dirs,
    huber_loss,
    load_data,
    get_rolling_window_splits,
    prepare_common_features,
    tune_catboost_with_optuna,
    train_catboost_quantile_models,
    weighted_interval_score_by_row,
    save_model,
    save_predictions,
    save_wis_by_state_date,
)

OUTPUT_DIR = Path('outputs')
MODEL_DIR = Path('models')


def main():
    ensure_dirs(OUTPUT_DIR, MODEL_DIR)
    df = load_data('preprocessed_No_OHE.csv')
    start_date = pd.Timestamp('2026-02-01')
    all_rows = []
    all_results = []
    final_model_params = None

    for cutoff_date, train_df, test_df in get_rolling_window_splits(df, start_date):
        X_train, y_train, X_test, y_test, feature_cols = prepare_common_features(train_df, test_df, drop_cols=('date',))
        model, best_params, best_value = tune_catboost_with_optuna(X_train, y_train, train_df['date'], n_trials=500)
        final_model_params = best_params
        quantile_models = train_catboost_quantile_models(X_train, y_train, best_params)
        lower_95 = quantile_models[0.025].predict(X_test)
        preds = quantile_models[0.5].predict(X_test)
        upper_95 = quantile_models[0.975].predict(X_test)
        train_lower_95 = quantile_models[0.025].predict(X_train)
        train_preds = quantile_models[0.5].predict(X_train)
        train_upper_95 = quantile_models[0.975].predict(X_train)

        train_wis = weighted_interval_score_by_row(y_train, train_lower_95, train_preds, train_upper_95, alpha=0.05)
        test_wis = weighted_interval_score_by_row(y_test, lower_95, preds, upper_95, alpha=0.05)

        train_predictions = train_df[['date', 'state', 'cases_per_100k']].copy()
        train_predictions['cutoff_date'] = cutoff_date
        train_predictions['model'] = 'catboost'
        train_predictions['split'] = 'train'
        train_predictions['prediction'] = train_preds
        train_predictions['lower_95'] = train_lower_95
        train_predictions['upper_95'] = train_upper_95
        train_predictions['wis'] = train_wis

        test_predictions = test_df[['date', 'state', 'cases_per_100k']].copy()
        test_predictions['cutoff_date'] = cutoff_date
        test_predictions['model'] = 'catboost'
        test_predictions['split'] = 'test'
        test_predictions['prediction'] = preds
        test_predictions['lower_95'] = lower_95
        test_predictions['upper_95'] = upper_95
        test_predictions['wis'] = test_wis

        all_rows.extend([train_predictions, test_predictions])
        all_results.append({
            'cutoff_date': cutoff_date,
            'validation_huber_loss': best_value,
            'test_huber_loss': huber_loss(y_test, preds),
            'best_params': best_params,
        })

    predictions_df = pd.concat(all_rows, ignore_index=True)
    results_df = pd.DataFrame(all_results)
    save_predictions(predictions_df, OUTPUT_DIR / 'catboost_test_predictions.csv')
    save_wis_by_state_date(predictions_df, OUTPUT_DIR / 'catboost_test_wis_by_state_date.csv')

    if final_model_params is not None:
        X_all, y_all, _, _, _ = prepare_common_features(df, df, drop_cols=('date',))
        final_model = tune_catboost_with_optuna(X_all, y_all, df['date'], n_trials=500)[0]
        save_model(final_model, MODEL_DIR / 'catboost_model.joblib')

    print('CatBoost training completed')
    print(results_df.head())


if __name__ == '__main__':
    main()
