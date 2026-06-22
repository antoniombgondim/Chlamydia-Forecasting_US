import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import optuna


HUBER_DELTA = 1.0


def huber_loss(y_true, y_pred, delta=HUBER_DELTA):
    error = np.asarray(y_true) - np.asarray(y_pred)
    abs_error = np.abs(error)
    quadratic = np.minimum(abs_error, delta)
    linear = abs_error - quadratic
    return np.mean(0.5 * quadratic**2 + delta * linear)


def ensure_dirs(*paths):
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def load_data(path):
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    df['week'] = ((df['date'] - df['date'].min()).dt.days // 7).astype(int)
    return df


def get_rolling_window_splits(df, start_date, horizon_days=14, step='2W'):
    for cutoff_date in pd.date_range(start=start_date, end=df['date'].max(), freq=step):
        train_df = df[df['date'] <= cutoff_date].copy()
        test_df = df[(df['date'] > cutoff_date) & (df['date'] <= cutoff_date + pd.Timedelta(days=horizon_days))].copy()
        if len(test_df) == 0 or len(train_df) == 0:
            continue
        yield cutoff_date, train_df, test_df


def prepare_common_features(train_df, test_df, target_col='cases_per_100k', drop_cols=('date',)):
    candidate_cols = [col for col in train_df.columns if col != target_col and col not in drop_cols]
    feature_cols = [col for col in candidate_cols if col in test_df.columns]
    X_train = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()
    if 'state' in feature_cols:
        categories = pd.Categorical(pd.concat([X_train['state'], X_test['state']], ignore_index=True))
        X_train['state'] = X_train['state'].astype(categories.dtype)
        X_test['state'] = X_test['state'].astype(categories.dtype)
    return X_train, train_df[target_col], X_test, test_df[target_col], feature_cols


def prepare_xgb_features(
    train_df,
    test_df,
    target_col='cases_per_100k',
    drop_cols=('date',),
):
    X_train, y_train, X_test, y_test, feature_cols = prepare_common_features(
        train_df,
        test_df,
        target_col=target_col,
        drop_cols=drop_cols,
    )

    if 'state' in feature_cols:
        train_state_ohe = pd.get_dummies(X_train['state'], prefix='state')
        test_state_ohe = pd.get_dummies(X_test['state'], prefix='state')

        X_train = X_train.drop(columns=['state'])
        X_test = X_test.drop(columns=['state'])

        X_train = pd.concat([X_train, train_state_ohe], axis=1)
        X_test = pd.concat([X_test, test_state_ohe], axis=1)

        feature_cols = list(X_train.columns)

    return X_train, y_train, X_test, y_test, feature_cols


def tune_lgbm_with_optuna(X_train, y_train, train_dates, n_trials=50, random_state=42):
    validation_start = train_dates.max() - pd.Timedelta(days=14)
    train_mask = train_dates <= validation_start
    valid_mask = train_dates > validation_start
    if train_mask.sum() == 0 or valid_mask.sum() == 0:
        model = LGBMRegressor(
            objective='huber',
            metric='huber',
            random_state=random_state,
            verbose=-1,
        )
        model.fit(X_train, y_train)
        return model, {}, np.nan
    X_tr, y_tr = X_train.loc[train_mask], y_train.loc[train_mask]
    X_val, y_val = X_train.loc[valid_mask], y_train.loc[valid_mask]

    def objective(trial):
        params = {
            'objective': 'huber',
            'metric': 'huber',
            'boosting_type': 'gbdt',
            'n_estimators': trial.suggest_int('n_estimators', 100, 800),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 16, 128),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 80),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'random_state': random_state,
            'verbose': -1,
            'n_jobs': -1,
        }
        model = LGBMRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric='huber', callbacks=[lgb.early_stopping(30, verbose=False)])
        preds = model.predict(X_val)
        return huber_loss(y_val, preds)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_params = {
        **study.best_params,
        'objective': 'huber',
        'metric': 'huber',
        'boosting_type': 'gbdt',
        'random_state': random_state,
        'verbose': -1,
        'n_jobs': -1,
    }
    model = LGBMRegressor(**best_params)
    model.fit(X_train, y_train)
    return model, study.best_params, study.best_value


def train_lgbm_quantile_models(X_train, y_train, best_params, quantiles=(0.025, 0.5, 0.975), random_state=42):
    base_params = {
        **best_params,
        'boosting_type': 'gbdt',
        'random_state': random_state,
        'verbose': -1,
        'n_jobs': -1,
    }
    quantile_models = {}
    for alpha in quantiles:
        params = {**base_params, 'objective': 'quantile', 'alpha': alpha, 'metric': 'quantile'}
        model = LGBMRegressor(**params)
        model.fit(X_train, y_train)
        quantile_models[alpha] = model
    return quantile_models


def tune_catboost_with_optuna(X_train, y_train, train_dates, n_trials=50, random_state=42):
    validation_start = train_dates.max() - pd.Timedelta(days=14)
    train_mask = train_dates <= validation_start
    valid_mask = train_dates > validation_start
    cat_features = [col for col in X_train.columns if str(X_train[col].dtype) in ('category', 'object')]
    if train_mask.sum() == 0 or valid_mask.sum() == 0:
        model = CatBoostRegressor(
            loss_function=f'Huber:delta={HUBER_DELTA}',
            eval_metric=f'Huber:delta={HUBER_DELTA}',
            random_seed=random_state,
            verbose=False,
            allow_writing_files=False,
            cat_features=cat_features,
        )
        model.fit(X_train, y_train)
        return model, {}, np.nan
    X_tr, y_tr = X_train.loc[train_mask], y_train.loc[train_mask]
    X_val, y_val = X_train.loc[valid_mask], y_train.loc[valid_mask]

    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 200, 800),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'depth': trial.suggest_int('depth', 3, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
            'random_strength': trial.suggest_float('random_strength', 1e-3, 10.0, log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 5.0),
            'rsm': trial.suggest_float('rsm', 0.6, 1.0),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 80),
            'loss_function': f'Huber:delta={HUBER_DELTA}',
            'eval_metric': f'Huber:delta={HUBER_DELTA}',
            'random_seed': random_state,
            'verbose': False,
            'allow_writing_files': False,
            'cat_features': cat_features,
        }
        model = CatBoostRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True, early_stopping_rounds=30, verbose=False)
        preds = model.predict(X_val)
        return huber_loss(y_val, preds)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_params = {
        **study.best_params,
        'loss_function': f'Huber:delta={HUBER_DELTA}',
        'eval_metric': f'Huber:delta={HUBER_DELTA}',
        'random_seed': random_state,
        'verbose': False,
        'allow_writing_files': False,
        'cat_features': cat_features,
    }
    model = CatBoostRegressor(**best_params)
    model.fit(X_train, y_train)
    return model, study.best_params, study.best_value


def train_catboost_quantile_models(X_train, y_train, best_params, quantiles=(0.025, 0.5, 0.975), random_state=42):
    cat_features = [col for col in X_train.columns if str(X_train[col].dtype) in ('category', 'object')]
    base_params = {
        **best_params,
        'random_seed': random_state,
        'verbose': False,
        'allow_writing_files': False,
        'cat_features': cat_features,
    }
    quantile_models = {}
    for alpha in quantiles:
        params = {**base_params, 'loss_function': f'Quantile:alpha={alpha}', 'eval_metric': f'Quantile:alpha={alpha}'}
        model = CatBoostRegressor(**params)
        model.fit(X_train, y_train)
        quantile_models[alpha] = model
    return quantile_models


def tune_xgb_with_optuna(X_train, y_train, train_dates, n_trials=50, random_state=42):
    validation_start = train_dates.max() - pd.Timedelta(days=14)
    train_mask = train_dates <= validation_start
    valid_mask = train_dates > validation_start
    if train_mask.sum() == 0 or valid_mask.sum() == 0:
        model = XGBRegressor(objective='reg:pseudohubererror', random_state=random_state, n_jobs=-1, verbosity=0)
        model.fit(X_train, y_train)
        return model, {}, np.nan
    X_tr, y_tr = X_train.loc[train_mask], y_train.loc[train_mask]
    X_val, y_val = X_train.loc[valid_mask], y_train.loc[valid_mask]

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 800),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'objective': 'reg:pseudohubererror',
            'eval_metric': 'mphe',
            'early_stopping_rounds': 30,
            'random_state': random_state,
            'n_jobs': -1,
            'verbosity': 0,
        }
        model = XGBRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict(X_val)
        return huber_loss(y_val, preds)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_params = {
        **study.best_params,
        'objective': 'reg:pseudohubererror',
        'eval_metric': 'mphe',
        'random_state': random_state,
        'n_jobs': -1,
        'verbosity': 0,
    }
    model = XGBRegressor(**best_params)
    model.fit(X_train, y_train)
    return model, study.best_params, study.best_value


def train_xgb_quantile_model(X_train, y_train, best_params, quantiles=(0.025, 0.5, 0.975), random_state=42):
    params = {
        **best_params,
        'objective': 'reg:quantileerror',
        'quantile_alpha': np.asarray(quantiles),
        'tree_method': 'hist',
        'random_state': random_state,
        'n_jobs': -1,
        'verbosity': 0,
    }
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    return model



def pack_test_rows(train_df, test_df, predictions, train_preds, lower_95, upper_95, train_wis, test_wis, model_name):
    train_predictions = train_df[['date', 'state', 'cases_per_100k']].copy()
    train_predictions['cutoff_date'] = predictions['cutoff_date']
    train_predictions['model'] = model_name
    train_predictions['split'] = 'train'
    train_predictions['prediction'] = train_preds
    train_predictions['lower_95'] = lower_95
    train_predictions['upper_95'] = upper_95
    train_predictions['wis'] = train_wis

    test_predictions = test_df[['date', 'state', 'cases_per_100k']].copy()
    test_predictions['cutoff_date'] = predictions['cutoff_date']
    test_predictions['model'] = model_name
    test_predictions['split'] = 'test'
    test_predictions['prediction'] = predictions['prediction']
    test_predictions['lower_95'] = predictions['lower_95']
    test_predictions['upper_95'] = predictions['upper_95']
    test_predictions['wis'] = test_wis
    return train_predictions, test_predictions


def interval_score(y_true, lower, upper, alpha=0.05):
    y_true, lower, upper = np.asarray(y_true), np.asarray(lower), np.asarray(upper)
    return (upper - lower) + (2 / alpha) * (lower - y_true) * (y_true < lower) + (2 / alpha) * (y_true - upper) * (y_true > upper)


def weighted_interval_score_by_row(y_true, lower, median, upper, alpha=0.05):
    absolute_error = np.abs(np.asarray(y_true) - np.asarray(median))
    score = interval_score(y_true, lower, upper, alpha=alpha)
    return (0.5 * absolute_error + (alpha / 2) * score) / 1.5


def weighted_interval_score(y_true, lower, median, upper, alpha=0.05):
    if lower is None or upper is None:
        return np.nan
    return np.mean(weighted_interval_score_by_row(y_true, lower, median, upper, alpha=alpha))


def inverse_wis_weights(wis_scores, eps=1e-8):
    wis_scores = np.asarray(wis_scores, dtype=float)
    safe = np.maximum(wis_scores, eps)
    weights = 1 / safe
    return weights / weights.sum()


def save_model(model, path):
    ensure_dirs(Path(path).parent)
    joblib.dump(model, path)


def save_predictions(predictions, path):
    ensure_dirs(Path(path).parent)
    predictions.to_csv(path, index=False)


def save_wis_by_state_date(predictions, path):
    test_df = predictions[predictions['split'] == 'test'].copy()
    grouped = test_df.groupby(['date', 'state', 'model'], as_index=False).agg(
        cases_per_100k=('cases_per_100k', 'first'),
        prediction=('prediction', 'mean'),
        lower_95=('lower_95', 'mean'),
        upper_95=('upper_95', 'mean'),
        wis=('wis', 'mean'),
    )
    ensure_dirs(Path(path).parent)
    grouped.to_csv(path, index=False)
    return grouped


def safe_mean(values):
    if len(values) == 0:
        return np.nan
    return np.mean(values)
