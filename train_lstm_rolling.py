from pathlib import Path

import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from dl_data_utils import build_dataloaders


QUANTILES = (0.025, 0.5, 0.975)


def save_predictions(predictions, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(path, index=False)


def interval_score(y_true, lower, upper, alpha=0.05):
    y_true = np.asarray(y_true)
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    return (
        (upper - lower)
        + (2 / alpha) * (lower - y_true) * (y_true < lower)
        + (2 / alpha) * (y_true - upper) * (y_true > upper)
    )


def weighted_interval_score_by_row(y_true, lower, median, upper, alpha=0.05):
    absolute_error = np.abs(np.asarray(y_true) - np.asarray(median))
    score = interval_score(y_true, lower, upper, alpha=alpha)
    return (0.5 * absolute_error + (alpha / 2) * score) / 1.5


def save_wis_by_state_date(predictions, path):
    test_df = predictions[predictions["split"] == "test"].copy()
    grouped = test_df.groupby(["date", "state", "model"], as_index=False).agg(
        cases_per_100k=("cases_per_100k", "first"),
        prediction=("prediction", "mean"),
        lower_95=("lower_95", "mean"),
        upper_95=("upper_95", "mean"),
        wis=("wis", "mean"),
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(path, index=False)
    return grouped


class LSTMForecastModel(nn.Module):
    def __init__(
        self,
        num_features,
        num_states,
        forecast_window,
        num_quantiles=3,
        hidden_size=30,
        num_layers=1,
        dropout=0.2,
        state_embedding_dim=2,
    ):
        super().__init__()
        self.forecast_window = forecast_window
        self.num_quantiles = num_quantiles
        self.state_embedding = nn.Embedding(num_states, state_embedding_dim)
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size + state_embedding_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, forecast_window * num_quantiles),
        )

    def forward(self, x, state):
        # x: [batch_size, lookback, num_features]
        # state: [batch_size]
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        state_embedding = self.state_embedding(state)
        features = torch.cat([last_hidden, state_embedding], dim=1)
        out = self.head(features)
        out = out.view(-1, self.forecast_window, self.num_quantiles)
        return torch.sort(out, dim=-1).values


def quantile_loss(y_pred, y_true, quantiles):
    y_true = y_true.unsqueeze(-1)
    errors = y_true - y_pred
    losses = torch.maximum(quantiles * errors, (quantiles - 1) * errors)
    return losses.mean()


def train_one_epoch(model, train_loader, quantiles, optimizer, device):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for X_batch, state_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        state_batch = state_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        y_pred = model(X_batch, state_batch)
        loss = quantile_loss(y_pred, y_batch, quantiles)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        total_samples += X_batch.size(0)

    return total_loss / max(total_samples, 1)


def evaluate(model, data_loader, quantiles, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_targets = []
    all_states = []

    with torch.no_grad():
        for X_batch, state_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            state_batch = state_batch.to(device)
            y_batch = y_batch.to(device)

            y_pred = model(X_batch, state_batch)
            loss = quantile_loss(y_pred, y_batch, quantiles)

            total_loss += loss.item() * X_batch.size(0)
            total_samples += X_batch.size(0)
            all_preds.append(y_pred.cpu())
            all_targets.append(y_batch.cpu())
            all_states.append(state_batch.cpu())

    if total_samples == 0:
        return np.nan, None, None, None

    return (
        total_loss / total_samples,
        torch.cat(all_preds),
        torch.cat(all_targets),
        torch.cat(all_states),
    )


def inverse_transform_targets(values, target_scaler):
    if target_scaler is None:
        return values
    original_shape = values.shape
    return target_scaler.inverse_transform(values.reshape(-1, 1)).reshape(original_shape)


def make_prediction_frame(
    dataset,
    preds,
    targets,
    cutoff_date,
    model_name,
    split,
    state_encoder,
    target_scaler=None,
):
    rows = []
    preds = preds.numpy()
    targets = targets.numpy()
    preds = inverse_transform_targets(preds, target_scaler)
    targets = inverse_transform_targets(targets, target_scaler)
    states = dataset.state_data.numpy()

    for sample_idx, target_dates in enumerate(dataset.target_dates):
        state_name = state_encoder.inverse_transform([int(states[sample_idx])])[0]
        for horizon_idx, target_date in enumerate(target_dates):
            lower_95, median, upper_95 = preds[sample_idx, horizon_idx]
            actual = targets[sample_idx, horizon_idx]
            wis = weighted_interval_score_by_row(
                [actual],
                [lower_95],
                [median],
                [upper_95],
                alpha=0.05,
            )[0]
            rows.append(
                {
                    "date": pd.to_datetime(target_date),
                    "state": state_name,
                    "cases_per_100k": float(actual),
                    "cutoff_date": cutoff_date,
                    "model": model_name,
                    "split": split,
                    "prediction": float(median),
                    "lower_95": float(lower_95),
                    "upper_95": float(upper_95),
                    "wis": float(wis),
                }
            )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    return (
        frame.groupby(["date", "state", "cutoff_date", "model", "split"], as_index=False)
        .agg(
            cases_per_100k=("cases_per_100k", "first"),
            prediction=("prediction", "mean"),
            lower_95=("lower_95", "mean"),
            upper_95=("upper_95", "mean"),
            wis=("wis", "mean"),
        )
    )


def iter_cutoff_dates(df, start_date, step_days, forecast_window):
    start_date = pd.to_datetime(start_date)
    max_date = pd.to_datetime(df["date"]).max()
    # Need enough future rows for the whole target window.
    latest_start = max_date - pd.Timedelta(days=7 * (forecast_window - 1))
    return pd.date_range(start=start_date, end=latest_start, freq=f"{step_days}D")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="preprocessed_No_OHE.csv")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--start-date", default="2026-02-02")
    parser.add_argument("--step-days", type=int, default=14)
    parser.add_argument("--lookback", type=int, default=52)
    parser.add_argument("--forecast-window", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--max-cutoffs", type=int, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    quantiles = torch.tensor(QUANTILES, dtype=torch.float32, device=device)
    df = pd.read_csv(args.data_path)
    df["date"] = pd.to_datetime(df["date"])

    metrics = []
    prediction_rows = []
    cutoffs = list(iter_cutoff_dates(df, args.start_date, args.step_days, args.forecast_window))
    if args.max_cutoffs is not None:
        cutoffs = cutoffs[: args.max_cutoffs]

    for cutoff_date in cutoffs:
        end_test_date = cutoff_date + pd.Timedelta(days=args.step_days)
        print(f"\nCutoff {cutoff_date.date()} -> test target starts before {end_test_date.date()}")

        train_loader, test_loader, train_dataset, test_dataset, processed = build_dataloaders(
            df,
            target_col="cases_per_100k",
            start_test_date=cutoff_date,
            end_test_date=end_test_date,
            lookback=args.lookback,
            forecast_window=args.forecast_window,
            batch_size=args.batch_size,
            drop_nan_rows=True,
            scale_target=True,
            skip_nan_windows=False,
        )

        if len(train_dataset) == 0 or len(test_dataset) == 0:
            print("Skipping cutoff because train or test dataset is empty")
            continue

        model = LSTMForecastModel(
            num_features=len(processed.feature_cols),
            num_states=processed.df["state"].nunique(),
            forecast_window=args.forecast_window,
            num_quantiles=len(QUANTILES),
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        best_test_loss = float("inf")
        best_epoch = 0
        best_state_dict = None
        epochs_without_improvement = 0

        for epoch in range(args.epochs):
            train_loss = train_one_epoch(model, train_loader, quantiles, optimizer, device)
            test_loss, _, _, _ = evaluate(model, test_loader, quantiles, device)
            print(
                f"epoch {epoch + 1:03d}/{args.epochs} "
                f"train_quantile_loss={train_loss:.4f} test_quantile_loss={test_loss:.4f}"
            )

            if test_loss < best_test_loss - args.min_delta:
                best_test_loss = test_loss
                best_epoch = epoch + 1
                best_state_dict = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= args.patience:
                print(
                    f"early stopping at epoch {epoch + 1}; "
                    f"best_epoch={best_epoch} best_test_quantile_loss={best_test_loss:.4f}"
                )
                break

        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)

        train_eval_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
        train_loss, train_preds, train_targets, _ = evaluate(model, train_eval_loader, quantiles, device)
        test_loss, test_preds, test_targets, _ = evaluate(model, test_loader, quantiles, device)

        train_wis = np.nan
        if train_preds is not None:
            train_frame = make_prediction_frame(
                train_dataset,
                train_preds,
                train_targets,
                cutoff_date,
                "lstm",
                "train",
                processed.state_encoder,
                processed.target_scaler,
            )
            train_wis = train_frame["wis"].mean() if not train_frame.empty else np.nan
            prediction_rows.append(train_frame)

        test_wis = np.nan
        if test_preds is not None:
            test_frame = make_prediction_frame(
                test_dataset,
                test_preds,
                test_targets,
                cutoff_date,
                "lstm",
                "test",
                processed.state_encoder,
                processed.target_scaler,
            )
            test_wis = test_frame["wis"].mean() if not test_frame.empty else np.nan
            prediction_rows.append(test_frame)

        metrics.append(
            {
                "cutoff_date": cutoff_date,
                "end_test_date": end_test_date,
                "train_samples": len(train_dataset),
                "test_samples": len(test_dataset),
                "train_quantile_loss": train_loss,
                "test_quantile_loss": test_loss,
                "best_epoch": best_epoch,
                "best_test_quantile_loss": best_test_loss,
                "train_wis": train_wis,
                "test_wis": test_wis,
            }
        )

    metrics_df = pd.DataFrame(metrics)
    predictions_df = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    metrics_df.to_csv(output_dir / "lstm_rolling_metrics.csv", index=False)
    save_predictions(predictions_df, output_dir / "lstm_test_predictions.csv")
    save_wis_by_state_date(predictions_df, output_dir / "lstm_test_wis_by_state_date.csv")

    print("\nLSTM rolling training completed")
    print(metrics_df.head())


if __name__ == "__main__":
    main()
