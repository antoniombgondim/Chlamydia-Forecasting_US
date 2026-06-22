from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset


DEFAULT_SKIP_SCALING = [
    "holiday",
    "state",
    "sine_week",
    "cosine_week",
    "sine_month",
    "cosine_month",
    "sine_quarter",
    "cosine_quarter",
    "date",
    "year",
]


@dataclass
class PreprocessResult:
    df: pd.DataFrame
    feature_cols: list[str]
    scale_cols: list[str]
    scaler: StandardScaler
    target_scaler: Optional[StandardScaler]
    state_encoder: LabelEncoder


class StateDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df,
        feature_cols,
        target_col,
        lookback,
        forecast_window,
        start_test_date=None,
        end_test_date=None,
        split=None,
        skip_nan_windows=True,
    ):
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.lookback = lookback
        self.forecast_window = forecast_window
        self.start_test_date = start_test_date
        self.end_test_date = end_test_date
        self.split = split
        self.skip_nan_windows = skip_nan_windows
        self.Xdata = []
        self.Ydata = []
        self.state_data = []
        self.target_dates = []

        if split not in {None, "train", "test"}:
            raise ValueError("split must be None, 'train', or 'test'")

        if split is not None and start_test_date is None:
            raise ValueError("start_test_date is required when split is 'train' or 'test'")

        if start_test_date is not None:
            start_test_date = pd.to_datetime(start_test_date)
        if end_test_date is not None:
            end_test_date = pd.to_datetime(end_test_date)

        for state, group in df.groupby("state"):
            group = group.sort_values("date").reset_index(drop=True)
            features = group[feature_cols].to_numpy(dtype=np.float32)
            targets = group[target_col].to_numpy(dtype=np.float32)
            dates = pd.to_datetime(group["date"]).to_numpy()

            n_windows = len(group) - lookback - forecast_window + 1
            for i in range(max(n_windows, 0)):
                y_start_date = pd.to_datetime(dates[i + lookback])

                if split == "train" and y_start_date >= start_test_date:
                    continue
                if split == "test" and y_start_date < start_test_date:
                    continue
                if split == "test" and end_test_date is not None and y_start_date >= end_test_date:
                    continue

                X_window = features[i : i + lookback]
                y_window = targets[i + lookback : i + lookback + forecast_window]

                if skip_nan_windows and (np.isnan(X_window).any() or np.isnan(y_window).any()):
                    continue

                self.Xdata.append(X_window)
                self.Ydata.append(y_window)
                self.state_data.append(state)
                self.target_dates.append(pd.to_datetime(dates[i + lookback : i + lookback + forecast_window]))

        if self.Xdata:
            self.Xdata = torch.FloatTensor(np.array(self.Xdata))
            self.Ydata = torch.FloatTensor(np.array(self.Ydata))
            self.state_data = torch.LongTensor(np.array(self.state_data))
        else:
            self.Xdata = torch.empty((0, lookback, len(feature_cols)), dtype=torch.float32)
            self.Ydata = torch.empty((0, forecast_window), dtype=torch.float32)
            self.state_data = torch.empty((0,), dtype=torch.long)

    def __len__(self):
        return len(self.Xdata)

    def __getitem__(self, idx):
        return self.Xdata[idx], self.state_data[idx], self.Ydata[idx]


def preprocess_df(
    df,
    target_col="cases_per_100k",
    start_test_date="2026-01-01",
    skip_scaling_cols=None,
    scale_target=False,
):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    start_test_date = pd.to_datetime(start_test_date)

    state_encoder = LabelEncoder()
    df["state"] = state_encoder.fit_transform(df["state"])

    if "week" in df.columns:
        df["sine_week"] = np.sin(2 * np.pi * df["week"] / 52)
        df["cosine_week"] = np.cos(2 * np.pi * df["week"] / 52)
        df = df.drop(columns=["week"])

    if "month" in df.columns:
        df["sine_month"] = np.sin(2 * np.pi * df["month"] / 12)
        df["cosine_month"] = np.cos(2 * np.pi * df["month"] / 12)
        df = df.drop(columns=["month"])

    if "quarter" in df.columns:
        df["sine_quarter"] = np.sin(2 * np.pi * df["quarter"] / 4)
        df["cosine_quarter"] = np.cos(2 * np.pi * df["quarter"] / 4)
        df = df.drop(columns=["quarter"])

    feature_cols = [col for col in df.columns if col not in ["date", "state", target_col]]
    skip_scaling_cols = DEFAULT_SKIP_SCALING if skip_scaling_cols is None else skip_scaling_cols
    scale_cols = [col for col in feature_cols if col not in skip_scaling_cols]

    scaler = StandardScaler()
    if scale_cols:
        df[scale_cols] = df[scale_cols].astype(float)
        train_rows = df["date"] < start_test_date
        scaler.fit(df.loc[train_rows, scale_cols])
        df.loc[:, scale_cols] = scaler.transform(df.loc[:, scale_cols])

    target_scaler = None
    if scale_target:
        target_scaler = StandardScaler()
        train_rows = df["date"] < start_test_date
        df[target_col] = df[target_col].astype(float)
        target_scaler.fit(df.loc[train_rows, [target_col]])
        df.loc[:, target_col] = target_scaler.transform(df.loc[:, [target_col]])

    return PreprocessResult(
        df=df,
        feature_cols=feature_cols,
        scale_cols=scale_cols,
        scaler=scaler,
        target_scaler=target_scaler,
        state_encoder=state_encoder,
    )


def create_datasets(
    df,
    feature_cols,
    target_col="cases_per_100k",
    lookback=52,
    forecast_window=2,
    start_test_date="2026-01-01",
    end_test_date=None,
    skip_nan_windows=True,
):
    train_dataset = StateDataset(
        df,
        feature_cols,
        target_col,
        lookback,
        forecast_window,
        start_test_date=start_test_date,
        end_test_date=end_test_date,
        split="train",
        skip_nan_windows=skip_nan_windows,
    )
    test_dataset = StateDataset(
        df,
        feature_cols,
        target_col,
        lookback,
        forecast_window,
        start_test_date=start_test_date,
        end_test_date=end_test_date,
        split="test",
        skip_nan_windows=skip_nan_windows,
    )
    return train_dataset, test_dataset


def create_dataloaders(train_dataset, test_dataset, batch_size=32, num_workers=0):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, test_loader


def build_dataloaders(
    df,
    target_col="cases_per_100k",
    start_test_date="2026-01-01",
    lookback=52,
    forecast_window=2,
    batch_size=32,
    drop_nan_rows=False,
    scale_target=False,
    end_test_date=None,
    skip_nan_windows=True,
    skip_scaling_cols=None,
    num_workers=0,
):
    processed = preprocess_df(
        df,
        target_col=target_col,
        start_test_date=start_test_date,
        skip_scaling_cols=skip_scaling_cols,
        scale_target=scale_target,
    )
    if drop_nan_rows:
        processed.df = processed.df.dropna(
            subset=processed.feature_cols + [target_col]
        ).copy()

    train_dataset, test_dataset = create_datasets(
        processed.df,
        processed.feature_cols,
        target_col=target_col,
        lookback=lookback,
        forecast_window=forecast_window,
        start_test_date=start_test_date,
        end_test_date=end_test_date,
        skip_nan_windows=skip_nan_windows,
    )
    train_loader, test_loader = create_dataloaders(
        train_dataset,
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    return train_loader, test_loader, train_dataset, test_dataset, processed


def save_dataset_tensors(path, train_dataset, test_dataset, processed, batch_size=32):
    torch.save(
        {
            "X_train": train_dataset.Xdata,
            "y_train": train_dataset.Ydata,
            "state_train": train_dataset.state_data,
            "X_test": test_dataset.Xdata,
            "y_test": test_dataset.Ydata,
            "state_test": test_dataset.state_data,
            "feature_cols": processed.feature_cols,
            "scale_cols": processed.scale_cols,
            "target_scaler": processed.target_scaler,
            "lookback": train_dataset.lookback,
            "forecast_window": train_dataset.forecast_window,
            "batch_size": batch_size,
        },
        path,
    )


def load_saved_dataloaders(path, batch_size=None, map_location="cpu"):
    data = torch.load(path, map_location=map_location)
    if batch_size is None:
        batch_size = data.get("batch_size", 32)

    train_dataset = TensorDataset(
        data["X_train"],
        data["state_train"],
        data["y_train"],
    )
    test_dataset = TensorDataset(
        data["X_test"],
        data["state_test"],
        data["y_test"],
    )
    train_loader, test_loader = create_dataloaders(
        train_dataset,
        test_dataset,
        batch_size=batch_size,
    )
    return train_loader, test_loader, data
