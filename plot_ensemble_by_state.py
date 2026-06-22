from pathlib import Path

import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_CSVS = {
    "lgbm": Path("outputs/lgbm_test_predictions.csv"),
    "catboost": Path("outputs/catboost_test_predictions.csv"),
    "xgboost": Path("outputs/xgboost_test_predictions.csv"),
    "lstm": Path("outputs/lstm_test_predictions.csv"),
    "gru": Path("outputs/gru_test_predictions.csv"),
}


def inverse_wis_weights(wis_scores, eps=1e-8):
    wis_scores = np.asarray(wis_scores, dtype=float)
    safe = np.maximum(wis_scores, eps)
    weights = 1 / safe
    return weights / weights.sum()


def load_model_predictions(model_csvs):
    frames = []
    for model_name, path in model_csvs.items():
        if not path.exists():
            continue
        df = pd.read_csv(path, parse_dates=["date", "cutoff_date"])
        df["model"] = model_name
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    predictions = pd.concat(frames, ignore_index=True)
    return (
        predictions.groupby(["date", "state", "split", "model"], as_index=False)
        .agg(
            cases_per_100k=("cases_per_100k", "first"),
            prediction=("prediction", "mean"),
            lower_95=("lower_95", "mean"),
            upper_95=("upper_95", "mean"),
            wis=("wis", "mean"),
        )
    )


def add_test_set_wis_weights(predictions):
    history = predictions[predictions["split"] == "test"][["state", "model", "wis"]].copy()
    history["wis"] = pd.to_numeric(history["wis"], errors="coerce")
    history = history[np.isfinite(history["wis"])]
    weights = history.groupby(["state", "model"])["wis"].mean().rename("weight_wis")
    return predictions.join(weights, on=["state", "model"])


def compute_split_ensemble(predictions, split, require_all_models=True):
    rows = add_test_set_wis_weights(predictions)
    rows = rows[rows["split"] == split].copy()
    expected_count = rows["model"].nunique()
    ensemble_rows = []

    for (date, state), group in rows.groupby(["date", "state"]):
        if require_all_models and group["model"].nunique() < expected_count:
            continue

        predictions_values = group["prediction"].astype(float).to_numpy()
        lower_values = group["lower_95"].astype(float).to_numpy()
        upper_values = group["upper_95"].astype(float).to_numpy()
        weight_wis = group["weight_wis"].astype(float).to_numpy()

        if np.isfinite(weight_wis).all():
            weights = inverse_wis_weights(weight_wis)
            prediction_weighted = float((predictions_values * weights).sum())
            lower = float((lower_values * weights).sum())
            upper = float((upper_values * weights).sum())
        else:
            prediction_weighted = float(predictions_values.mean())
            lower = float(lower_values.mean())
            upper = float(upper_values.mean())

        ensemble_rows.append(
            {
                "date": date,
                "state": state,
                "cases_per_100k": float(group["cases_per_100k"].iloc[0]),
                "prediction": float(predictions_values.mean()),
                "prediction_weighted": prediction_weighted,
                "lower_95": lower,
                "upper_95": upper,
                "model_count": group["model"].nunique(),
            }
        )

    return pd.DataFrame(ensemble_rows)


def compute_train_test_ensemble(predictions, require_all_models=True):
    frames = []
    for split in ["train", "test"]:
        split_ensemble = compute_split_ensemble(
            predictions,
            split=split,
            require_all_models=require_all_models,
        )
        if not split_ensemble.empty:
            split_ensemble["split"] = split
            frames.append(split_ensemble)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def plot_state_grid(df, output_path, prediction_col="prediction_weighted", ncols=5, title="Ensemble Predictions by State"):
    states = sorted(df["state"].dropna().unique())
    nrows = math.ceil(len(states) / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4.2 * ncols, 2.8 * nrows),
        sharex=True,
    )
    axes = axes.flatten()

    for ax, state in zip(axes, states):
        state_df = df[df["state"] == state].sort_values("date")
        ax.plot(
            state_df["date"],
            state_df["cases_per_100k"],
            color="#222222",
            linewidth=1.4,
            label="Actual",
        )
        ax.plot(
            state_df["date"],
            state_df[prediction_col],
            color="#1f77b4",
            linewidth=1.4,
            label="Ensemble",
        )
        if {"lower_95", "upper_95"}.issubset(state_df.columns):
            ax.fill_between(
                state_df["date"],
                state_df["lower_95"],
                state_df["upper_95"],
                color="#1f77b4",
                alpha=0.16,
                linewidth=0,
            )
        ax.set_title(state, fontsize=10)
        ax.tick_params(axis="x", labelrotation=45, labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(alpha=0.25)

    for ax in axes[len(states) :]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle(title, y=0.995, fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_state_grid_by_split(
    df,
    output_path,
    prediction_col="prediction_weighted",
    ncols=5,
    title=None,
):
    states = sorted(df["state"].dropna().unique())
    nrows = math.ceil(len(states) / ncols)
    test_start = df.loc[df["split"] == "test", "date"].min()

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4.2 * ncols, 2.8 * nrows),
        sharex=True,
    )
    axes = axes.flatten()

    for ax, state in zip(axes, states):
        state_df = df[df["state"] == state].sort_values("date")
        ax.plot(
            state_df["date"],
            state_df["cases_per_100k"],
            color="#222222",
            linewidth=1.2,
            label="Actual",
        )

        for split, color, label in [
            ("train", "#2ca02c", "Train ensemble"),
            ("test", "#d62728", "Test ensemble"),
        ]:
            split_df = state_df[state_df["split"] == split]
            if split_df.empty:
                continue
            ax.plot(
                split_df["date"],
                split_df[prediction_col],
                color=color,
                linewidth=1.5,
                label=label,
            )
            if {"lower_95", "upper_95"}.issubset(split_df.columns):
                ax.fill_between(
                    split_df["date"],
                    split_df["lower_95"],
                    split_df["upper_95"],
                    color=color,
                    alpha=0.12,
                    linewidth=0,
                )

        if pd.notna(test_start):
            ax.axvline(
                test_start,
                color="#555555",
                linestyle="--",
                linewidth=1.0,
                alpha=0.8,
                label="Test start",
            )

        ax.set_title(state, fontsize=10)
        ax.tick_params(axis="x", labelrotation=45, labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(alpha=0.25)

    for ax in axes[len(states) :]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(unique.values(), unique.keys(), loc="upper center", ncol=4, frameon=False)
    if title:
        fig.suptitle(title, y=0.995, fontsize=16)
        fig.tight_layout(rect=(0, 0, 1, 0.965))
    else:
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_individual_states_by_split(df, output_dir, prediction_col="prediction_weighted"):
    output_dir.mkdir(parents=True, exist_ok=True)
    test_start = df.loc[df["split"] == "test", "date"].min()

    for state, state_df in df.groupby("state"):
        state_df = state_df.sort_values("date")
        fig, ax = plt.subplots(figsize=(10, 4.8))
        ax.plot(
            state_df["date"],
            state_df["cases_per_100k"],
            color="#222222",
            linewidth=1.8,
            label="Actual",
        )

        for split, color, label in [
            ("train", "#2ca02c", "Train ensemble"),
            ("test", "#d62728", "Test ensemble"),
        ]:
            split_df = state_df[state_df["split"] == split]
            if split_df.empty:
                continue
            ax.plot(
                split_df["date"],
                split_df[prediction_col],
                color=color,
                linewidth=1.8,
                label=label,
            )
            if {"lower_95", "upper_95"}.issubset(split_df.columns):
                ax.fill_between(
                    split_df["date"],
                    split_df["lower_95"],
                    split_df["upper_95"],
                    color=color,
                    alpha=0.12,
                    linewidth=0,
                )

        if pd.notna(test_start):
            ax.axvline(
                test_start,
                color="#555555",
                linestyle="--",
                linewidth=1.2,
                label="Test start",
            )

        ax.set_title(f"{state}: Train/Test Ensemble vs Actual")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cases per 100k")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.autofmt_xdate()
        fig.tight_layout()

        safe_state = str(state).replace(" ", "_").replace("/", "_")
        fig.savefig(output_dir / f"{safe_state}.png", dpi=180)
        plt.close(fig)


def plot_individual_states(df, output_dir, prediction_col="prediction_weighted"):
    output_dir.mkdir(parents=True, exist_ok=True)
    for state, state_df in df.groupby("state"):
        state_df = state_df.sort_values("date")
        fig, ax = plt.subplots(figsize=(10, 4.8))
        ax.plot(
            state_df["date"],
            state_df["cases_per_100k"],
            color="#222222",
            linewidth=1.8,
            label="Actual",
        )
        ax.plot(
            state_df["date"],
            state_df[prediction_col],
            color="#1f77b4",
            linewidth=1.8,
            label="Ensemble",
        )
        if {"lower_95", "upper_95"}.issubset(state_df.columns):
            ax.fill_between(
                state_df["date"],
                state_df["lower_95"],
                state_df["upper_95"],
                color="#1f77b4",
                alpha=0.16,
                linewidth=0,
                label="95% interval",
            )
        ax.set_title(f"{state}: Ensemble vs Actual")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cases per 100k")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.autofmt_xdate()
        fig.tight_layout()

        safe_state = str(state).replace(" ", "_").replace("/", "_")
        fig.savefig(output_dir / f"{safe_state}.png", dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="outputs/ensemble_test_predictions.csv",
        help="Ensemble CSV to plot.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/ensemble_plots",
        help="Directory where plots will be saved.",
    )
    parser.add_argument(
        "--prediction-col",
        default="prediction_weighted",
        choices=["prediction", "prediction_weighted"],
        help="Which ensemble prediction column to plot.",
    )
    parser.add_argument(
        "--ncols",
        type=int,
        default=5,
        help="Number of columns in the faceted state grid.",
    )
    parser.add_argument(
        "--skip-training-grid",
        action="store_true",
        help="Only plot the input CSV; do not build the training-set ensemble grid from model CSVs.",
    )
    parser.add_argument(
        "--allow-partial-training-ensemble",
        action="store_true",
        help="Allow training ensemble rows when not every available model has that date/state.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path, parse_dates=["date"])
    required_cols = {"date", "state", "cases_per_100k", args.prediction_col}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in {input_path}: {sorted(missing_cols)}")

    df = df.sort_values(["state", "date"])
    grid_path = output_dir / f"{input_path.stem}_by_state_grid.png"
    plot_state_grid(
        df,
        grid_path,
        prediction_col=args.prediction_col,
        ncols=args.ncols,
        title=f"{input_path.stem}: Ensemble Predictions by State",
    )
    plot_individual_states(df, output_dir / "states", prediction_col=args.prediction_col)

    print(f"Plotted {df['state'].nunique()} states from {input_path}")
    print(f"Saved grid plot to {grid_path}")
    print(f"Saved individual plots to {output_dir / 'states'}")

    if not args.skip_training_grid:
        model_predictions = load_model_predictions(MODEL_CSVS)
        if model_predictions.empty:
            print("No model prediction CSVs found; skipped training-set ensemble grid.")
            return

        train_ensemble = compute_split_ensemble(
            model_predictions,
            split="train",
            require_all_models=not args.allow_partial_training_ensemble,
        )
        if train_ensemble.empty:
            print("No training ensemble rows available; skipped training-set ensemble grid.")
            return

        train_dir = output_dir / "train"
        train_dir.mkdir(parents=True, exist_ok=True)
        train_grid_path = train_dir / "ensemble_train_predictions_by_state_grid.png"
        train_ensemble.to_csv(train_dir / "ensemble_train_predictions.csv", index=False)
        train_test_ensemble = compute_train_test_ensemble(
            model_predictions,
            require_all_models=not args.allow_partial_training_ensemble,
        )
        train_test_path = train_dir / "ensemble_train_test_predictions.csv"
        train_test_grid_path = train_dir / "ensemble_train_test_predictions_by_state_grid.png"
        train_test_ensemble.to_csv(train_test_path, index=False)
        plot_state_grid_by_split(
            train_test_ensemble,
            train_test_grid_path,
            prediction_col=args.prediction_col,
            ncols=args.ncols,
        )
        plot_individual_states_by_split(
            train_test_ensemble,
            train_dir / "states",
            prediction_col=args.prediction_col,
        )
        print(f"Plotted {train_ensemble['state'].nunique()} training states from model CSVs")
        print(f"Saved training-only data to {train_dir / 'ensemble_train_predictions.csv'}")
        print(f"Saved train/test data to {train_test_path}")
        print(f"Saved train/test grid plot to {train_test_grid_path}")
        print(f"Saved training individual plots to {train_dir / 'states'}")


if __name__ == "__main__":
    main()
