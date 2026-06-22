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

ENSEMBLE_GROUPS = {
    "ensemble_all_5": ["lgbm", "catboost", "xgboost", "lstm", "gru"],
    "ensemble_tree": ["lgbm", "catboost", "xgboost"],
    "ensemble_dl": ["lstm", "gru"],
}


def interval_score(y_true, lower, upper, alpha=0.05):
    y_true = np.asarray(y_true, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    return (
        (upper - lower)
        + (2 / alpha) * (lower - y_true) * (y_true < lower)
        + (2 / alpha) * (y_true - upper) * (y_true > upper)
    )


def weighted_interval_score_by_row(y_true, lower, median, upper, alpha=0.05):
    absolute_error = np.abs(np.asarray(y_true, dtype=float) - np.asarray(median, dtype=float))
    score = interval_score(y_true, lower, upper, alpha=alpha)
    return (0.5 * absolute_error + (alpha / 2) * score) / 1.5


def inverse_wis_weights(wis_scores, eps=1e-8):
    wis_scores = np.asarray(wis_scores, dtype=float)
    safe = np.maximum(wis_scores, eps)
    weights = 1 / safe
    return weights / weights.sum()


def load_model_predictions(model_csvs):
    frames = []
    for model_name, path in model_csvs.items():
        if not path.exists():
            print(f"Skipping missing file: {path}")
            continue
        df = pd.read_csv(path, parse_dates=["date", "cutoff_date"])
        df["model"] = model_name
        frames.append(df)

    if not frames:
        raise FileNotFoundError("No model prediction CSVs were found.")

    predictions = pd.concat(frames, ignore_index=True)
    required = {
        "date",
        "state",
        "split",
        "model",
        "cases_per_100k",
        "prediction",
        "lower_95",
        "upper_95",
        "wis",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    return predictions


def collapse_duplicate_model_rows(predictions):
    # Sequence models may predict the same date/state from overlapping horizons.
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


def model_wis_by_date(predictions):
    return (
        predictions.groupby(["date", "split", "model"], as_index=False)
        .agg(
            mean_wis=("wis", "mean"),
            state_count=("state", "nunique"),
            row_count=("wis", "size"),
        )
    )


def add_weight_wis(predictions):
    history = predictions[predictions["split"] == "test"][["state", "model", "wis"]].copy()
    history["wis"] = pd.to_numeric(history["wis"], errors="coerce")
    history = history[np.isfinite(history["wis"])]
    weights = history.groupby(["state", "model"])["wis"].mean().rename("weight_wis")
    return predictions.join(weights, on=["state", "model"])


def compute_ensemble_rows(predictions, model_names, ensemble_name, require_all_models=True):
    rows = predictions[predictions["model"].isin(model_names)].copy()
    rows = add_weight_wis(rows)
    expected_count = len([model for model in model_names if model in rows["model"].unique()])
    ensemble_rows = []

    for (date, state, split), group in rows.groupby(["date", "state", "split"]):
        if require_all_models and group["model"].nunique() < expected_count:
            continue

        predictions_values = group["prediction"].astype(float).to_numpy()
        lower_values = group["lower_95"].astype(float).to_numpy()
        upper_values = group["upper_95"].astype(float).to_numpy()
        weight_wis = group["weight_wis"].astype(float).to_numpy()

        if np.isfinite(weight_wis).all():
            weights = inverse_wis_weights(weight_wis)
            prediction = float((predictions_values * weights).sum())
            lower = float((lower_values * weights).sum())
            upper = float((upper_values * weights).sum())
        else:
            prediction = float(predictions_values.mean())
            lower = float(lower_values.mean())
            upper = float(upper_values.mean())

        actual = float(group["cases_per_100k"].iloc[0])
        wis = float(weighted_interval_score_by_row([actual], [lower], [prediction], [upper])[0])
        ensemble_rows.append(
            {
                "date": date,
                "state": state,
                "split": split,
                "cases_per_100k": actual,
                "prediction": prediction,
                "lower_95": lower,
                "upper_95": upper,
                "wis": wis,
                "model": ensemble_name,
                "model_count": group["model"].nunique(),
            }
        )

    return pd.DataFrame(ensemble_rows)


def ensemble_wis_by_date(predictions, require_all_models=True):
    ensemble_predictions = compute_ensemble_predictions(
        predictions,
        require_all_models=require_all_models,
    )
    if ensemble_predictions.empty:
        return pd.DataFrame(columns=["date", "split", "model", "mean_wis", "state_count", "row_count"])

    return model_wis_by_date(ensemble_predictions)


def compute_ensemble_predictions(predictions, require_all_models=True):
    frames = []
    available = set(predictions["model"].unique())
    for ensemble_name, model_names in ENSEMBLE_GROUPS.items():
        present_models = [model for model in model_names if model in available]
        if len(present_models) < 2:
            print(f"Skipping {ensemble_name}: fewer than two model files available.")
            continue
        ensemble_rows = compute_ensemble_rows(
            predictions,
            present_models,
            ensemble_name,
            require_all_models=require_all_models,
        )
        frames.append(ensemble_rows)

    if not frames:
        return pd.DataFrame(
            columns=[
                "date",
                "state",
                "split",
                "cases_per_100k",
                "prediction",
                "lower_95",
                "upper_95",
                "wis",
                "model",
                "model_count",
            ]
        )

    return pd.concat(frames, ignore_index=True)


def ensemble_wis_by_state_date(ensemble_predictions, split="train", models=None):
    rows = ensemble_predictions[ensemble_predictions["split"] == split].copy()
    if models is not None:
        rows = rows[rows["model"].isin(models)]

    if rows.empty:
        return pd.DataFrame(columns=["date", "state", "split", "model", "mean_wis", "row_count"])

    return (
        rows.groupby(["date", "state", "split", "model"], as_index=False)
        .agg(
            mean_wis=("wis", "mean"),
            row_count=("wis", "size"),
        )
    )


def model_weights_by_state_date(predictions, split="test"):
    rows = predictions[predictions["split"] == split].copy()
    rows["wis"] = pd.to_numeric(rows["wis"], errors="coerce")
    rows = rows[np.isfinite(rows["wis"])]

    if rows.empty:
        return pd.DataFrame(columns=["date", "state", "split", "model", "weight", "wis"])

    weight_rows = []
    for (date, state), group in rows.groupby(["date", "state"]):
        group = group.copy()
        group["weight"] = inverse_wis_weights(group["wis"].astype(float).to_numpy())
        weight_rows.append(group[["date", "state", "split", "model", "weight", "wis"]])

    return pd.concat(weight_rows, ignore_index=True) if weight_rows else pd.DataFrame()


def plot_accuracy(wis_by_date, output_path):
    splits = [split for split in ["train", "test"] if split in set(wis_by_date["split"])]
    fig, axes = plt.subplots(len(splits), 1, figsize=(14, 5 * len(splits)), sharex=False)
    if len(splits) == 1:
        axes = [axes]

    model_order = [
        "lgbm",
        "catboost",
        "xgboost",
        "lstm",
        "gru",
        "ensemble_all_5",
        "ensemble_tree",
        "ensemble_dl",
    ]

    for ax, split in zip(axes, splits):
        split_df = wis_by_date[wis_by_date["split"] == split].copy()
        for model in model_order:
            model_df = split_df[split_df["model"] == model].sort_values("date")
            if model_df.empty:
                continue
            linestyle = "--" if model.startswith("ensemble") else "-"
            linewidth = 2.4 if model.startswith("ensemble") else 1.4
            ax.plot(
                model_df["date"],
                model_df["mean_wis"],
                label=model,
                linestyle=linestyle,
                linewidth=linewidth,
            )

        ax.set_title(f"{split}/mean wis by date")
        ax.set_ylabel("Mean WIS (lower is better)")
        ax.grid(alpha=0.25)
        ax.legend(ncol=4, frameon=False)

    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_test_model_weights_by_state(weights_by_state_date, output_path, ncols=5):
    states = sorted(weights_by_state_date["state"].dropna().unique())
    if not states:
        print("No test-set model weight rows available; skipped state grid plot.")
        return

    nrows = math.ceil(len(states) / ncols)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4.2 * ncols, 2.8 * nrows),
        sharex=True,
    )
    axes = np.atleast_1d(axes).flatten()

    model_order = ["lgbm", "catboost", "xgboost", "lstm", "gru"]
    colors = {
        "lgbm": "#1f77b4",
        "catboost": "#ff7f0e",
        "xgboost": "#2ca02c",
        "lstm": "#d62728",
        "gru": "#9467bd",
    }

    for ax, state in zip(axes, states):
        state_df = weights_by_state_date[weights_by_state_date["state"] == state].copy()
        for model in model_order:
            model_df = state_df[state_df["model"] == model].sort_values("date")
            if model_df.empty:
                continue
            ax.plot(
                model_df["date"],
                model_df["weight"],
                color=colors[model],
                linewidth=1.5,
                label=model,
            )

        ax.set_title(state, fontsize=10)
        ax.set_ylim(-0.02, 1.02)
        ax.tick_params(axis="x", labelrotation=45, labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(alpha=0.25)

    for ax in axes[len(states) :]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(unique.values(), unique.keys(), loc="upper center", ncol=5, frameon=False)
    fig.supxlabel("Date", y=0.01)
    fig.supylabel("Inverse-WIS Weight", x=0.005)
    fig.tight_layout(rect=(0.015, 0.02, 1, 0.98))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/model_accuracy_plots")
    parser.add_argument(
        "--ncols",
        type=int,
        default=5,
        help="Number of columns in the faceted state grid.",
    )
    parser.add_argument(
        "--allow-partial-ensembles",
        action="store_true",
        help="Allow ensemble lines when not every selected model is available for a date/state.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = load_model_predictions(MODEL_CSVS)
    predictions = collapse_duplicate_model_rows(predictions)

    individual = model_wis_by_date(predictions)
    ensemble_predictions = compute_ensemble_predictions(
        predictions,
        require_all_models=not args.allow_partial_ensembles,
    )
    ensembles = model_wis_by_date(ensemble_predictions)
    wis_by_date = pd.concat([individual, ensembles], ignore_index=True)
    wis_by_date = wis_by_date.sort_values(["split", "model", "date"])

    csv_path = output_dir / "model_and_ensemble_wis_by_date.csv"
    plot_path = output_dir / "model_and_ensemble_wis_by_date.png"
    wis_by_date.to_csv(csv_path, index=False)
    plot_accuracy(wis_by_date, plot_path)

    test_model_weights = model_weights_by_state_date(predictions, split="test").sort_values(["state", "model", "date"])
    test_weights_csv_path = output_dir / "test_model_inverse_wis_weights_by_state_date.csv"
    test_weights_plot_path = output_dir / "test_model_inverse_wis_weights_by_state_grid.png"
    test_model_weights.to_csv(test_weights_csv_path, index=False)
    plot_test_model_weights_by_state(
        test_model_weights,
        test_weights_plot_path,
        ncols=args.ncols,
    )

    print(f"Saved per-date WIS data to {csv_path}")
    print(f"Saved plot to {plot_path}")
    print(f"Saved test-set model weight data to {test_weights_csv_path}")
    print(f"Saved test-set model weight grid plot to {test_weights_plot_path}")
    print("Models plotted:", ", ".join(sorted(wis_by_date["model"].unique())))


if __name__ == "__main__":
    main()
