import json
import pandas as pd
import folium
import matplotlib.pyplot as plt
from pathlib import Path
from urllib.request import urlopen

# The official public source is weekly NNDSS data from CDC
# Check https://data.cdc.gov/browse?category=NNDSS&sortBy=relevance&page=1&pageSize=20
url = "https://data.cdc.gov/api/views/x9gk-5huc/rows.csv?accessType=DOWNLOAD"
raw_csv_candidates = [
    Path("00_cdc_raw_download.csv"),
    Path("cdc_raw.csv"),
]

# US states only (exclude territories and DC)
US_STATES = {
    "ALABAMA", "ALASKA", "ARIZONA", "ARKANSAS", "CALIFORNIA", "COLORADO", "CONNECTICUT",
    "DELAWARE", "FLORIDA", "GEORGIA", "HAWAII", "IDAHO", "ILLINOIS", "INDIANA", "IOWA",
    "KANSAS", "KENTUCKY", "LOUISIANA", "MAINE", "MARYLAND", "MASSACHUSETTS", "MICHIGAN",
    "MINNESOTA", "MISSISSIPPI", "MISSOURI", "MONTANA", "NEBRASKA", "NEVADA", "NEW HAMPSHIRE",
    "NEW JERSEY", "NEW MEXICO", "NEW YORK", "NORTH CAROLINA", "NORTH DAKOTA", "OHIO",
    "OKLAHOMA", "OREGON", "PENNSYLVANIA", "RHODE ISLAND", "SOUTH CAROLINA", "SOUTH DAKOTA",
    "TENNESSEE", "TEXAS", "UTAH", "VERMONT", "VIRGINIA", "WASHINGTON", "WEST VIRGINIA",
    "WISCONSIN", "WYOMING"
}

FILTER_DISEASE = "Chlamydia trachomatis infection"


def save_stage(table, filename, description):
    """Save a CSV checkpoint and print enough context to follow the pipeline."""
    table.to_csv(filename, index=False)
    print(f"Saved {description}: {filename} ({table.shape[0]} rows, {table.shape[1]} columns)")


def load_raw_cdc_data():
    for candidate in raw_csv_candidates:
        if candidate.exists():
            print(f"Loading raw CDC data from existing checkpoint: {candidate}")
            return pd.read_csv(candidate)

    print("Downloading raw CDC data from CDC...")
    with urlopen(url, timeout=30) as response:
        return pd.read_csv(response)




df = load_raw_cdc_data()
save_stage(df, "00_cdc_raw_download.csv", "stage 00 raw CDC download")

label_col = "Label"
area_col = "Reporting Area"
year_col = "Current MMWR Year"
week_col = "MMWR WEEK"
current_week_col = "Current week"

print("Report columns:", df.columns.tolist())
print("Sample diseases:", df[label_col].dropna().unique()[:40])

# Keep only US states
states_df = df[df[area_col].astype(str).str.upper().isin(US_STATES)].copy()
states_df[label_col] = states_df[label_col].astype(str).str.strip()
states_df[area_col] = states_df[area_col].astype(str).str.strip().str.title()
print(f"\nRows after filtering to US states only: {states_df.shape[0]}")
print(f"States kept: {sorted(states_df[area_col].dropna().unique())}")

# Filter for the target disease
filtered = states_df[states_df[label_col].str.lower() == FILTER_DISEASE.lower()].copy()

save_stage(
    filtered,
    "01_chlamydia_by_state_raw.csv",
    "stage 01 filtered Chlamydia trachomatis infection rows for US states",
)

# Clean weekly incidence values
filtered[current_week_col] = (
    filtered[current_week_col]
    .replace({"-": 0, "U": 0, "N": 0, "NN": 0, "NP": 0, "NC": 0})
    .fillna(0)
    .astype("Int64")
)

filtered["week_str"] = filtered[week_col].astype(str).str.zfill(2)
filtered["date"] = pd.to_datetime(
    filtered[year_col].astype(str) + filtered["week_str"] + "1",
    format="%Y%W%w",
    errors="coerce"
)
filtered = filtered[filtered["date"].notna()].copy()

# Weekly incidence per state
weekly_by_state = (
    filtered.groupby(["date", area_col], as_index=False)[current_week_col]
    .sum()
    .rename(columns={current_week_col: "weekly_cases"})
)
weekly_pivot = weekly_by_state.pivot(index="date", columns=area_col, values="weekly_cases").fillna(0)
weekly_pivot = weekly_pivot.sort_index()
weekly_df = weekly_pivot.reset_index()
weekly_df.columns.name = None

save_stage(
    weekly_df,
    "02_chlamydia_weekly_by_state.csv",
    "stage 02 weekly Chlamydia incidence by state",
)

# Create a dataframe for a selected set of states (weekly reports)
# User-specified list (normalized to Title Case earlier)
selected_list = [
    "Alaska", "Mississippi", "New Mexico", "Florida", "Minnesota", "California",
    "Arizona", "Kansas", "Louisiana", "West Virginia", "Montana", "Maine",
    "Idaho", "Oklahoma", "South Carolina", "New York", "Michigan", "Virginia",
    "Washington", "Pennsylvania", "Utah", "Kentucky", "Wisconsin", "Alabama",
    "Oregon"
]

selected_states = weekly_by_state[weekly_by_state[area_col].isin(selected_list)].copy()
selected_states = selected_states.sort_values([area_col, "date"])  # order for readability

save_stage(
    selected_states,
    "06_chlamydia_selected_states_weekly.csv",
    "stage 06 weekly reports for selected states",
)

# Also produce a wide pivot (one column per selected state) for convenience
selected_wide = selected_states.pivot(index="date", columns=area_col, values="weekly_cases").fillna(0).sort_index()
selected_wide_df = selected_wide.reset_index()
selected_wide_df.columns.name = None
# Replace zeros with missing values in the saved wide CSV per user request
selected_wide_df = selected_wide_df.replace(0, pd.NA)
save_stage(
    selected_wide_df,
    "07_chlamydia_selected_states_weekly_wide.csv",
    "stage 07 weekly wide table for selected states",
)

print("\nSelected states weekly (long) sample:")
print(selected_states.head())
print(f"Selected states weekly (wide) shape: {selected_wide_df.shape}")
# Cumulative totals by state
state_totals = (
    weekly_by_state.groupby(area_col, as_index=False)["weekly_cases"]
    .sum()
    .rename(columns={"weekly_cases": "cumulative_cases"})
    .sort_values("cumulative_cases", ascending=False)
)

save_stage(
    state_totals,
    "03_chlamydia_cumulative_by_state.csv",
    "stage 03 cumulative Chlamydia cases by state",
)

print(f"\nTop 20 states by cumulative Chlamydia cases:")
print(state_totals.head(20))

# Identify zero-incidence weeks for the top 20 states
top20_states = state_totals.head(20)[area_col].tolist()
zero_weeks = weekly_by_state[
    weekly_by_state[area_col].isin(top20_states) &
    (weekly_by_state["weekly_cases"] == 0)
].sort_values([area_col, "date"])

save_stage(
    zero_weeks,
    "04_chlamydia_top20_zero_weeks.csv",
    "stage 04 zero-incidence weeks for top 20 Chlamydia states",
)
print(f"Saved state-week zero incidence records for top 20 states: {zero_weeks.shape[0]} rows")

# Zero-week counts for all 50 states
zero_counts = (
    weekly_by_state[weekly_by_state["weekly_cases"] == 0]
    .groupby(area_col, as_index=False)["weekly_cases"]
    .count()
    .rename(columns={"weekly_cases": "zero_week_count"})
    .sort_values("zero_week_count", ascending=False)
)

save_stage(
    zero_counts,
    "05_chlamydia_zero_week_counts_by_state.csv",
    "stage 05 zero weekly incidence counts for all 50 states",
)
print(f"Saved zero-week counts for all states: {zero_counts.shape[0]} rows")

# Plot zero-week counts for all states
plt.figure(figsize=(16, 12))
plt.bar(zero_counts[area_col], zero_counts["zero_week_count"], color="tab:purple")
plt.xlabel("State")
plt.ylabel("Zero-incidence weeks")
plt.title("Number of zero-incidence Chlamydia weeks by state")
plt.xticks(rotation=90)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("chlamydia_zero_week_counts_all_states.png", dpi=150, bbox_inches="tight")
print("Saved plot: chlamydia_zero_week_counts_all_states.png")
plt.close()

# Plot weekly incidence time series for the top 10 cumulative states
top10_states = state_totals.head(10)[area_col].tolist()
plot_data = weekly_pivot[top10_states]

plt.figure(figsize=(16, 10))
for state in top10_states:
    plt.plot(plot_data.index, plot_data[state], marker="o", linewidth=1.5, label=state)

plt.xlabel("Week start date")
plt.ylabel("Weekly Chlamydia cases")
plt.title("Weekly Chlamydia trachomatis incidence for top 10 US states")
plt.legend(loc="upper left", fontsize=9)
plt.grid(alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("chlamydia_weekly_top10_states.png", dpi=150, bbox_inches="tight")
print("Saved plot: chlamydia_weekly_top10_states.png")
plt.close()

# Plot cumulative cases for all 50 states as a horizontal bar chart
plt.figure(figsize=(16, 14))
plt.barh(state_totals[area_col].iloc[::-1], state_totals["cumulative_cases"].iloc[::-1], color="tab:blue")
plt.xlabel("Cumulative Chlamydia cases")
plt.ylabel("State")
plt.title("Cumulative Chlamydia trachomatis cases across all 50 US states")
plt.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("chlamydia_all_states_cumulative.png", dpi=150, bbox_inches="tight")
print("Saved plot: chlamydia_all_states_cumulative.png")
plt.close()


