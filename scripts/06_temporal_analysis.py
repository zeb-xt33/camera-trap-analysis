import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams.update({"figure.dpi": 150, "font.family": "sans-serif", "font.sans-serif": ["SimSun", "DejaVu Sans"]})

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DETECTION_LONG = PROJECT_ROOT / "processed_data" / "detection_long.csv"
SITE_INFO = PROJECT_ROOT / "processed_data" / "site_info.csv"
OUTPUT_DIR = PROJECT_ROOT / "processed_data"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"

KEY_SPECIES = {
    "狍": "Capreolus pygargus",
    "野猪": "Sus scrofa",
    "紫貂": "Martes zibellina",
    "亚洲狗獾": "Meles leucurus",
    "马鹿": "Cervus elaphus",
    "松鼠": "Sciurus vulgaris",
}


def load_data():
    det_long = pd.read_csv(DETECTION_LONG)
    site_info = pd.read_csv(SITE_INFO)
    return det_long, site_info


def aggregate_by_year(det_long):
    annual = det_long.groupby("year").agg(
        total_events=("events", "sum"),
        total_camera_days=("working_days", "sum"),
    ).reset_index()

    species_per_year = det_long[det_long["events"] > 0].groupby("year")["species"].nunique().reset_index()
    species_per_year.columns = ["year", "total_species"]

    annual = annual.merge(species_per_year, on="year")
    return annual


def calc_diversity(det_long, annual):
    rai_data = det_long.groupby(["year", "species"])["events"].sum().reset_index()
    rai_data = rai_data.merge(annual[["year", "total_camera_days"]], on="year")
    rai_data["rai"] = rai_data["events"] / rai_data["total_camera_days"]

    diversity_list = []
    for year, group in rai_data.groupby("year"):
        rai_vals = group["rai"].values
        rai_vals = rai_vals[rai_vals > 0]
        p = rai_vals / rai_vals.sum()
        shannon = -np.sum(p * np.log(p))
        simpson = 1 - np.sum(p ** 2)
        diversity_list.append({"year": year, "shannon": shannon, "simpson": simpson})

    diversity_df = pd.DataFrame(diversity_list)
    annual = annual.merge(diversity_df, on="year")
    return annual


def calc_turnover(det_long):
    species_by_year = det_long[det_long["events"] > 0].groupby("year")["species"].apply(set).reset_index()
    species_by_year.columns = ["year", "species_set"]
    species_by_year = species_by_year.sort_values("year")

    turnover_list = []
    years = species_by_year["year"].tolist()
    for i in range(len(years) - 1):
        y1, y2 = years[i], years[i + 1]
        set1 = species_by_year.loc[species_by_year["year"] == y1, "species_set"].iloc[0]
        set2 = species_by_year.loc[species_by_year["year"] == y2, "species_set"].iloc[0]
        jaccard = len(set1 & set2) / len(set1 | set2) if len(set1 | set2) > 0 else 0
        turnover_list.append({"year_from": y1, "year_to": y2, "jaccard_dissimilarity": 1 - jaccard})

    return pd.DataFrame(turnover_list)


def calc_rai_by_species(det_long, annual):
    rai_overall = det_long.groupby(["year", "species"])["events"].sum().reset_index()
    rai_overall = rai_overall.merge(annual[["year", "total_camera_days"]], on="year")
    rai_overall["rai"] = rai_overall["events"] / rai_overall["total_camera_days"]
    return rai_overall


def calc_site_annual_summary(det_long):
    site_annual = det_long.groupby(["camera_id", "year"]).agg(
        species_count=("species", "nunique"),
        total_events=("events", "sum"),
    ).reset_index()
    return site_annual


def filter_key_species(rai_overall):
    pattern = "|".join(KEY_SPECIES.values())
    mask = rai_overall["species"].str.contains(pattern, case=False, na=False)
    return rai_overall[mask].copy()


def extract_cn_name(species_str):
    for cn_name, en_name in KEY_SPECIES.items():
        if en_name in species_str:
            return cn_name
    return species_str


def save_csv_files(annual, rai_overall):
    annual_out = annual[["year", "total_events", "total_species", "shannon", "simpson", "total_camera_days"]]
    annual_out.to_csv(OUTPUT_DIR / "annual_diversity.csv", index=False)
    print(f"  OK  annual_diversity.csv")

    key_data = filter_key_species(rai_overall)
    key_data["species_cn"] = key_data["species"].apply(extract_cn_name)
    key_pivot = key_data.pivot_table(
        index="year", columns="species_cn", values="rai", aggfunc="first"
    ).reset_index()
    key_pivot.to_csv(OUTPUT_DIR / "annual_rai_trend.csv", index=False)
    print(f"  OK  annual_rai_trend.csv")


def plot_annual_trend(annual, turnover):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    ax1 = axes[0]
    years = annual["year"].values
    ax1.bar(years - 0.15, annual["total_species"].values, width=0.3, color="#4C72B0", alpha=0.85, label="物种数")
    ax1_twin = ax1.twinx()
    ax1_twin.plot(years, annual["total_events"].values, "o-", color="#DD8452", linewidth=2, markersize=6, label="检测事件数")
    ax1.set_xlabel("年份")
    ax1.set_ylabel("物种数", color="#4C72B0")
    ax1_twin.set_ylabel("检测事件数", color="#DD8452")
    ax1_twin.tick_params(axis="y", colors="#DD8452")
    ax1.tick_params(axis="y", colors="#4C72B0")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines1_twin, labels1_twin = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines1_twin, labels1 + labels1_twin, loc="upper left")

    ax2 = axes[1]
    ax2.plot(years, annual["shannon"].values, "s-", color="#55A868", linewidth=2, markersize=6, label="Shannon")
    ax2_twin = ax2.twinx()
    ax2_twin.plot(years, annual["simpson"].values, "D-", color="#C44E52", linewidth=2, markersize=6, label="Simpson")
    ax2.set_xlabel("年份")
    ax2.set_ylabel("Shannon 指数", color="#55A868")
    ax2_twin.set_ylabel("Simpson 指数", color="#C44E52")
    ax2_twin.tick_params(axis="y", colors="#C44E52")
    ax2.tick_params(axis="y", colors="#55A868")
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines2_twin, labels2_twin = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines2 + lines2_twin, labels2 + labels2_twin, loc="upper left")

    ax3 = axes[2]
    mid_years = (turnover["year_from"].values + turnover["year_to"].values) / 2
    ax3.bar(mid_years, turnover["jaccard_dissimilarity"].values, width=0.6, color="#8172B2", alpha=0.85)
    ax3.set_xlabel("年份")
    ax3.set_ylabel("Jaccard 相异度")
    ax3.set_xticks(turnover["year_from"].values)
    ax3.set_xticklabels([f"{int(yf)}-{int(yt)}" for yf, yt in zip(turnover["year_from"].values, turnover["year_to"].values)])

    for ax in axes:
        ax.set_title(["年际物种数与检测数", "多样性指数年际变化", "物种周转率年际变化"][axes.tolist().index(ax)])

    plt.tight_layout()
    fig_path = FIG_DIR / "fig5_annual_trend.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    print(f"  OK  fig5_annual_trend.png")


def plot_key_species_trend(rai_overall):
    fig, ax = plt.subplots(figsize=(12, 6))

    key_data = filter_key_species(rai_overall)
    colors = sns.color_palette("Set2", n_colors=len(KEY_SPECIES))

    for i, (cn_name, en_name) in enumerate(KEY_SPECIES.items()):
        sp_data = key_data[key_data["species"].str.contains(en_name, case=False, na=False)].sort_values("year")
        ax.plot(sp_data["year"].values, sp_data["rai"].values, "o-", color=colors[i],
                linewidth=2, markersize=6, label=cn_name)

    ax.set_xlabel("年份")
    ax.set_ylabel("相对多度指数 (RAI)")
    ax.legend(loc="best", frameon=True)
    ax.set_title("关键物种 RAI 年际趋势")

    plt.tight_layout()
    fig_path = FIG_DIR / "fig6_key_species_trend.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    print(f"  OK  fig6_key_species_trend.png")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    print("Loading data...")
    det_long, site_info = load_data()
    print(f"  detection_long: {det_long.shape}")
    print(f"  site_info:      {site_info.shape}")

    print("\nAggregating by year...")
    annual = aggregate_by_year(det_long)
    print(annual[["year", "total_events", "total_species", "total_camera_days"]].to_string(index=False))

    print("\nCalculating diversity indices...")
    annual = calc_diversity(det_long, annual)

    print("\nCalculating species turnover...")
    turnover = calc_turnover(det_long)

    print("\nCalculating RAI for key species...")
    rai_overall = calc_rai_by_species(det_long, annual)

    print("\nCalculating site-year summary...")
    site_annual = calc_site_annual_summary(det_long)
    print(f"  site-year records: {len(site_annual)}")

    print("\nSaving CSV files...")
    save_csv_files(annual, rai_overall)

    print("\nGenerating figures...")
    plot_annual_trend(annual, turnover)
    plot_key_species_trend(rai_overall)

    print("\nDone!")


if __name__ == "__main__":
    main()
