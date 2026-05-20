import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import entropy
import warnings
warnings.filterwarnings("ignore")

plt.style.use("seaborn-v0_8-whitegrid")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "processed_data"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

detection = pd.read_csv(PROCESSED_DIR / "detection_long.csv")
site_info = pd.read_csv(PROCESSED_DIR / "site_info.csv")

detection = detection[detection["camera_id"] != "Grand Total"]

years = sorted(detection["year"].unique())

# ========== Fig 7: Bubble Maps ==========
fig7_data = detection.groupby(["camera_id", "year"]).agg(
    total_events=("events", "sum"),
    species_count=("species", "nunique")
).reset_index()

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

vmin_sp, vmax_sp = fig7_data["species_count"].min(), fig7_data["species_count"].max()
size_max = fig7_data["total_events"].max()
bubble_scale = 500

for ax, year in zip(axes, years):
    year_data = fig7_data[fig7_data["year"] == year]
    year_data = year_data.merge(site_info[["camera_id", "longitude", "latitude"]], on="camera_id", how="left")
    sizes = year_data["total_events"] / size_max * bubble_scale + 20
    sc = ax.scatter(
        year_data["longitude"], year_data["latitude"],
        s=sizes, c=year_data["species_count"],
        cmap="YlOrRd", edgecolors="gray", linewidth=0.5, alpha=0.8,
        vmin=vmin_sp, vmax=vmax_sp
    )
    ax.set_xlim(127.9, 128.6)
    ax.set_ylim(43.7, 44.3)
    ax.set_title(f"{int(year)}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal")

cbar = fig.colorbar(sc, ax=axes, orientation="horizontal", fraction=0.02, pad=0.05)
cbar.set_label("Species Count")

legend_elements = []
for val, label in [(20, "20"), (50, "50"), (100, "100+")]:
    pt_size = 2 * np.sqrt((val / size_max * bubble_scale + 20) / np.pi)
    legend_elements.append(plt.Line2D([0], [0], marker="o", color="gray",
                                      markerfacecolor="lightgray",
                                      markersize=pt_size, label=label, linestyle="None"))
fig.legend(handles=legend_elements, title="Total Events", loc="lower center",
           ncol=3, bbox_to_anchor=(0.5, -0.02))
fig.suptitle("Bubble Maps of Total Events and Species Count (2015-2020)",
             fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(FIGURE_DIR / "fig7_bubble_maps_2015_2020.png", dpi=300, bbox_inches="tight")
plt.close()
print("Fig 7 saved: fig7_bubble_maps_2015_2020.png")

# ========== Fig 8: Diversity Hotspot Migration ==========
def calc_shannon_year(group):
    events = group["events"].values
    total = events.sum()
    if total == 0:
        return 0.0
    props = events / total
    return entropy(props, base=np.e)

diversity_year = detection.groupby(["camera_id", "year"]).apply(calc_shannon_year).reset_index()
diversity_year.columns = ["camera_id", "year", "shannon"]

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

vmin_div, vmax_div = diversity_year["shannon"].min(), diversity_year["shannon"].max()

for ax, year in zip(axes, years):
    year_div = diversity_year[diversity_year["year"] == year]
    year_div = year_div.merge(site_info[["camera_id", "longitude", "latitude"]], on="camera_id", how="left")
    sc = ax.scatter(
        year_div["longitude"], year_div["latitude"],
        s=120, c=year_div["shannon"],
        cmap="viridis", edgecolors="black", linewidth=0.3, alpha=0.9,
        vmin=vmin_div, vmax=vmax_div
    )
    ax.set_xlim(127.9, 128.6)
    ax.set_ylim(43.7, 44.3)
    ax.set_title(f"{int(year)}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal")

cbar = fig.colorbar(sc, ax=axes, orientation="horizontal", fraction=0.02, pad=0.05)
cbar.set_label("Shannon Diversity Index")
fig.suptitle("Diversity Hotspot Migration (2015-2020)",
             fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(FIGURE_DIR / "fig8_diversity_hotspot_migration.png", dpi=300, bbox_inches="tight")
plt.close()
print("Fig 8 saved: fig8_diversity_hotspot_migration.png")

# ========== Fig 9: Species Range Dynamics ==========
target_species = ["狍 Capreolus pygargus", "野猪 Sus scrofa"]

range_records = []
for sp in target_species:
    for i, year in enumerate(years):
        year = int(year)
        prev_year = int(years[i - 1]) if i > 0 else None
        sites_this_year = set(detection[(detection["species"] == sp) & (detection["year"] == year)]["camera_id"])
        sites_occupied = len(sites_this_year)
        new_sites = set()
        lost_sites = set()
        if prev_year is not None:
            sites_prev_year = set(detection[(detection["species"] == sp) & (detection["year"] == prev_year)]["camera_id"])
            new_sites = sites_this_year - sites_prev_year
            lost_sites = sites_prev_year - sites_this_year
        range_records.append({
            "species": sp, "year": year,
            "sites_occupied": sites_occupied,
            "new_sites": ";".join(sorted(new_sites)) if new_sites else "",
            "lost_sites": ";".join(sorted(lost_sites)) if lost_sites else ""
        })

range_df = pd.DataFrame(range_records)
range_df.to_csv(PROCESSED_DIR / "species_range_change.csv", index=False, encoding="utf-8-sig")
print("species_range_change.csv saved")

fig, axes = plt.subplots(2, len(years), figsize=(22, 8))
species_labels_short = ["Roe deer", "Wild boar"]
site_coords = site_info[["camera_id", "longitude", "latitude"]]

for row, sp in enumerate(target_species):
    for col, year in enumerate(years):
        year = int(year)
        ax = axes[row, col]
        ax.set_xlim(127.9, 128.6)
        ax.set_ylim(43.7, 44.3)
        ax.set_aspect("equal")
        ax.set_title(f"{species_labels_short[row]} - {year}", fontsize=11)
        if col == 0:
            ax.set_ylabel("Latitude")
        ax.set_xlabel("Longitude")

        ax.scatter(site_coords["longitude"], site_coords["latitude"],
                   s=15, c="lightgray", alpha=0.4, edgecolors="none")

        sp_year = detection[(detection["species"] == sp) & (detection["year"] == year)]
        sp_prev = detection[(detection["species"] == sp) & (detection["year"] == year - 1)] if year > 2015 else pd.DataFrame()

        current_sites = set(sp_year["camera_id"])
        prev_sites = set(sp_prev["camera_id"]) if not sp_prev.empty else set()

        stable = current_sites & prev_sites
        new = current_sites - prev_sites
        lost = prev_sites - current_sites

        if stable:
            stable_geo = site_coords[site_coords["camera_id"].isin(stable)]
            ax.scatter(stable_geo["longitude"], stable_geo["latitude"],
                       s=80, c="#2ecc71", edgecolors="black", linewidth=0.5,
                       alpha=0.85, label="Stable", zorder=3)
        if new:
            new_geo = site_coords[site_coords["camera_id"].isin(new)]
            ax.scatter(new_geo["longitude"], new_geo["latitude"],
                       s=80, c="#e74c3c", edgecolors="black", linewidth=0.5,
                       alpha=0.85, marker="^", label="New", zorder=4)
        if lost:
            lost_geo = site_coords[site_coords["camera_id"].isin(lost)]
            ax.scatter(lost_geo["longitude"], lost_geo["latitude"],
                       s=80, c="#3498db", edgecolors="black", linewidth=0.5,
                       alpha=0.85, marker="v", label="Lost", zorder=3)

for ax in axes[0, :]:
    ax.legend(loc="upper right", fontsize=8, markerscale=0.6)

fig.suptitle("Species Range Dynamics: New, Lost, and Stable Sites (2015-2020)",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(FIGURE_DIR / "fig9_species_range_dynamics.png", dpi=300, bbox_inches="tight")
plt.close()
print("Fig 9 saved: fig9_species_range_dynamics.png")
print("All visualizations complete!")
