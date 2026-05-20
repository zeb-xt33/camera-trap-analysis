import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
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
all_species = sorted(detection["species"].unique())
n_species = len(all_species)

color_map = plt.cm.tab20(np.linspace(0, 1, 20))
color_map = np.vstack([color_map, plt.cm.tab20b(np.linspace(0, 1, 20))])
color_map = np.vstack([color_map, plt.cm.Set3(np.linspace(0, 1, 10))])
species_colors = {sp: color_map[i % 50] for i, sp in enumerate(all_species)}

fig, ax = plt.subplots(figsize=(12, 10))
ax.set_xlim(127.9, 128.6)
ax.set_ylim(43.7, 44.3)
ax.set_aspect("equal")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

title = ax.set_title("", fontsize=16, fontweight="bold")

scatter_plots = {}
for sp in all_species:
    sc = ax.scatter([], [], s=0, c=[species_colors[sp]], label=sp[:10],
                    alpha=0.7, edgecolors="none", zorder=3)
    scatter_plots[sp] = sc

ax.scatter(site_info["longitude"], site_info["latitude"],
           s=20, c="lightgray", alpha=0.5, edgecolors="none", zorder=1)

legend = ax.legend(loc="upper left", fontsize=6, ncol=2,
                   markerscale=0.8, framealpha=0.8,
                   bbox_to_anchor=(1.02, 1))
legend.set_title("Species", prop={"size": 8})

def update(frame):
    year = years[frame]
    title.set_text(f"Species Detection Distribution - {int(year)}")
    for sp in all_species:
        sp_year = detection[(detection["species"] == sp) & (detection["year"] == year)]
        if sp_year.empty:
            scatter_plots[sp].set_offsets(np.empty((0, 2)))
            scatter_plots[sp].set_sizes([])
        else:
            coords = np.column_stack([sp_year["longitude"].values, sp_year["latitude"].values])
            scatter_plots[sp].set_offsets(coords)
            scatter_plots[sp].set_sizes(np.full(len(sp_year), 60))
    return [scatter_plots[sp] for sp in all_species] + [title]

n_frames = len(years)
anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=1500, repeat=True)

plt.tight_layout()
anim.save(FIGURE_DIR / "fig10_annual_comparison.gif", writer="pillow", fps=1, dpi=150)
plt.close()
print("Animation saved: fig10_annual_comparison.gif")
