import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import kruskal

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sns.set_style('whitegrid')
plt.rcParams['font.family'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False

det_long = pd.read_csv(PROJECT_ROOT / 'processed_data' / 'detection_long.csv')
site_info = pd.read_csv(PROJECT_ROOT / 'processed_data' / 'site_info.csv')
site_info = site_info[site_info['camera_id'] != '合计'].copy()
site_info['elevation'] = pd.to_numeric(site_info['elevation'], errors='coerce')

det_long['events'] = pd.to_numeric(det_long['events'], errors='coerce').fillna(0).astype(int)

def calc_diversity(grp):
    events = grp['events'].values.astype(float)
    total = events.sum()
    if total == 0:
        return pd.Series({'shannon': 0, 'simpson': 0, 'richness': 0, 'evenness': 0})
    p = events / total
    shannon = -np.sum(p * np.log(p))
    simpson = 1 - np.sum(p ** 2)
    richness = int((events > 0).sum())
    evenness = shannon / np.log(richness) if richness > 1 else 0
    return pd.Series({'shannon': shannon, 'simpson': simpson,
                      'richness': richness, 'evenness': evenness})

diversity = det_long.groupby('camera_id').apply(calc_diversity).reset_index()
div_site = diversity.merge(
    site_info[['camera_id', 'elevation', 'habitat_type']],
    on='camera_id', how='left'
)
div_site = div_site.dropna(subset=['elevation', 'habitat_type'])

elev = div_site['elevation'].values
shan = div_site['shannon'].values
simp = div_site['simpson'].values

coeff_lin_s = np.polyfit(elev, shan, 1)
pred_lin_s = np.polyval(coeff_lin_s, elev)
ss_res_s = np.sum((shan - pred_lin_s) ** 2)
ss_tot_s = np.sum((shan - np.mean(shan)) ** 2)
r2_lin_s = 1 - ss_res_s / ss_tot_s

coeff_quad_s = np.polyfit(elev, shan, 2)
pred_quad_s = np.polyval(coeff_quad_s, elev)
r2_quad_s = 1 - np.sum((shan - pred_quad_s) ** 2) / ss_tot_s

coeff_lin_si = np.polyfit(elev, simp, 1)
pred_lin_si = np.polyval(coeff_lin_si, elev)
ss_res_si = np.sum((simp - pred_lin_si) ** 2)
ss_tot_si = np.sum((simp - np.mean(simp)) ** 2)
r2_lin_si = 1 - ss_res_si / ss_tot_si

coeff_quad_si = np.polyfit(elev, simp, 2)
pred_quad_si = np.polyval(coeff_quad_si, elev)
r2_quad_si = 1 - np.sum((simp - pred_quad_si) ** 2) / ss_tot_si

habitats = div_site['habitat_type'].dropna().unique()
groups = [div_site.loc[div_site['habitat_type'] == h, 'shannon'].dropna() for h in habitats]
groups = [g for g in groups if len(g) > 1]
if len(groups) >= 2:
    kw_stat, kw_p = kruskal(*groups)
else:
    kw_stat, kw_p = np.nan, np.nan

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

elev_range = np.linspace(elev.min(), elev.max(), 100)

axes[0].scatter(elev, shan, alpha=0.6, color='#3498db', edgecolor='black', zorder=3)
axes[0].plot(elev_range, np.polyval(coeff_lin_s, elev_range), 'r--',
             label=f'Linear (R²={r2_lin_s:.3f})', linewidth=1.5)
axes[0].plot(elev_range, np.polyval(coeff_quad_s, elev_range), 'g-',
             label=f'Quadratic (R²={r2_quad_s:.3f})', linewidth=1.5)
axes[0].set_xlabel('Elevation (m)')
axes[0].set_ylabel('Shannon Index')
axes[0].set_title('Shannon Index vs Elevation', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=9)

axes[1].scatter(elev, simp, alpha=0.6, color='#e74c3c', edgecolor='black', zorder=3)
axes[1].plot(elev_range, np.polyval(coeff_lin_si, elev_range), 'r--',
             label=f'Linear (R²={r2_lin_si:.3f})', linewidth=1.5)
axes[1].plot(elev_range, np.polyval(coeff_quad_si, elev_range), 'g-',
             label=f'Quadratic (R²={r2_quad_si:.3f})', linewidth=1.5)
axes[1].set_xlabel('Elevation (m)')
axes[1].set_ylabel('Simpson Index')
axes[1].set_title('Simpson Index vs Elevation', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=9)

order = div_site.groupby('habitat_type')['shannon'].median().sort_values(ascending=False).index
sns.boxplot(x='habitat_type', y='shannon', hue='habitat_type', data=div_site, ax=axes[2],
            palette='Set2', order=order, legend=False)
sns.stripplot(x='habitat_type', y='shannon', data=div_site, ax=axes[2],
              color='black', alpha=0.4, size=4, order=order)
axes[2].set_xlabel('Habitat Type')
axes[2].set_ylabel('Shannon Index')
kw_label = f'KW p={kw_p:.4f}' if not np.isnan(kw_p) else 'KW: N/A'
axes[2].set_title(f'Shannon Index by Habitat ({kw_label})',
                  fontsize=14, fontweight='bold')
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(PROJECT_ROOT / 'outputs' / 'figures' / 'fig2_diversity_vs_elevation.png',
            dpi=300, bbox_inches='tight')
plt.close()

div_site.to_csv(PROJECT_ROOT / 'processed_data' / 'site_diversity.csv', index=False)
