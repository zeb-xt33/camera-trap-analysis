import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sns.set_style('whitegrid')
plt.rcParams['font.family'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False

species = pd.read_csv(PROJECT_ROOT / 'processed_data' / 'species_list.csv')
species = species[pd.to_numeric(species['rai'], errors='coerce').notna()].copy()
species['rai'] = species['rai'].astype(float)
species['independent_events'] = species['independent_events'].astype(int)

bird_keywords = ['鸟', '鸦', '鸫', '鹀', '啄木鸟', '杜鹃', '鹡鸰', '鹰', '鵟', '鸮',
                 '雉', '榛鸡', '鹬', '鸠', '鸳鸯', '䴓', '雀', '鸲']

def infer_class(row):
    name = str(row['species'])
    if any(kw in name for kw in bird_keywords):
        return 'Bird'
    return 'Mammal'

species['class'] = species.apply(infer_class, axis=1)

prot_levels = ['Ⅰ', 'Ⅱ']
species['prot_group'] = species['national_protection'].apply(
    lambda x: x if str(x) in prot_levels else 'Unprotected'
)
prot_counts = species.groupby(['class', 'prot_group']).size().unstack(fill_value=0)
for col in ['Ⅰ', 'Ⅱ', 'Unprotected']:
    if col not in prot_counts.columns:
        prot_counts[col] = 0
prot_counts = prot_counts[['Ⅰ', 'Ⅱ', 'Unprotected']]

rai_ranked = species.sort_values('rai', ascending=False).reset_index(drop=True)
rai_ranked['rank'] = range(1, len(rai_ranked) + 1)
rai_ranked[['rank', 'species', 'class', 'national_protection',
            'independent_events', 'rai']].to_csv(
    PROJECT_ROOT / 'processed_data' / 'species_rai.csv', index=False
)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

prot_counts.plot(kind='bar', ax=axes[0],
                 color=['#e74c3c', '#e67e22', '#95a5a6'],
                 edgecolor='black', width=0.7)
axes[0].set_title('Species Count by Protection Level', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Class')
axes[0].set_ylabel('Number of Species')
axes[0].legend(title='Protection Level', fontsize=9)
axes[0].tick_params(axis='x', rotation=0)

top15 = rai_ranked.head(15)
color_map = {'Mammal': '#3498db', 'Bird': '#2ecc71'}
bar_colors = [color_map[c] for c in top15['class']]
axes[1].barh(range(len(top15)), top15['rai'], color=bar_colors, edgecolor='black', height=0.7)
axes[1].set_yticks(range(len(top15)))
axes[1].set_yticklabels(top15['species'].values, fontsize=9)
axes[1].set_xlabel('RAI')
axes[1].set_title('Top 15 Species by RAI', fontsize=14, fontweight='bold')
axes[1].invert_yaxis()
legend_elements = [
    Patch(facecolor='#3498db', label='Mammal'),
    Patch(facecolor='#2ecc71', label='Bird')
]
axes[1].legend(handles=legend_elements, title='Class', fontsize=9)

rank_vals = rai_ranked['rai'].values
axes[2].plot(range(1, len(rank_vals) + 1), rank_vals, 'o-',
             color='#8e44ad', markersize=4, linewidth=1.5)
axes[2].set_xlabel('Species Rank')
axes[2].set_ylabel('RAI')
axes[2].set_title('Species Rank-Abundance Curve', fontsize=14, fontweight='bold')
axes[2].set_xscale('log')

plt.tight_layout()
plt.savefig(PROJECT_ROOT / 'outputs' / 'figures' / 'fig1_species_composition.png',
            dpi=300, bbox_inches='tight')
plt.close()
