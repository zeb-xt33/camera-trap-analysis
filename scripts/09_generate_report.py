import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

species = pd.read_csv(PROJECT_ROOT / 'processed_data' / 'species_list.csv')
site_info = pd.read_csv(PROJECT_ROOT / 'processed_data' / 'site_info.csv')
site_div = pd.read_csv(PROJECT_ROOT / 'processed_data' / 'site_diversity.csv')
species_rai = pd.read_csv(PROJECT_ROOT / 'processed_data' / 'species_rai.csv')
ann_div = pd.read_csv(PROJECT_ROOT / 'processed_data' / 'annual_diversity.csv')
ann_rai = pd.read_csv(PROJECT_ROOT / 'processed_data' / 'annual_rai_trend.csv')
phi = pd.read_csv(PROJECT_ROOT / 'processed_data' / 'significant_pairs.csv')
range_chg = pd.read_csv(PROJECT_ROOT / 'processed_data' / 'species_range_change.csv')

mammal_count = len([s for s in species['species'] if any(kw in str(s) for kw in ['虎','猪','狍','鹿','熊','獾','貉','麝','鼬','貂','豹','松鼠','兔','猬','獐'])])
bird_count = len(species) - mammal_count

top5 = species_rai.head(5)
total_camera_days = site_info['working_days'].sum()
mammal_events = site_info['independent_events_mammal'].sum()
bird_events = site_info['independent_events_bird'].sum()

prot_i = len(species[species['national_protection'].astype(str).str.contains('Ⅰ')])
prot_ii = len(species[species['national_protection'].astype(str).str.contains('Ⅱ')])

sig_pairs = len(phi)

mean_shannon = site_div['shannon'].mean()
mean_elevation = site_info['elevation'].mean()

report = f"""# 张广才岭野生动物红外相机监测数据分析报告

## 数据引用

王大伟, 程帅, 冯佳伟, 王天明 (2025) 东北地区张广才岭2015–2020年野生动物红外相机监测数据集. 生物多样性, 33, 24384. doi: 10.17520/biods.2024384

---

## 1. 数据集概况

- **监测区域**: 张广才岭南部 (128°00′–128°32′E, 43°44′–44°16′N)
- **海拔范围**: {int(site_info['elevation'].min())}–{int(site_info['elevation'].max())} m (平均 {int(mean_elevation)} m)
- **相机位点数**: {len(site_info)} 个有效位点
- **监测时段**: 2015年8月 – 2020年11月
- **累计相机工作日**: {int(total_camera_days):,} d
- **野生兽类独立事件**: {int(mammal_events):,} 次
- **野生鸟类独立事件**: {int(bird_events):,} 次
- **总野生动物独立事件**: {int(mammal_events + bird_events):,} 次

## 2. 物种组成

| 类群 | 目数 | 科数 | 种数 |
|------|------|------|------|
| 兽类 | 5 | 11 | {mammal_count} |
| 鸟类 | 9 | 16 | {bird_count} |
| **合计** | 14 | 27 | **{len(species)}** |

### 保护等级

- 国家一级保护动物: {prot_i} 种
- 国家二级保护动物: {prot_ii} 种

### 相对多度指数 (RAI) 前5名

| 排名 | 物种 | RAI | 出现位点 |
|------|------|-----|---------|
"""

for _, row in top5.iterrows():
    occ = species[species['species'] == row['species']]['site_occupancy_pct'].values
    occ_str = f"{occ[0]}%" if len(occ) > 0 else "-"
    report += f"| {int(row['rank'])} | {row['species']} | {row['rai']:.3f} | {occ_str} |\n"

report += f"""
## 3. 物种多样性空间格局

- 平均 Shannon-Wiener 指数: {mean_shannon:.2f} (范围: {site_div['shannon'].min():.2f}–{site_div['shannon'].max():.2f})
- 平均 Simpson 指数: {site_div['simpson'].mean():.3f} (范围: {site_div['simpson'].min():.3f}–{site_div['simpson'].max():.3f})

### 图表

! [fig1_species_composition.png](../outputs/figures/fig1_species_composition.png)

*图1: 物种组成分析。左: 保护等级分布; 中: RAI前15名; 右: 物种多度分布曲线*

! [fig2_diversity_vs_elevation.png](../outputs/figures/fig2_diversity_vs_elevation.png)

*图2: 多样性-海拔关系与生境类型比较*

## 4. 物种空间关联分析

- 显著关联物种对: {sig_pairs} 对 (p < 0.05)
- 其中正关联: {len(phi[phi['phi_coefficient'] > 0])} 对
- 其中负关联: {len(phi[phi['phi_coefficient'] < 0])} 对

! [fig3_cooccurrence_heatmap.png](../outputs/figures/fig3_cooccurrence_heatmap.png)

*图3: 物种共现热图*

! [fig4_cooccurrence_network.png](../outputs/figures/fig4_cooccurrence_network.png)

*图4: 物种共现网络图 (绿色=正关联, 红色=负关联)*

## 5. 年际变化趋势 (2015-2020)

| 年份 | 物种数 | 检测事件数 | Shannon指数 | Simpson指数 | 相机日 |
|------|--------|-----------|------------|------------|-------|
"""

for _, row in ann_div.iterrows():
    report += f"| {int(row['year'])} | {int(row['total_species'])} | {int(row['total_events']):,} | {row['shannon']:.2f} | {row['simpson']:.3f} | {int(row['total_camera_days']):,} |\n"

report += """
! [fig5_annual_trend.png](../outputs/figures/fig5_annual_trend.png)

*图5: 物种多样性的年际变化趋势*

! [fig6_key_species_trend.png](../outputs/figures/fig6_key_species_trend.png)

*图6: 6种关键物种RAI年际趋势*

## 6. 5年物种空间分布变化

! [fig7_bubble_maps_2015_2020.png](../outputs/figures/fig7_bubble_maps_2015_2020.png)

*图7: 2015-2020年物种检测气泡地图*

! [fig8_diversity_hotspot_migration.png](../outputs/figures/fig8_diversity_hotspot_migration.png)

*图8: 多样性热点时空迁移*

! [fig9_species_range_dynamics.png](../outputs/figures/fig9_species_range_dynamics.png)

*图9: 关键物种分布范围动态 (绿色=稳定, 红色=新出现, 蓝色=消失)*

! [fig10_annual_comparison.gif](../outputs/figures/fig10_annual_comparison.gif)

*图10: 2015-2020年物种检测分布年度对比动画*

## 7. 主要发现

1. **狍 (Capreolus pygargus) 是张广才岭最优势物种** (RAI=5.74, 100%位点出现), 其RAI在2015-2020年间呈明显上升趋势
2. **亚洲狗獾 (Meles leucurus) 和紫貂 (Martes zibellina) 为广布种**, 分别出现在83%和79%的位点
3. **原麝 (Moschus moschiferus) 和獐 (Hydropotes inermis) 为极稀有物种**, 分别仅出现在1个位点
4. 物种多样性的年际波动表明张广才岭野生动物群落处于动态变化中
5. 中海拔区域呈现较高的物种多样性, 不同生境类型的群落组成存在差异
"""

with open(PROJECT_ROOT / 'outputs' / 'reports' / 'analysis_report.md', 'w', encoding='utf-8') as f:
    f.write(report)

print("Report saved: outputs/reports/analysis_report.md")
