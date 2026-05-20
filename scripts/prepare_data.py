import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import entropy

xlsx = Path(r"E:\trae-ai\camera-trap-analysis\data\raw_data\ZGCL_Camera-trapping Dataset 2015-2020.xlsx")
processed = Path(r"E:\trae-ai\camera-trap-analysis\processed_data")
processed.mkdir(parents=True, exist_ok=True)

site_info = pd.read_excel(xlsx, sheet_name=0, skiprows=1)
site_info = site_info.rename(columns={
    "Camera code": "site",
    "Longitude (E)": "longitude",
    "Latitude (N)": "latitude"
})
site_info = site_info[["site", "longitude", "latitude"]].copy()
site_info.to_csv(processed / "site_info.csv", index=False, encoding="utf-8-sig")
print(f"site_info.csv: {len(site_info)} sites")

df2 = pd.read_excel(xlsx, sheet_name=1)
df2 = df2[df2.iloc[:, 1].notna()].copy()
df2.iloc[:, 1] = df2.iloc[:, 1].astype(int)
col_map = {df2.columns[0]: "site", df2.columns[1]: "year", df2.columns[2]: "work_days"}
df2 = df2.rename(columns=col_map)
species_cols = list(df2.columns[3:])
id_vars = ["site", "year", "work_days"]
df_long = df2.melt(id_vars=id_vars, value_vars=species_cols, var_name="species", value_name="detections")
df_long["species"] = df_long["species"].str.strip()
df_long["species"] = df_long["species"].str.replace(r"\s+", " ", regex=True)
df_long = df_long[df_long["detections"] > 0].reset_index(drop=True)
df_long.to_csv(processed / "detection_long.csv", index=False, encoding="utf-8-sig")
print(f"detection_long.csv: {len(df_long)} rows")
print(f"Unique species: {df_long['species'].nunique()}")

def calc_shannon(group):
    counts = group["detections"].values
    total = counts.sum()
    if total == 0:
        return 0.0
    props = counts / total
    return entropy(props, base=np.e)

diversity = df_long.groupby(["site", "year"], group_keys=False).apply(calc_shannon).reset_index()
diversity.columns = ["site", "year", "shannon"]
diversity.to_csv(processed / "site_diversity.csv", index=False, encoding="utf-8-sig")
print(f"site_diversity.csv: {len(diversity)} rows")
print("All CSV files created successfully!")
