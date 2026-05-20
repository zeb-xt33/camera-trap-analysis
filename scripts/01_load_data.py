from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA = PROJECT_ROOT / "data" / "raw_data" / "ZGCL_Camera-trapping Dataset 2015-2020.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "processed_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SITE_COLUMNS = {
    "相机编号": "camera_id",
    "经度": "longitude",
    "纬度": "latitude",
    "海拔": "elevation",
    "相机开始日期": "start_date",
    "相机结束日期": "end_date",
    "累计工作日": "working_days",
    "独立事件数(兽）": "independent_events_mammal",
    "独立事件数(鸟）": "independent_events_bird",
    "拍摄野生动物物种数": "species_richness",
    "生境类型": "habitat_type",
    "植被类型": "vegetation_type",
}

SPECIES_COLUMNS = {
    "物种": "species",
    "国家重点保护野生动物名录": "national_protection",
    "IUCN 红色名录": "iucn",
    "中国生物多样性红色名录": "china_red_list",
    "拍摄位点数及比例(%)": "site_occupancy_pct",
    "独立事件数": "independent_events",
    "相对多度指数": "rai",
}

DETECTION_COLUMNS = {
    "相机编号 Camera code": "camera_id",
    "年份 Year": "year",
    "累计工作日 Accumulated camera days (d)": "working_days",
}

sheet1 = pd.read_excel(RAW_DATA, sheet_name=0, skiprows=[1])
sheet1.rename(columns=SITE_COLUMNS, inplace=True)
sheet1.to_csv(OUTPUT_DIR / "site_info.csv", index=False)
print(f"site_info.csv: {len(sheet1)} rows, columns={list(sheet1.columns)}")

sheet3_raw = pd.read_excel(RAW_DATA, sheet_name=2, skiprows=[1])
sheet3_raw.rename(columns=SPECIES_COLUMNS, inplace=True)
species_list = sheet3_raw.dropna(subset=["rai"]).copy()
species_list["species"] = species_list["species"].str.replace(
    r"^\d+\.\s*", "", regex=True
)
species_list.to_csv(OUTPUT_DIR / "species_list.csv", index=False)
print(f"species_list.csv: {len(species_list)} rows, columns={list(species_list.columns)}")

sheet2_raw = pd.read_excel(RAW_DATA, sheet_name=1)
sheet2_raw.rename(columns=DETECTION_COLUMNS, inplace=True)
sheet2_raw = sheet2_raw[sheet2_raw["camera_id"] != "Grand Total"].copy()
sheet2_raw.to_csv(OUTPUT_DIR / "detection_raw.csv", index=False)
print(f"detection_raw.csv: {len(sheet2_raw)} rows, columns={list(sheet2_raw.columns)}")

print("All files saved to processed_data/")
