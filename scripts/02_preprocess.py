from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "processed_data"

detection_raw = pd.read_csv(PROCESSED_DIR / "detection_raw.csv")
site_info = pd.read_csv(PROCESSED_DIR / "site_info.csv")

id_vars = ["camera_id", "year", "working_days"]
species_cols = [c for c in detection_raw.columns if c not in id_vars]

detection_long = detection_raw.melt(
    id_vars=id_vars,
    value_vars=species_cols,
    var_name="species",
    value_name="events",
)

detection_long = detection_long[detection_long["events"] > 0].copy()

coord_cols = ["camera_id", "longitude", "latitude"]
detection_long = detection_long.merge(
    site_info[coord_cols], on="camera_id", how="left"
)

detection_long.to_csv(PROCESSED_DIR / "detection_long.csv", index=False)
print(f"detection_long.csv: {len(detection_long)} rows")

detection_matrix = detection_long.pivot_table(
    index="camera_id",
    columns="species",
    values="events",
    aggfunc="count",
    fill_value=0,
)
detection_matrix = (detection_matrix > 0).astype(int)
detection_matrix.to_csv(PROCESSED_DIR / "detection_matrix.csv")
print(
    f"detection_matrix.csv: {detection_matrix.shape[0]} sites x {detection_matrix.shape[1]} species"
)

print("All preprocessing complete.")
