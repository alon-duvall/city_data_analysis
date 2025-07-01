import os
import pandas as pd
import sqlite3
from scipy.stats import entropy
import folium
from folium.plugins import HeatMap

# -------------------------------
# Step 1: Load CSV and Create SQLite Database
# -------------------------------
#Gets the 2023-2025 Boston Crime data file
csv_path = os.path.expanduser("~/Downloads/tmpqlb4rmud.csv")
df_csv = pd.read_csv(csv_path)

conn = sqlite3.connect("crime_data.db")
table_name = "crime_reports"
df_csv.to_sql(table_name, conn, if_exists="replace", index=False)
print(f"Database '{table_name}' created with {len(df_csv)} rows.")

# -------------------------------
# Step 2: Compute Location Entropies
# -------------------------------

query = """
    SELECT OFFENSE_DESCRIPTION, HOUR, Lat, Long
    FROM crime_reports
    WHERE HOUR IS NOT NULL 
      AND Lat IS NOT NULL 
      AND Long IS NOT NULL 
      AND OFFENSE_DESCRIPTION IS NOT NULL
"""
df = pd.read_sql(query, conn)

# Filter to offenses with at least 100 total occurrences
valid_offenses = df["OFFENSE_DESCRIPTION"].value_counts()
valid_offenses = valid_offenses[valid_offenses >= 100].index
df = df[df["OFFENSE_DESCRIPTION"].isin(valid_offenses)]

# Bin latitude and longitude into 20×20 grid
df["lat_bin"] = pd.cut(df["Lat"], bins=20, labels=False)
df["lon_bin"] = pd.cut(df["Long"], bins=20, labels=False)
df["location_bin"] = df["lat_bin"] * 20 + df["lon_bin"]

# Compute entropy for each (OFFENSE_DESCRIPTION, HOUR) group
results = []
grouped = df.groupby(["OFFENSE_DESCRIPTION", "HOUR"])

for (offense, hour), group in grouped:
    if len(group) >= 50:
        loc_counts = group["location_bin"].value_counts()
        probs = loc_counts / loc_counts.sum()
        ent = entropy(probs, base=2)
        results.append((offense, hour, ent, len(group)))

# Sort and display
results.sort(key=lambda x: -x[2])
print("\n--- Location Entropies by Offense and Hour (≥ 50 samples) ---")
for i, (offense, hour, ent, count) in enumerate(results, 1):
    print(f"{i}. {offense} @ Hour {hour:02d}: Entropy = {ent:.4f} — Count = {count}")

# -------------------------------
# Step 3: Plot Heatmap for Drug Offenses (Hours 7–9)
# -------------------------------

heatmap_query = """
    SELECT Lat, Long 
    FROM crime_reports 
    WHERE OFFENSE_DESCRIPTION = 'DRUGS - POSSESSION/ SALE/ MANUFACTURING/ USE' 
      AND HOUR IN (7, 8, 9)
      AND Lat IS NOT NULL 
      AND Long IS NOT NULL
"""
df_heatmap = pd.read_sql(heatmap_query, conn)
conn.close()

# Create folium heatmap
map_center = [df_heatmap["Lat"].mean(), df_heatmap["Long"].mean()]
m = folium.Map(location=map_center, zoom_start=13)
heat_data = df_heatmap[["Lat", "Long"]].values.tolist()
HeatMap(heat_data, radius=10).add_to(m)

# Save to HTML
map_filename = "drug_heatmap_hour07_09.html"
m.save(map_filename)
print(f"\nHeatmap saved as {map_filename}")
