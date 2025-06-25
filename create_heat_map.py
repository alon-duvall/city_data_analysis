import pandas as pd
import sqlite3
import folium
from folium.plugins import HeatMap

# Step 1: Load data from SQLite
conn = sqlite3.connect('my_database.db')
df = pd.read_sql("SELECT Location FROM my_table WHERE Location IS NOT NULL", conn)
conn.close()

# Step 2: Extract latitude and longitude from 'Location' column
df[['latitude', 'longitude']] = df['Location'].str.extract(r'\(?\s*([-\d\.]+)\s*,\s*([-\d\.]+)\s*\)?').astype(float)

# Step 3: Drop rows with missing or invalid coordinates
df = df.dropna(subset=['latitude', 'longitude'])

# Step 4: Optionally sample if too large
df_sample = df.sample(n=10000) if len(df) > 10000 else df

# Step 5: Create heatmap
center = [df_sample['latitude'].mean(), df_sample['longitude'].mean()]
m = folium.Map(location=center, zoom_start=12)
heat_data = df_sample[['latitude', 'longitude']].values.tolist()
HeatMap(heat_data).add_to(m)

# Step 6: Save to HTML
m.save("crime_heatmap.html")
print("Heatmap saved as crime_heatmap.html")
