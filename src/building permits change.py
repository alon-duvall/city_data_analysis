import requests
import pandas as pd
from scipy import stats
import numpy as np
import folium
import matplotlib.cm as cm
import matplotlib.colors as colors
from time import sleep

def get_year_total_count(year):
    """
    Fetches the total count of building permits for a specific year.
    """
    print(f"Fetching total count for {year}...")
    sql_query = (
        f'SELECT COUNT(*) as count from "6ddcd912-32a0-43df-9908-63574f8c7e77" '
        f'WHERE "issued_date" >= \'{year}-01-01T00:00:00\' '
        f'AND "issued_date" < \'{year + 1}-01-01T00:00:00\''
    )
    params = {"sql": sql_query}
    try:
        response = requests.get(
            "https://data.boston.gov/api/3/action/datastore_search_sql",
            params=params
        )
        response.raise_for_status()
        # The result is a list with one dictionary, e.g., [{'count': '35000'}]
        count = int(response.json()["result"]["records"][0]['count'])
        print(f"✅ Found {count} total records for {year}.")
        return count
    except Exception as e:
        print(f"❌ An error occurred while fetching count for {year}: {e}")
        return 0

def fetch_all_data_for_year(year, chunk_size=32000, max_records=50000):
    """
    Fetches all building permit data for a specific year by paging through results.
    Note: Some endpoints limit to 32,000 records per call.
    """
    print(f"Fetching all data for {year}...")

    all_records = []
    offset = 0

    while True:
        sql_query = (
            f'SELECT * FROM "6ddcd912-32a0-43df-9908-63574f8c7e77" '
            f'WHERE "issued_date" >= \'{year}-01-01T00:00:00\' '
            f'AND "issued_date" < \'{year + 1}-01-01T00:00:00\' '
            f'ORDER BY "_id" LIMIT {chunk_size} OFFSET {offset}'
        )
        params = {"sql": sql_query}

        try:
            response = requests.get(
                "https://data.boston.gov/api/3/action/datastore_search_sql",
                params=params
            )
            response.raise_for_status()
            records = response.json()["result"]["records"]

            if not records:
                break

            all_records.extend(records)
            print(f"Fetched {len(records)} records (offset {offset})")
            offset += chunk_size

            # Prevent hitting rate limits
            sleep(1)

            # Optional: stop if too much data (avoid bugs, runaway loop)
            if len(all_records) > max_records:
                print("⚠️ Max record limit reached; stopping early.")
                break

        except Exception as e:
            print(f"❌ Error at offset {offset}: {e}")
            break

    print(f"✅ Fetched total {len(all_records)} records for {year}.")
    return pd.DataFrame(all_records)

def clean_data(df):
    """Cleans and filters the permit DataFrame."""
    df['lat'] = pd.to_numeric(df['y_latitude'], errors='coerce')
    df['lon'] = pd.to_numeric(df['x_longitude'], errors='coerce')
    df.dropna(subset=['lat', 'lon'], inplace=True)
    df = df[(df['lat'] >= 42.2) & (df['lat'] <= 42.4) &
            (df['lon'] >= -71.2) & (df['lon'] <= -70.9)]
    return df

def create_density_difference_heatmap():
    """
    Fetches Boston permit data for 2023 & 2024, calculates the scaled difference
    in permit density, and generates an interactive heatmap.
    """
    # --- 1. Fetch Total Counts and Sample Data for Each Year ---
    total_2023 = get_year_total_count(2023)
    total_2024 = get_year_total_count(2024)
    
    sample_df_2023 = fetch_all_data_for_year(2023)
    sample_df_2024 = fetch_all_data_for_year(2024)

    if sample_df_2023.empty or sample_df_2024.empty or total_2023 == 0 or total_2024 == 0:
        print("❌ Insufficient data to perform comparison. Exiting.")
        return

    # --- 2. Clean Sampled Data ---
    print("Cleaning and preparing sampled data...")
    clean_sample_2023 = clean_data(sample_df_2023)
    clean_sample_2024 = clean_data(sample_df_2024)
        
    print(f"✅ Total valid permits: {total_2023} in 2023, {total_2024} in 2024.")
    print(f"✅ Using {len(clean_sample_2023)} and {len(clean_sample_2024)} valid samples.")

    # --- 3. Perform KDE on Samples ---
    print("Performing Gaussian KDE on samples... (This may take a moment)")
    kde_2023 = stats.gaussian_kde(np.vstack([clean_sample_2023['lon'], clean_sample_2023['lat']]))
    kde_2024 = stats.gaussian_kde(np.vstack([clean_sample_2024['lon'], clean_sample_2024['lat']]))
    print("✅ KDE models created.")

    # --- 4. Evaluate KDE on a Grid and Scale by Total Permits ---
    print("Evaluating and scaling density difference on a grid...")
    GRID_POINTS = 75
    combined_df = pd.concat([clean_sample_2023, clean_sample_2024])
    lon_grid_1d = np.linspace(combined_df['lon'].min(), combined_df['lon'].max(), GRID_POINTS)
    lat_grid_1d = np.linspace(combined_df['lat'].min(), combined_df['lat'].max(), GRID_POINTS)
    grid_lon, grid_lat = np.meshgrid(lon_grid_1d, lat_grid_1d)
    grid_positions = np.vstack([grid_lon.ravel(), grid_lat.ravel()])

    density_2023 = kde_2023(grid_positions)
    density_2024 = kde_2024(grid_positions)
    
    scaled_density_2023 = density_2023 * total_2023
    scaled_density_2024 = density_2024 * total_2024
    
    density_diff = scaled_density_2024 - scaled_density_2023
    diff_grid = density_diff.reshape(GRID_POINTS, GRID_POINTS)

    # --- 5. Create the Difference Heatmap ---
    print("Generating interactive heatmap...")
    m = folium.Map(location=[combined_df['lat'].mean(), combined_df['lon'].mean()], zoom_start=12)

    colormap = cm.get_cmap('RdBu_r')
    max_abs_diff = np.max(np.abs(diff_grid))
    norm = colors.Normalize(vmin=-max_abs_diff, vmax=max_abs_diff)

    lat_step = (lat_grid_1d[-1] - lat_grid_1d[0]) / (GRID_POINTS - 1)
    lon_step = (lon_grid_1d[-1] - lon_grid_1d[0]) / (GRID_POINTS - 1)

    for i in range(GRID_POINTS - 1):
        for j in range(GRID_POINTS - 1):
            cell_value = diff_grid[i, j]
            rgba_color = colormap(norm(cell_value))
            hex_color = colors.to_hex(rgba_color)
            fill_opacity = abs(cell_value) / max_abs_diff * 1.0 if max_abs_diff > 0 else 0
            
            # Correctly index the 2D grid arrays
            south = grid_lat[i, j]
            north = south + lat_step
            west = grid_lon[i, j]
            east = west + lon_step
            
            folium.Rectangle(
                bounds=[(south, west), (north, east)],
                color=None,
                fill=True,
                fill_color=hex_color,
                fill_opacity=fill_opacity,
                popup=f"Scaled Change: {cell_value:.2e}"
            ).add_to(m)

    map_filename = "boston_permit_density_change_map_all_data.html"
    m.save(map_filename)
    print(f"✅ Interactive heatmap saved to '{map_filename}'")

if __name__ == "__main__":
    create_density_difference_heatmap()
