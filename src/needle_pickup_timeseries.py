import requests
import pandas as pd
import matplotlib.pyplot as plt

API_URL = "https://data.boston.gov/api/3/action/datastore_search_sql"

# Resource IDs for 311 Service Requests (2023–2025)
calls_2023_id = "e6013a93-1321-4f2a-bf91-8d8a02f1e62f"
calls_2024_id = "dff4d804-5031-443a-8409-8344efd0e5c8"
calls_2025_id = "9d7c2214-4709-478a-a2e8-fb2020a5bb94"

# SQL builder for each year
def needle_pickup_sql(resource_id):
    return f"""
        SELECT "open_dt"
        FROM "{resource_id}"
        WHERE "case_title" = 'Needle Pickup'
        LIMIT 100000
    """

# Fetch and convert to datetime
def fetch_open_dates(sql):
    response = requests.get(API_URL, params={"sql": sql})
    df = pd.DataFrame(response.json()["result"]["records"])
    df["open_dt"] = pd.to_datetime(df["open_dt"], errors="coerce")
    return df

# Fetch all three years
calls_2023 = fetch_open_dates(needle_pickup_sql(calls_2023_id))
calls_2024 = fetch_open_dates(needle_pickup_sql(calls_2024_id))
calls_2025 = fetch_open_dates(needle_pickup_sql(calls_2025_id))

# Combine and group by day
all_calls = pd.concat([calls_2023, calls_2024, calls_2025], ignore_index=True)
needle_daily = all_calls.groupby(all_calls["open_dt"].dt.date).size()

# Plot
plt.figure(figsize=(12, 6))
plt.plot(needle_daily.index, needle_daily.values, label="311 Needle Pickup", linewidth=2)
plt.title("Daily 311 Needle Pickup Calls (2023–2025)")
plt.xlabel("Date")
plt.ylabel("Number of Reports")
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()


# Save the figure
plt.savefig("needle_pickup_timeseries.png", dpi=300)  # or .svg for vector format
plt.show()
