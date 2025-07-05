import requests
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

API_URL = "https://data.boston.gov/api/3/action/datastore_search_sql"

resource_ids = {
    2018: "2be28d90-3a90-4af1-a3f6-f28c1e25880a",
    2019: "ea2e4696-4a2d-429c-9807-d02eb92e0222",
    2020: "6ff6a6fd-3141-4440-a880-6f60a37fe789",
    2021: "f53ebccd-bc61-49f9-83db-625f209c95f5",
    2022: "81a7b022-f8fc-4da5-80e4-b160058ca207",
    2023: "e6013a93-1321-4f2a-bf91-8d8a02f1e62f",
    2024: "dff4d804-5031-443a-8409-8344efd0e5c8",
    2025: "9d7c2214-4709-478a-a2e8-fb2020a5bb94",
}

def needle_pickup_sql(resource_id):
    return f"""
        SELECT open_dt
        FROM "{resource_id}"
        WHERE case_title = 'Needle Pickup'
        LIMIT 100000
    """

def fetch_open_dates(resource_id):
    sql = needle_pickup_sql(resource_id)
    response = requests.get(API_URL, params={"sql": sql})
    df = pd.DataFrame(response.json()["result"]["records"])
    df["open_dt"] = pd.to_datetime(df["open_dt"], errors="coerce")
    return df[["open_dt"]]

# Fetch and combine all years
all_dataframes = [fetch_open_dates(rid) for rid in resource_ids.values()]
all_calls = pd.concat(all_dataframes, ignore_index=True)

# Group by day
needle_daily = all_calls.groupby(all_calls["open_dt"].dt.date).size()
needle_daily = needle_daily.sort_index()
needle_daily.index = pd.to_datetime(needle_daily.index)

# Plot raw data
plt.figure(figsize=(12, 6))
plt.plot(needle_daily.index, needle_daily.values, label="311 Needle Pickup", linewidth=2)
plt.title("Daily 311 Needle Pickup Calls (2018–2025)")
plt.xlabel("Date")
plt.ylabel("Number of Reports")
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# LOESS smoothing
date_nums = needle_daily.index.map(pd.Timestamp.toordinal)
smoothed = lowess(needle_daily.values, date_nums, frac=0.04, return_sorted=False)

# Plot LOESS-smoothed curve
plt.figure(figsize=(12, 6))
plt.plot(needle_daily.index, smoothed, color="orange", linewidth=2, label="LOESS Smoothed")
plt.title("LOESS Smoothing of Daily 311 Needle Pickup Calls")
plt.xlabel("Date")
plt.ylabel("Smoothed Reports")
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
