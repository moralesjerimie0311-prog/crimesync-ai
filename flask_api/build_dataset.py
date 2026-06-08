import requests
import pandas as pd

# 1. Load data from your API
url = "http://localhost/incident_system/api/incidents.php"

response = requests.get(url)
data = response.json()

df = pd.DataFrame(data)

# 2. Keep ONLY useful ML columns
df = df[[
    "incident_type",
    "barangay",
    "latitude",
    "longitude",
    "severity",
    "incident_date"
]]

# 3. Clean data
df = df.dropna()

# 4. Fix text inconsistencies
df["incident_type"] = df["incident_type"].str.lower().str.strip()
df["barangay"] = df["barangay"].str.lower().str.strip()

# 5. Save clean dataset
df.to_csv("clean_incidents.csv", index=False)

print("✅ Dataset created successfully!")
print(df.shape)