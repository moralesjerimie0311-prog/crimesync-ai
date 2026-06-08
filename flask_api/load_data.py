import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_json("data.json")

# =========================
# 2. CLEAN DATA
# =========================
df = df[[
    "incident_type",
    "barangay",
    "latitude",
    "longitude",
    "severity"
]].dropna()

# =========================
# 3. ENCODE CATEGORICAL DATA
# =========================
le_barangay = LabelEncoder()
df["barangay_encoded"] = le_barangay.fit_transform(df["barangay"])

le_type = LabelEncoder()
df["incident_type_encoded"] = le_type.fit_transform(df["incident_type"])

# =========================
# 4. FEATURES (X) & TARGET (y)
# =========================
X = df[["barangay_encoded", "latitude", "longitude", "severity"]]
y = df["incident_type_encoded"]

# =========================
# 5. SPLIT DATA
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)

# =========================
# 6. TRAIN MODEL
# =========================
model = RandomForestClassifier()
model.fit(X_train, y_train)

# =========================
# 7. PREDICTION
# =========================
y_pred = model.predict(X_test)

# =========================
# 8. EVALUATION
# =========================
print("Accuracy:", accuracy_score(y_test, y_pred))