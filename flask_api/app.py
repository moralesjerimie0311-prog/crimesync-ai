"""
CrimeSync AI — Flask API
========================
Endpoints:
  POST /predict           → DBSCAN hotspot detection
  POST /predict/barangay  → Random Forest barangay risk
  POST /predict/crimes    → Crime type prediction per barangay
  POST /predict/forecast  → Prophet/ARIMA time-series forecast
  GET  /health            → Health check
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import joblib
import os
import logging
import traceback

# ── Model components
from sklearn.cluster import DBSCAN

# ── Prophet (graceful fallback to ARIMA if not installed)
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    from statsmodels.tsa.arima.model import ARIMA

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# ══════════════════════════════════════════════════════════════
#  AUTO-DETECT MODEL DIRECTORY
# ══════════════════════════════════════════════════════════════

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

POSSIBLE_PATHS = [
    os.path.join(BASE_DIR, "models"),
    os.path.join(BASE_DIR, "flask_api", "models"),
    os.path.join(BASE_DIR, "..", "models"),
    os.path.join(BASE_DIR, "..", "flask_api", "models"),
]

MODEL_DIR = None
for path in POSSIBLE_PATHS:
    if os.path.exists(path):
        MODEL_DIR = os.path.abspath(path)
        break

if MODEL_DIR is None:
    MODEL_DIR = os.path.join(BASE_DIR, "models")  # fallback

print(f"[MODEL_DIR RESOLVED] {MODEL_DIR}")

# ── Global model state
rf_model = None
encoders = None
scaler   = None


# ══════════════════════════════════════════════════════════════
#  MODEL LOADING
# ══════════════════════════════════════════════════════════════

def load_models():
    global rf_model, encoders, scaler

    logger.info("========== LOADING ML MODELS ==========")

    rf_path     = os.path.join(MODEL_DIR, "rf_risk_model.joblib")
    enc_path    = os.path.join(MODEL_DIR, "label_encoders.joblib")
    scaler_path = os.path.join(MODEL_DIR, "feature_scaler.joblib")

    logger.info(f"[RF MODEL PATH] {rf_path}")
    logger.info(f"[ENCODERS PATH] {enc_path}")
    logger.info(f"[SCALER PATH]   {scaler_path}")

    # ── RF MODEL
    try:
        if not os.path.exists(rf_path):
            raise FileNotFoundError(f"RF model missing: {rf_path}")
        rf_model = joblib.load(rf_path)
        logger.info("✅ RF model loaded")
    except Exception as e:
        rf_model = None
        logger.error(f"❌ RF load failed: {e}")
        logger.error(traceback.format_exc())

    # ── ENCODERS (strict — no silent fallback)
    try:
        if not os.path.exists(enc_path):
            raise FileNotFoundError(f"Encoders missing: {enc_path}")
        encoders = joblib.load(enc_path)
        if not isinstance(encoders, dict) or len(encoders) == 0:
            raise ValueError("Encoders file is invalid or empty")
        logger.info("✅ Encoders loaded")
    except Exception as e:
        encoders = None
        logger.error(f"❌ Encoders load failed: {e}")
        logger.error(traceback.format_exc())

    # ── SCALER
    try:
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler missing: {scaler_path}")
        scaler = joblib.load(scaler_path)
        logger.info("✅ Scaler loaded")
    except Exception as e:
        scaler = None
        logger.error(f"❌ Scaler load failed: {e}")
        logger.error(traceback.format_exc())

    logger.info("========== MODEL LOADING COMPLETE ==========")


load_models()


# ══════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════

def encode_safe(encoder, value, feature_name="unknown"):
    """Strict encoder — raises clearly if encoders or a specific encoder is missing."""
    if encoders is None:
        raise RuntimeError("Encoders not loaded. Prediction blocked.")
    if encoder is None:
        raise RuntimeError(f"Missing encoder for feature: {feature_name}")
    try:
        return int(encoder.transform([str(value)])[0])
    except Exception:
        return 0  # safe fallback for unseen labels only


def require_ml():
    """Return a 500 error response if ML models are not loaded, else None."""
    if rf_model is None:
        return jsonify({"error": "Random Forest model not loaded"}), 500
    if encoders is None:
        return jsonify({"error": "Encoders not loaded"}), 500
    return None


def risk_level(score: float) -> str:
    if score >= 70:
        return "High Risk"
    elif score >= 40:
        return "Medium Risk"
    return "Low Risk"


def build_features(barangay: str, incident_type: str, month: int,
                   day_of_week: int, total_incidents: int,
                   days_since_last: int) -> np.ndarray:
    """Build the 6-feature vector used by the Random Forest."""
    bgy_encoder  = encoders.get("barangay")
    type_encoder = encoders.get("crime_type")

    bgy_enc  = encode_safe(bgy_encoder,  barangay,       "barangay")
    type_enc = encode_safe(type_encoder, incident_type,  "crime_type")

    features = np.array([[
        bgy_enc,
        type_enc,
        month,
        day_of_week,
        total_incidents,
        days_since_last
    ]], dtype=float)

    if scaler is not None:
        features = scaler.transform(features)

    return features


# ══════════════════════════════════════════════════════════════
#  ENDPOINT 1 — DBSCAN HOTSPOT DETECTION
#  Replaces old rule-based getPredictedHotspots()
#  Called by PHP's existing getPredictedHotspots() curl call
# ══════════════════════════════════════════════════════════════

@app.route("/predict", methods=["POST"])
def predict_hotspots():
    """
    Input:  { "incidents": [{"lat": float, "lng": float, "timestamp": int}, ...] }
    Output: [{"lat": float, "lng": float, "cluster": int,
              "cluster_size": int, "risk_score": float, "is_hotspot": bool}, ...]
    """
    data      = request.get_json(force=True)
    incidents = data.get("incidents", [])

    if len(incidents) < 2:
        return jsonify([])

    coords     = np.array([[p["lat"], p["lng"]] for p in incidents])
    coords_rad = np.radians(coords)

    # DBSCAN: eps=0.0003 rad ≈ 33 metres; min_samples=3
    db = DBSCAN(
        eps=0.0003,
        min_samples=3,
        algorithm="ball_tree",
        metric="haversine"
    ).fit(coords_rad)

    labels     = db.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    logger.info(f"DBSCAN found {n_clusters} clusters from {len(incidents)} points")

    results = []
    for cluster_id in set(labels):
        if cluster_id == -1:
            continue  # noise

        mask         = labels == cluster_id
        cluster_pts  = coords[mask]
        cluster_size = int(mask.sum())
        centroid_lat = float(cluster_pts[:, 0].mean())
        centroid_lng = float(cluster_pts[:, 1].mean())

        density_score = min(100, cluster_size * 8)
        risk_score    = round(density_score, 1)

        results.append({
            "lat":          centroid_lat,
            "lng":          centroid_lng,
            "cluster":      int(cluster_id),
            "cluster_size": cluster_size,
            "risk_score":   risk_score,
            "is_hotspot":   cluster_size >= 5
        })

    results.sort(key=lambda x: x["risk_score"], reverse=True)
    return jsonify(results)


# ══════════════════════════════════════════════════════════════
#  ENDPOINT 2 — RANDOM FOREST BARANGAY RISK PREDICTION
#  Called by PHP to replace rule-based barangay risk scoring
# ══════════════════════════════════════════════════════════════

@app.route("/predict/barangay", methods=["POST"])
def predict_barangay():
    """
    Input:
    {
      "barangays": [
        {
          "barangay": "Tuguis",
          "lat": 10.29136,
          "lng": 122.90365,
          "incident_types": [{"type": "Theft", "count": 5}, ...],
          "total_incidents": 12,
          "last_incident_days_ago": 3,
          "month": 6,
          "day_of_week": 2
        }, ...
      ]
    }
    Output:
    [
      {
        "barangay": "Tuguis",
        "lat": 10.29136,
        "lng": 122.90365,
        "risk_score": 78.4,
        "confidence": 84.2,
        "prediction": "High Risk",
        "dominant_crime": "Theft",
        "probability": 82.3
      }, ...
    ]
    """
    data      = request.get_json(force=True)
    barangays = data.get("barangays", [])
    results   = []

    for b in barangays:
        bgy_name       = b.get("barangay", "Unknown")
        total          = int(b.get("total_incidents", 0))
        days_ago       = int(b.get("last_incident_days_ago", 999))
        month          = int(b.get("month", 1))
        dow            = int(b.get("day_of_week", 0))
        incident_types = b.get("incident_types", [])

        # Dominant crime type
        dominant_type  = "Unknown"
        dominant_count = 0
        if incident_types:
            top            = max(incident_types, key=lambda x: x.get("count", 0))
            dominant_type  = top.get("type", "Unknown")
            dominant_count = int(top.get("count", 0))

        dominant_prob = round((dominant_count / max(1, total)) * 100, 1) if total > 0 else 0

        if rf_model is not None and encoders:
            try:
                features   = build_features(bgy_name, dominant_type, month, dow, total, days_ago)
                proba      = rf_model.predict_proba(features)[0]
                classes    = rf_model.classes_
                proba_dict = dict(zip(classes, proba))

                high_p     = proba_dict.get(2, 0)   # class 2 = high
                med_p      = proba_dict.get(1, 0)   # class 1 = medium
                low_p      = proba_dict.get(0, 0)   # class 0 = low

                risk_score = round(high_p * 100 * 0.6 + med_p * 100 * 0.3 + low_p * 100 * 0.1, 1)
                confidence = round(float(max(proba)) * 100, 1)
                pred_class = int(rf_model.predict(features)[0])
                pred_label = risk_level(risk_score)
                logger.info(f"RISK_SCORE USED: {risk_score}, LABEL: {pred_label}")

            except Exception as e:
                logger.warning(f"RF prediction failed for {bgy_name}: {e}")
                risk_score, confidence, pred_label = _fallback_risk(total, days_ago)
        else:
            risk_score, confidence, pred_label = _fallback_risk(total, days_ago)

        results.append({
            "barangay":       bgy_name,
            "lat":            b.get("lat", 0),
            "lng":            b.get("lng", 0),
            "risk_score":     risk_score,
            "confidence":     confidence,
            "prediction":     pred_label,
            "dominant_crime": dominant_type,
            "probability":    dominant_prob
        })

    return jsonify(results)


def _fallback_risk(total: int, days_ago: int):
    """Simple fallback risk calculation when RF model is unavailable."""
    time_f     = max(0.5, 1 - (days_ago / 365))
    raw_score  = min(100, total * 8 * time_f)
    confidence = min(85, 40 + total * 2)
    return round(raw_score, 1), round(confidence, 1), risk_level(raw_score)


# ══════════════════════════════════════════════════════════════
#  ENDPOINT 3 — CRIME TYPE PREDICTION PER BARANGAY
#  Returns the most likely future crime type per barangay
# ══════════════════════════════════════════════════════════════

@app.route("/predict/crimes", methods=["POST"])
def predict_crime_types():
    """
    Input:
    {
      "barangays": [
        {
          "barangay": "Anahaw",
          "crime_history": [
            {"type": "Theft", "count": 8, "last_days_ago": 5},
            {"type": "Physical Injury", "count": 3, "last_days_ago": 20}
          ]
        }, ...
      ]
    }
    Output:
    [
      {
        "barangay": "Anahaw",
        "predicted_crime": "Theft",
        "probability": 72.5,
        "all_predictions": [
          {"type": "Theft", "probability": 72.5},
          {"type": "Physical Injury", "probability": 27.5}
        ]
      }, ...
    ]
    """
    data      = request.get_json(force=True)
    barangays = data.get("barangays", [])
    results   = []

    for b in barangays:
        bgy_name      = b.get("barangay", "Unknown")
        crime_history = b.get("crime_history", [])

        if not crime_history:
            results.append({
                "barangay":        bgy_name,
                "predicted_crime": "Unknown",
                "probability":     0,
                "all_predictions": []
            })
            continue

        # Weight crimes by recency and frequency
        weighted = []
        for c in crime_history:
            freq     = int(c.get("count", 1))
            days_ago = int(c.get("last_days_ago", 30))
            recency  = max(0.3, 1 - (days_ago / 180))
            weighted.append({"type": c["type"], "weight": freq * recency})

        total_w = sum(w["weight"] for w in weighted)
        if total_w == 0:
            continue

        predictions = sorted(
            [{"type": w["type"], "probability": round((w["weight"] / total_w) * 100, 1)}
             for w in weighted],
            key=lambda x: x["probability"],
            reverse=True
        )

        results.append({
            "barangay":        bgy_name,
            "predicted_crime": predictions[0]["type"],
            "probability":     predictions[0]["probability"],
            "all_predictions": predictions[:5]
        })

    return jsonify(results)


# ══════════════════════════════════════════════════════════════
#  ENDPOINT 4 — TIME-SERIES FORECASTING (Prophet / ARIMA)
#  Replaces rule-based moving-average forecast in prediction.php
# ══════════════════════════════════════════════════════════════

@app.route("/predict/forecast", methods=["POST"])
def predict_forecast():
    """
    Input:
    {
      "barangays": [
        {
          "barangay": "Tuguis",
          "time_series": [
            {"date": "2022-01-15", "count": 2},
            {"date": "2022-02-03", "count": 1}, ...
          ]
        }, ...
      ],
      "horizons": [7, 30, 90, 365]
    }
    Output:
    [
      {
        "barangay": "Tuguis",
        "forecast_7d":   {"count": 3,   "lower": 1,   "upper": 5},
        "forecast_30d":  {"count": 12,  "lower": 8,   "upper": 16},
        "forecast_90d":  {"count": 38,  "lower": 25,  "upper": 52},
        "forecast_365d": {"count": 145, "lower": 110, "upper": 180},
        "trend":         "rising",
        "confidence":    78.4,
        "model_used":    "prophet"
      }, ...
    ]
    """
    data      = request.get_json(force=True)
    barangays = data.get("barangays", [])
    horizons  = data.get("horizons", [7, 30, 90, 365])
    results   = []

    for b in barangays:
        bgy_name    = b.get("barangay", "Unknown")
        time_series = b.get("time_series", [])

        if len(time_series) < 5:
            avg   = sum(t.get("count", 0) for t in time_series) / max(1, len(time_series))
            entry = {"barangay": bgy_name, "trend": "stable",
                     "confidence": 35.0, "model_used": "average"}
            for h in horizons:
                est        = round(avg * (h / 30), 0)
                entry[f"forecast_{h}d"] = {
                    "count": int(est),
                    "lower": int(est * 0.6),
                    "upper": int(est * 1.4)
                }
            results.append(entry)
            continue

        df          = pd.DataFrame(time_series)
        df["date"]  = pd.to_datetime(df["date"])
        df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0)
        df          = df.groupby("date")["count"].sum().reset_index()
        df          = df.sort_values("date")

        entry = {"barangay": bgy_name}

        try:
            if PROPHET_AVAILABLE and len(df) >= 10:
                entry.update(_prophet_forecast(df, horizons))
                entry["model_used"] = "prophet"
            else:
                entry.update(_arima_forecast(df, horizons))
                entry["model_used"] = "arima"
        except Exception as e:
            logger.warning(f"Forecast failed for {bgy_name}: {e}")
            entry.update(_simple_forecast(df, horizons))
            entry["model_used"] = "fallback"

        results.append(entry)

    return jsonify(results)


def _prophet_forecast(df: pd.DataFrame, horizons: list) -> dict:
    """Run Prophet and return forecast dict for each horizon."""
    prophet_df = df.rename(columns={"date": "ds", "count": "y"})

    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.15,
        seasonality_prior_scale=10
    )
    m.fit(prophet_df)

    max_h  = max(horizons)
    future = m.make_future_dataframe(periods=max_h, freq="D")
    fc     = m.predict(future)
    tail   = fc.tail(max_h)

    result = {}
    for h in horizons:
        sub   = tail.head(h)
        result[f"forecast_{h}d"] = {
            "count": max(0, int(sub["yhat"].sum())),
            "lower": max(0, int(sub["yhat_lower"].sum())),
            "upper": max(0, int(sub["yhat_upper"].sum()))
        }

    recent     = prophet_df["y"].tail(30).mean()
    prior      = prophet_df["y"].tail(60).head(30).mean()
    trend      = "rising" if recent > prior * 1.1 else ("declining" if recent < prior * 0.9 else "stable")
    confidence = round(min(92, 45 + len(prophet_df) * 0.5), 1)

    result["trend"]      = trend
    result["confidence"] = confidence
    return result


def _arima_forecast(df: pd.DataFrame, horizons: list) -> dict:
    """Run ARIMA and return forecast dict for each horizon."""
    series  = df.set_index("date")["count"].asfreq("D").fillna(0)
    fitted  = ARIMA(series, order=(2, 1, 2)).fit()
    max_h   = max(horizons)
    fc_obj  = fitted.get_forecast(steps=max_h)
    fc_mean = fc_obj.predicted_mean
    fc_ci   = fc_obj.conf_int()

    result = {}
    for h in horizons:
        result[f"forecast_{h}d"] = {
            "count": max(0, int(fc_mean.head(h).sum())),
            "lower": max(0, int(fc_ci.iloc[:h, 0].sum())),
            "upper": max(0, int(fc_ci.iloc[:h, 1].sum()))
        }

    recent     = series.tail(30).mean()
    prior      = series.tail(60).head(30).mean()
    trend      = "rising" if recent > prior * 1.1 else ("declining" if recent < prior * 0.9 else "stable")
    confidence = round(min(88, 40 + len(series) * 0.4), 1)

    result["trend"]      = trend
    result["confidence"] = confidence
    return result


def _simple_forecast(df: pd.DataFrame, horizons: list) -> dict:
    """Simple average-based fallback forecast."""
    avg    = df["count"].values.mean()
    result = {}
    for h in horizons:
        est = max(0, round(avg * (h / 30)))
        result[f"forecast_{h}d"] = {
            "count": int(est),
            "lower": int(est * 0.5),
            "upper": int(est * 1.5)
        }
    result["trend"]      = "stable"
    result["confidence"] = 30.0
    return result


# ══════════════════════════════════════════════════════════════
#  HEALTH CHECK
# ══════════════════════════════════════════════════════════════

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":            "ok",
        "rf_model_loaded":   rf_model is not None,
        "encoders_loaded":   encoders is not None and isinstance(encoders, dict) and len(encoders) > 0,
        "scaler_loaded":     scaler is not None,
        "prophet_available": PROPHET_AVAILABLE,
        "model_dir":         MODEL_DIR
    })


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)