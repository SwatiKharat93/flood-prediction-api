
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)

model = joblib.load("flood_prediction_model.pkl")
features = joblib.load("model_features.pkl")


@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    new_data = pd.DataFrame([{
        "Latitude": data["latitude"],
        "Longitude": data["longitude"],
        "Rainfall (mm)": data["rainfall"],
        "Temperature (°C)": data["temperature"],
        "Humidity (%)": data["humidity"],
        "River Discharge (m³/s)": data["river_discharge"],
        "Water Level (m)": data["water_level"],
        "Elevation (m)": data["elevation"],
        "Land Cover": data["land_cover"],
        "Soil Type": data["soil_type"],
        "Population Density": data["population_density"],
        "Infrastructure": data["infrastructure"],
        "Historical Floods": data["historical_floods"]
    }])

    new_data = pd.get_dummies(
        new_data,
        columns=["Land Cover", "Soil Type"],
        drop_first=True
    )

    new_data = new_data.reindex(
        columns=features,
        fill_value=0
    )

    prediction = model.predict(new_data)[0]
    probability = model.predict_proba(new_data)[0][1]

    if probability >= 0.70:
        risk = "High"
    elif probability >= 0.40:
        risk = "Moderate"
    else:
        risk = "Low"

    return jsonify({
        "prediction": int(prediction),
        "flood_probability": round(float(probability), 3),
        "risk": risk
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
