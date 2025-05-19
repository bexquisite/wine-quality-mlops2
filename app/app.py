from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("../model/wine_quality_model.pkl")

# Initialize Flask app
app = Flask(__name__)

# Define prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    # Get JSON data from request
    data = request.get_json()

    # Convert data to DataFrame (to match training format)
    df = pd.DataFrame([data])

    # Make prediction
    prediction = model.predict(df)[0]

    # Return prediction as JSON
    return jsonify({"predicted_quality": float(prediction)})

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
