from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_caching import Cache
import pandas as pd
import tensorflow as tf
import logging

app = Flask(__name__)
CORS(app)

#caching
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

# Load the model
model = tf.keras.models.load_model('./model_final.h5')
print("Model loaded successfully")

# Define the required feature columns
feature_columns = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'Toluene']

logging.basicConfig(level=logging.INFO)

# Input validation function
def validate_input(data):
    required_fields = feature_columns
    for field in required_fields:
        if field not in data or not isinstance(data[field], (int, float)):
            raise ValueError(f"Invalid or missing value for {field}")
        if data[field] < 0 or data[field] > 1000: 
            raise ValueError(f"{field} value out of bounds")
    return True

#caching
@cache.memoize(60)
@app.route("/predict", methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("Received data:", data)

        # Validate input data
        validate_input(data)
        
        # Prepare input data for the model
        user_input = pd.DataFrame(data, index=[0])  
        user_input = user_input[feature_columns]

        # Make prediction
        predictions = model.predict(user_input)
        print("Processed input for prediction:", user_input)
        
        # Return the prediction as JSON
        return jsonify({"predicted_aqi": predictions.flatten().tolist()})
    
    except ValueError as ve:
        # Handle validation errors
        logging.error(f"Validation Error: {ve}")
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        # Handle other errors
        logging.error(f"Error occurred: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
