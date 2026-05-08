from fastapi import FastAPI
import joblib
import numpy as np
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO
)

# Create FastAPI app
app = FastAPI()

# Load trained model
model = joblib.load('models/model.pkl')


# Home route
@app.get('/')
def home():

    return {
        'message': 'Heart Disease Prediction API'
    }


# Prediction route
@app.post('/predict')
def predict(data: dict):

    # Convert input to numpy array
    values = np.array(
        list(data.values())
    ).reshape(1, -1)

    # Make prediction
    prediction = int(
        model.predict(values)[0]
    )

    # Prediction confidence
    confidence = float(
        model.predict_proba(values)[0].max()
    )

    # Log prediction details
    logging.info(
        f'Prediction: {prediction}, Confidence: {confidence}'
    )

    # Return response
    return {
        'prediction': prediction,
        'confidence': confidence
    }