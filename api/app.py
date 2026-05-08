from fastapi import FastAPI
import joblib
import numpy as np

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

    values = np.array(
        list(data.values())
    ).reshape(1, -1)

    prediction = int(
        model.predict(values)[0]
    )

    confidence = float(
        model.predict_proba(values)[0].max()
    )

    return {
        'prediction': prediction,
        'confidence': confidence
    }