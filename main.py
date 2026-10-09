# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import warnings

# Suppress sklearn feature name warnings mentioned in your notebook logs
warnings.filterwarnings('ignore')

# Initialize FastAPI app
app = FastAPI(title="Solar Power Prediction API")

# Configure CORS so the front-end can communicate with this back-end
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace with your UI's specific domain
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model and scaler
try:
    model = joblib.load('solar_model.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    print("Error: Ensure 'solar_model.pkl' and 'scaler.pkl' are in the same directory.")

# Define the expected data structure for the API request
class SolarFeatures(BaseModel):
    ambient_temperature: float
    module_temperature: float
    irradiation: float

@app.post("/predict")
def predict_power(features: SolarFeatures):
    # Format input data identically to the training format
    input_data = np.array([[
        features.ambient_temperature, 
        features.module_temperature, 
        features.irradiation
    ]])
    
    # Scale and predict
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    
    # Prevent negative power outputs
    final_prediction = max(0.0, float(prediction[0]))
    
    return {"predicted_power": round(final_prediction, 2)}