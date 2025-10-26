from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model and scaler
model = joblib.load("diabetes_model.pkl")  # Load trained XGBoost model
scaler = joblib.load("scaler.pkl")  # Load pre-trained scaler

# Define input schema
class DiabetesInput(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float

@app.post("/predict")
async def predict(data: DiabetesInput):
    # Convert input data into a structured NumPy array with correct feature names
    input_df = np.array([[data.Pregnancies, data.Glucose, data.BloodPressure, 
                        data.SkinThickness, data.Insulin, data.BMI, 
                        data.DiabetesPedigreeFunction, data.Age]])

    # Convert to Pandas DataFrame to preserve feature names
    import pandas as pd
    feature_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
                    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
    input_df = pd.DataFrame(input_df, columns=feature_names)

    # Scale input
    input_scaled = scaler.transform(input_df)  # Fixes warning


    # Make prediction
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]  # Get probability of diabetes

    # Convert NumPy types to Python native types
    prediction = int(prediction)
    probability = float(probability)

    return {
        "prediction": "likely to have diabetes" if prediction == 1 else "unlikely to have diabetes",
        "probability": round(probability, 2)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
