
import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel

# Load the trained model
artifact = joblib.load("customer_subscription_model.joblib")

model = artifact["model"]
threshold = artifact["threshold"]
feature_names = artifact["features"]

# Create API
app = FastAPI(
    title="Customer Subscription Prediction API",
    description="Predicts whether a bank customer is likely to subscribe to a term deposit.",
    version="1.0.0"
)

# Expected customer information
class Customer(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    default: str
    balance: int
    housing: str
    loan: str
    contact: str
    day: int
    month: str
    campaign: int
    pdays: int
    previous: int
    poutcome: str

@app.get("/")
def home():
    return {
        "message": "Customer Subscription Prediction API is running"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True
    }

@app.post("/predict")
def predict(customer: Customer):

    customer_data = pd.DataFrame([customer.dict()])

    # Feature engineering
    customer_data["previously_contacted"] = (
        customer_data["pdays"] != -1
    ).astype(int)

    customer_data["pdays"] = customer_data["pdays"].replace(-1, 0)

    # Match training feature order
    customer_data = customer_data[feature_names]

    # Predict probability
    probability = model.predict_proba(customer_data)[0, 1]

    # Apply threshold
    prediction = (
        "yes" if probability >= threshold else "no"
    )

    return {
        "subscription_probability": round(float(probability), 4),
        "prediction": prediction,
        "threshold": float(threshold)
    }
