from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# Initialize FastAPI app
app = FastAPI()

# Load the saved model
model = joblib.load("../linear_regression_model.pkl")

# Define the input data model
class InputData(BaseModel):
    avg_session_length: float
    time_on_app: float
    time_on_website: float
    length_of_membership: float

# Define the root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Linear Regression Prediction API!"}

# Define the prediction endpoint
@app.post("/predict")
def predict(data: InputData):
    try:
        # Prepare the input data for prediction
        input_features = np.array([
            data.avg_session_length,
            data.time_on_app,
            data.time_on_website,
            data.length_of_membership
        ]).reshape(1, -1)

        # Make the prediction
        prediction = model.predict(input_features)

        return {"prediction": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")