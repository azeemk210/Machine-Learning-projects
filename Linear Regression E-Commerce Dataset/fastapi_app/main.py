from fastapi import FastAPI, HTTPException, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import numpy as np

# Initialize FastAPI app
app = FastAPI()

# Load the saved model
model = joblib.load("../linear_regression_model.pkl")

# Serve templates
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Define the input data model
class InputData(BaseModel):
    avg_session_length: float
    time_on_app: float
    time_on_website: float
    length_of_membership: float

# Render the input form
@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Handle form submission
@app.post("/predict")
def predict_form(
    avg_session_length: float = Form(...),
    time_on_app: float = Form(...),
    time_on_website: float = Form(...),
    length_of_membership: float = Form(...),
):
    try:
        # Prepare the input data for prediction
        input_features = np.array([
            avg_session_length,
            time_on_app,
            time_on_website,
            length_of_membership
        ]).reshape(1, -1)

        # Make the prediction
        prediction = model.predict(input_features)

        return {"prediction": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")