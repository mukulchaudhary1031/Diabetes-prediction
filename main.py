from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import numpy as np
import pickle

app = FastAPI()

# Templates folder mount
templates = Jinja2Templates(directory="templates")

# Static CSS folder mount
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load trained pipeline model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Show HTML Form
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Handle Form POST request
@app.post("/predict", response_class=HTMLResponse)
def predict(request: Request,
            Pregnancies: int = Form(...),
            Glucose: float = Form(...),
            BloodPressure: float = Form(...),
            SkinThickness: float = Form(...),
            Insulin: float = Form(...),
            BMI: float = Form(...),
            DiabetesPedigreeFunction: float = Form(...),
            Age: int = Form(...),
           ):

    # Convert user input to numpy array
    user_data = np.array([[Pregnancies, Glucose, BloodPressure,
                           SkinThickness, Insulin, BMI,
                           DiabetesPedigreeFunction, Age]])

    # Prediction
    prediction = model.predict(user_data)[0]

    result = "Diabetic" if prediction == 1 else "Not Diabetic"

    # Return HTML with result
    return templates.TemplateResponse("index.html",
                                      {"request": request, "result": result})