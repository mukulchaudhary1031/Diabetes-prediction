from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle

app = FastAPI()

# Load pipeline model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

class InputData(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

@app.post("/predict")
def predict(data: InputData):
    user_data = np.array([[data.Pregnancies, data.Glucose, data.BloodPressure,
                           data.SkinThickness, data.Insulin, data.BMI,
                           data.DiabetesPedigreeFunction, data.Age]])
    
    prediction = model.predict(user_data)[0]

    result = "Diabetic" if prediction == 1 else "Not Diabetic"
    return {"prediction": result}