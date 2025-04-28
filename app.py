from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import Request 
from fastapi import UploadFile, File
import pandas as pd
import io
# Load model
model = joblib.load('gradient_boosting_model.pkl')

# Create app
app = FastAPI()

# Define input data structure
class InputData(BaseModel):
    Gender: int
    Married: int
    Dependents: float
    Education: int
    Self_Employed: int
    ApplicantIncome: int
    CoapplicantIncome:    float
    LoanAmount:   float
    Loan_Amount_Term:  float
    Credit_History: float
    Property_Area: int

@app.post('/predict')
def predict(data: InputData):
    # Convert input data to numpy array
    input_array = np.array([[data.Gender,
                             data.Married,
                             data.Dependents,
                             data.Education,
                             data.Self_Employed,
                             data.ApplicantIncome,
                             data.CoapplicantIncome,
                             data.LoanAmount,
                             data.Loan_Amount_Term,
                             data.Credit_History,
                             data.Property_Area]])  # adjust based on number of features
    # Predict
    prediction = model.predict(input_array)
    return {'prediction': int(prediction[0])}


@app.post('/predict_csv')
async def predict_csv(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode('utf-8')))

    predictions = []
    for _, row in df.iterrows():
        input_array = np.array([row.values])
        prediction = model.predict(input_array)[0]
        predictions.append(prediction)

    df['prediction'] = predictions
    return df.to_dict(orient='records')





templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

