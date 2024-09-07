import pickle
import numpy as np
import uvicorn
import pandas as pd
from pydantic import BaseModel
from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "Welcome to the machine learning API"}


with open(
    r"D:\IT_projects\Turing_Colledge\Modul3\Sprint2\project\model.pkl", "rb"
) as f:
    model = pickle.load(f)


class InputData(BaseModel):
    gender: str
    age: int
    hypertension: int
    heart_disease: int
    ever_married: str
    work_type: str
    residence_type: str
    avg_glucose_level: float
    bmi: float
    smoking_status: str


@app.post("/predict/")
def predict(input_data: InputData):
    input_df = pd.DataFrame([input_data.dict()])

    prediction = model.predict(input_df)

    return {"prediction": int(prediction[0])}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
