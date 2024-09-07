import pickle
import numpy as np
import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI

app = FastAPI()


class PredicRequest(BaseModel):
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


# The /predict POST endpoint
@app.post("/predict")
def predict(data: list[float]):
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    data = np.array(data).reshape(1, -1)
    prediction = model.predict(data)
    return prediction


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
