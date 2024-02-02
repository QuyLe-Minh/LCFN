from fastapi import FastAPI
from pydantic import BaseModel
from recommend import recommend
import numpy as np

app = FastAPI()

class TextIn(BaseModel):
    text:int
    
class PredictionOut(BaseModel):
    recommendations:list
    
@app.get("/")
def home():
    return {"health_check": "OK"}

@app.post("/predict", response_model=PredictionOut)
def recommendation(payload: TextIn):
    rec = recommend(np.array([payload.text]))[0]
    return {"recommendations": rec}

