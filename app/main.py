from fastapi import FastAPI
from pydantic import BaseModel
from app.recommend import recommend
import numpy as np
import pickle
from typing import List

with open('app/movie_dict.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)

# print(loaded_dict["1607"])
app = FastAPI()

class TextIn(BaseModel):
    text:int
    
class PredictionOut(BaseModel):
    recommendations: List[str]
    
@app.get("/")
def home():
    return {"health_check": "OK"}

@app.post("/predict", response_model=PredictionOut)
def recommendation(payload: TextIn):
    rec = recommend(np.array([payload.text]))
    print(rec)
    res = [loaded_dict.get(str(i), "")[0] if loaded_dict.get(str(i), "") else "" for i in rec]
    return {"recommendations": res}

