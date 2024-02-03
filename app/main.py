import fastapi as FastAPI
from pydantic import BaseModel
import recommend

app = FastAPI()

class TextIn(BaseModel):
    text: str
    
class Prediction(BaseModel):
    out: str
    
