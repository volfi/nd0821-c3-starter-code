from typing import Union, List
from fastapi import FastAPI, HTTPException
import pickle
from pydantic import BaseModel, Field
import numpy as np
import json

class Data(BaseModel):
    X: List[float]

app = FastAPI(
    title="Inference API",
    description="An API that returns predictions.",
    version="1.0.0",
)

pickled_model = pickle.load(open('/Users/schwabw/Desktop/code/udacity_mlops/nd0821-c3-starter-code/starter/starter/model/model.pkl', 'rb'))

@app.get("/")
async def say_hello():
    return "Hello User!"


@app.post("/inference/")
async def model_inference(data: Data):
    X = np.array((data.X))
    y_pred = pickled_model.predict(X.reshape(1, -1))
    return str(y_pred[0])