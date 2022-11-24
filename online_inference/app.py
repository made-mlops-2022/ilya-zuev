import pandas as pd
import pickle
from typing import List
from scipy.sparse import issparse
from fastapi import FastAPI
from pydantic import BaseModel
import os


PATH_TO_MODEL = "./online_inference/model/model.pkl"
PATH_TO_TRANSFORMER = "./online_inference/model/transformer.pkl"

app = FastAPI()


class Data(BaseModel):
    data: List[List[str]]


def predict(data: List[List[str]]):
    data = pd.DataFrame(data)
    new_header = data.iloc[0]
    data = data[1:]
    data.columns = new_header
    data = data.astype("float64")

    with open(PATH_TO_TRANSFORMER, "rb") as f:
        transformer = pickle.load(f)

    data = transformer.transform(data)
    if issparse(data):
        data = data.toarray()
    data = pd.DataFrame(data)

    with open(PATH_TO_MODEL, "rb") as f:
        model = pickle.load(f)

    predicts = model.predict(data)

    return predicts


@app.post("/predict")
async def make_predict(data: Data):
    predicts = predict(data.data).tolist()
    return predicts


@app.get("/health")
async def is_model_ready():
    with open("./log.txt", "w") as f:
        f.write(f"{PATH_TO_MODEL}\n")
        f.write(f"{PATH_TO_TRANSFORMER}\n")
        f.write(f"{os.getcwd()}\n")
    if not os.path.isfile(PATH_TO_MODEL) or not os.path.isfile(PATH_TO_TRANSFORMER):
        raise

    return "model is ready"
