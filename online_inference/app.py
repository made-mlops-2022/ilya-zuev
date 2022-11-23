import pandas as pd
import pickle
from scipy.sparse import issparse
from fastapi import FastAPI

app = FastAPI()

def predict():
    data = pd.read_csv("/Users/administrator/zuev/prog/made/1_mlops/ilya-zuev/online_inference/data/heart_cleveland_upload.csv")

    with open("/Users/administrator/zuev/prog/made/1_mlops/ilya-zuev/online_inference/model/transformer.pkl", "rb") as f:
        transformer = pickle.load(f)

    data = transformer.transform(data)
    if issparse(data):
        data = data.toarray()
    data = pd.DataFrame(data)

    with open("/Users/administrator/zuev/prog/made/1_mlops/ilya-zuev/online_inference/model/model.pkl", "rb") as f:
        model = pickle.load(f)

    predicts = model.predict(data)

    return predicts


@app.get("/predict")
async def root():
    return predict()
