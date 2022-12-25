import os
import pandas as pd
import click
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression


@click.command("predict")
@click.option("--data-dir")
@click.option("--model-dir")
@click.option("--output-dir")
def predict(data_dir: str, model_dir: str, output_dir):
    data = pd.read_csv(os.path.join(data_dir, "data.csv"))

    with open(os.path.join(model_dir, "model.pkl"), "rb") as f:
        model = pickle.load(f)

    train_columns = sorted(set(data.columns) - set(["target", "ID"]))

    preds = model.predict(data[train_columns])

    os.makedirs(output_dir, exist_ok=True)

    np.savetxt(os.path.join(output_dir, "predicts.csv"), preds, delimiter=",")


if __name__ == '__main__':
    predict()
