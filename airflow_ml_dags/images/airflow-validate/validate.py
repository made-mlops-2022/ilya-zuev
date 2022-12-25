import os
import pandas as pd
import click
import pickle
import json
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


@click.command("validate")
@click.option("--data-dir")
@click.option("--model-dir")
@click.option("--output-dir")
def validate(data_dir: str, model_dir: str, output_dir):
    valid = pd.read_csv(os.path.join(data_dir, "valid.csv"))

    with open(os.path.join(model_dir, "model.pkl"), "rb") as f:
        model = pickle.load(f)

    train_columns = sorted(set(valid.columns) - set(["target", "ID"]))

    preds = model.predict(valid[train_columns])
    mae = mean_absolute_error(valid["target"], preds)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "scores.txt"), "w") as f:
        json.dump({"mae": mae}, f)


if __name__ == '__main__':
    validate()
