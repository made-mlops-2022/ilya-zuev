import os
import pandas as pd
import click
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


@click.command("train_model")
@click.option("--input-dir")
@click.option("--output-dir")
def train_model(input_dir: str, output_dir):
    TARGET = "target"
    train = pd.read_csv(os.path.join(input_dir, "train.csv"))

    model = LinearRegression()
    train_columns = sorted(set(train.columns) - set([TARGET, "ID"]))
    model.fit(train[train_columns], train[TARGET])

    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "model.pkl"), "wb") as f:
        pickle.dump(model, f)


if __name__ == '__main__':
    train_model()
