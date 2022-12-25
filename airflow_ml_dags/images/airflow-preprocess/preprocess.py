import os
import pandas as pd
import click
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


@click.command("preprocess")
@click.option("--input-dir")
@click.option("--output-dir")
def preprocess(input_dir: str, output_dir):
    data = pd.read_csv(os.path.join(input_dir, "data.csv"))
    target = pd.read_csv(os.path.join(input_dir, "target.csv"))

    train, valid = train_test_split(data, test_size=0.2, random_state=20)

    train_columns = sorted(set(train.columns) - set(["ID"]))

    ss = StandardScaler()
    ss.fit(train[train_columns])

    train[train_columns] = ss.transform(train[train_columns])
    valid[train_columns] = ss.transform(valid[train_columns])

    train = train.merge(target, on="ID", how="left")
    valid = valid.merge(target, on="ID", how="left")

    print(train.head())

    os.makedirs(output_dir, exist_ok=True)
    train.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    valid.to_csv(os.path.join(output_dir, "valid.csv"), index=False)


if __name__ == '__main__':
    preprocess()
