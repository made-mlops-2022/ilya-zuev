import pandas as pd


def read_csv(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    return data
