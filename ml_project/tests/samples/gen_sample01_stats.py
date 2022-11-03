import pickle
import pandas as pd


def gen_stats():
    FILE_NAME = "sample01"
    data = pd.read_csv(f"{FILE_NAME}.csv")

    stats = dict()
    stats["type"] = type(data)
    stats["shape"] = data.shape
    stats["columns"] = data.columns
    stats["describe"] = data.describe()

    with open(f"{FILE_NAME}_stats.pkl", "wb") as f:
        pickle.dump(stats, f)


if __name__ == "__main__":
    gen_stats()
