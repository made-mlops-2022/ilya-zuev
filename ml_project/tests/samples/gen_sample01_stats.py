import pickle
import pandas as pd
from tests.samples.generated.gen_samples_for_tests import (
    calc_stats_for_data_read_csv_answ
)


def gen_stats():
    FILE_NAME = "sample01"
    data = pd.read_csv(f"{FILE_NAME}.csv")

    stats = calc_stats_for_data_read_csv_answ(data)

    with open(f"{FILE_NAME}_stats.pkl", "wb") as f:
        pickle.dump(stats, f)


if __name__ == "__main__":
    gen_stats()
