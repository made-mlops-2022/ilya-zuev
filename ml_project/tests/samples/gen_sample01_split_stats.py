import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from tests.samples.generated.gen_samples_for_tests import (
    calc_stats_for_data_split_dataset_answ
)


def gen_stats():
    FILE_NAME = "sample01"
    data = pd.read_csv(f"{FILE_NAME}.csv")

    TEST_SIZE = 0.3
    RANDOM_STATE = 20
    train, test = train_test_split(data, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    stats = calc_stats_for_data_split_dataset_answ(train, test, TEST_SIZE, RANDOM_STATE)

    with open(f"{FILE_NAME}_split_stats.pkl", "wb") as f:
        pickle.dump(stats, f)


if __name__ == "__main__":
    gen_stats()
