import os
import pickle
import click
import pandas as pd
import numpy as np
from faker import Faker
from sklearn.model_selection import train_test_split
from tests.samples.generated.test_params import TestParams, read_test_params


def gen_sample_for_test(generator: Faker, test_params: TestParams, name: str) -> pd.DataFrame:
    rows = [
        {
            "age": generator.random.randint(0, 100),
            "sex": generator.random.randint(0, 1),
            "ca": generator.random.randint(0, 4),
            "chol": generator.random.randrange(0, 500),
            "cat": generator.random.choice(["a", "b", "c"]),
            "condition": generator.random.randint(0, 1)
        } for _ in range(
            np.random.randint(test_params.sample_rows_low, test_params.sample_rows_high)
        )
    ]

    data = pd.DataFrame(rows)

    path = os.path.join(
        test_params.output_samples_folder,
        name
    )

    data.to_csv(path, index=False)

    return data


def calc_stats_for_data_read_csv_answ(data: pd.DataFrame):
    stats = {}
    stats["type"] = type(data)
    stats["shape"] = data.shape
    stats["columns"] = data.columns
    stats["describe"] = data.describe()
    return stats


def gen_answ_for_data_read_csv(data: pd.DataFrame, test_params: TestParams, name: str):
    TEST_NAME = "data_read_csv"

    stats = calc_stats_for_data_read_csv_answ(data)

    path = os.path.join(
        test_params.output_answers_folder,
        TEST_NAME
    )

    if not os.path.exists(path):
        os.makedirs(path)

    path = os.path.join(
        path,
        name
    )

    with open(path, "wb") as f:
        pickle.dump(stats, f)


def calc_stats_for_data_split_dataset_answ(
    train: pd.DataFrame,
    test: pd.DataFrame,
    test_size: float,
    random_sate: int
):
    stats = {}
    stats["test_size"] = test_size
    stats["random_state"] = random_sate
    stats["train"] = train
    stats["test"] = test
    return stats


def gen_answ_for_data_split_dataset(
    data: pd.DataFrame,
    test_params:
    TestParams,
    name: str
):
    TEST_NAME = "data_split_dataset"
    TEST_SIZE = np.random.uniform(0.1, 0.9)
    RANDOM_STATE = test_params.random_state

    train, test = train_test_split(data, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    stats = calc_stats_for_data_split_dataset_answ(train, test, TEST_SIZE, RANDOM_STATE)

    path = os.path.join(
        test_params.output_answers_folder,
        TEST_NAME
    )

    if not os.path.exists(path):
        os.makedirs(path)

    path = os.path.join(
        path,
        name
    )

    with open(path, "wb") as f:
        pickle.dump(stats, f)


def create_folders(test_params: TestParams):
    if not os.path.exists(test_params.output_samples_folder):
        os.makedirs(test_params.output_samples_folder)

    if not os.path.exists(test_params.output_answers_folder):
        os.makedirs(test_params.output_answers_folder)


def gen_samples_for_tests(test_params: TestParams):
    np.random.seed(test_params.random_state)
    Faker.seed(test_params.random_state)
    generator = Faker()

    create_folders(test_params)

    for i in range(test_params.sample_count):
        name = f"{test_params.sample_base_name}{i:02d}"
        data = gen_sample_for_test(generator, test_params, f"{name}.csv")
        gen_answ_for_data_read_csv(data, test_params, f"{name}.pkl")
        gen_answ_for_data_split_dataset(data, test_params, f"{name}.pkl")


@click.command(name="train_pipeline")
@click.argument("config_path")
def gen_samples_for_tests_command(config_path: str):
    params = read_test_params(config_path)
    gen_samples_for_tests(params)


if __name__ == "__main__":
    gen_samples_for_tests_command()
