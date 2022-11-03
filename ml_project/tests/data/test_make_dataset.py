import unittest
import pickle
import logging
import os
from tests.samples.generated.test_params import read_test_params
from entities.split_params import SplittingParams
from data.make_dataset import read_csv, split_train_test_data


class TestMakeDataset(unittest.TestCase):
    PATH_TO_SAMPLES = "ml_project/tests/samples"
    PATH_TO_TEST_SAMPLE = f"{PATH_TO_SAMPLES}/sample01.csv"
    PATH_TO_TEST_SAMPLE_STATS = f"{PATH_TO_SAMPLES}/sample01_stats.pkl"
    PATH_TO_TEST_SAMPLE_SPLIT_STATS = f"{PATH_TO_SAMPLES}/sample01_split_stats.pkl"
    PATH_TO_FAKE_SAMPLES = "ml_project/tests/samples/generated/sampled"
    PATH_TO_FAKE_ANSWERS = "ml_project/tests/samples/generated/sampled/answers"
    PATH_TO_TEST_CONFIG = "ml_project/tests/samples/generated/test_config.yaml"

    def setUp(self):
        logging.disable(logging.CRITICAL)
        self.test_params = read_test_params(self.PATH_TO_TEST_CONFIG)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def test_read_csv(self):
        def check_base_asserts(path_to_data, path_to_answer):
            data = read_csv(path_to_data)
            with open(path_to_answer, "rb") as f:
                answer = pickle.load(f)

            self.assertIsInstance(data, answer["type"])
            self.assertEqual(data.shape, answer["shape"])
            self.assertEqual(list(data.columns), list(answer["columns"]))
            self.assertTrue(data.describe().equals(answer["describe"]))

        check_base_asserts(self.PATH_TO_TEST_SAMPLE, self.PATH_TO_TEST_SAMPLE_STATS)

        for i in range(self.test_params.sample_count):
            sample_path = os.path.join(
                self.test_params.output_samples_folder,
                f"{self.test_params.sample_base_name}{i:02d}.csv"
            )
            answ_path = os.path.join(
                self.test_params.output_answers_folder,
                "data_read_csv",
                f"{self.test_params.sample_base_name}{i:02d}.pkl"
            )

            check_base_asserts(sample_path, answ_path)

    def test_split_dataset(self):
        def check_splits(data, answer, test_size, random_state):
            train, test = split_train_test_data(data, SplittingParams(test_size, random_state))
            self.assertTrue(train.equals(answer["train"]))
            self.assertTrue(test.equals(answer["test"]))

        def check_base_asserts(path_to_data, path_to_answer):
            data = read_csv(path_to_data)
            with open(path_to_answer, "rb") as f:
                answer = pickle.load(f)
            check_splits(data, answer, answer["test_size"], answer["random_state"])

        data = read_csv(self.PATH_TO_TEST_SAMPLE)
        with open(self.PATH_TO_TEST_SAMPLE_SPLIT_STATS, "rb") as f:
            stats = pickle.load(f)

        test_size = 0.3
        random_state = 20

        self.assertEqual(test_size, stats["test_size"])
        self.assertEqual(random_state, stats["random_state"])
        check_splits(data, stats, test_size, random_state)

        test_size = 0.4
        random_state = 20

        train, test = split_train_test_data(data, SplittingParams(test_size, random_state))
        self.assertFalse(train.equals(stats["train"]))
        self.assertFalse(test.equals(stats["test"]))

        test_size = 0.3
        random_state = 22

        train, test = split_train_test_data(data, SplittingParams(test_size, random_state))
        self.assertFalse(train.equals(stats["train"]))
        self.assertFalse(test.equals(stats["test"]))

        for i in range(self.test_params.sample_count):
            sample_path = os.path.join(
                self.test_params.output_samples_folder,
                f"{self.test_params.sample_base_name}{i:02d}.csv"
            )
            answ_path = os.path.join(
                self.test_params.output_answers_folder,
                "data_split_dataset",
                f"{self.test_params.sample_base_name}{i:02d}.pkl"
            )

            check_base_asserts(sample_path, answ_path)
