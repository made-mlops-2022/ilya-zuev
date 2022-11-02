import unittest
import pickle
from data.make_dataset import read_csv, split_train_test_data
from entities.split_params import SplittingParams


class TestMakeDataset(unittest.TestCase):
    PATH_TO_TEST_SAMPLE = "tests/data/data/sample01.csv"
    PATH_TO_TEST_SAMPLE_STATS = "tests/data/data/sample01_stats.pkl"
    PATH_TO_TEST_SAMPLE_SPLIT_STATS = "tests/data/data/sample01_split_stats.pkl"

    def test_read_csv(self):
        data = read_csv(self.PATH_TO_TEST_SAMPLE)
        with open(self.PATH_TO_TEST_SAMPLE_STATS, "rb") as f:
            stats = pickle.load(f)

        self.assertIsInstance(data, stats['type'])
        self.assertEqual(data.shape, stats['shape'])
        self.assertEqual(list(data.columns), list(stats['columns']))
        self.assertTrue(data.describe().equals(stats['describe']))

    def test_split_dataset(self):
        data = read_csv(self.PATH_TO_TEST_SAMPLE)
        with open(self.PATH_TO_TEST_SAMPLE_SPLIT_STATS, "rb") as f:
            stats = pickle.load(f)

        test_size = 0.3
        random_state = 20

        train, test = split_train_test_data(data, SplittingParams(test_size, random_state))
        self.assertEqual(test_size, stats['test_size'])
        self.assertEqual(random_state, stats['random_state'])
        self.assertTrue(train.equals(stats['train']))
        self.assertTrue(test.equals(stats['test']))

        test_size = 0.4
        random_state = 20

        train, test = split_train_test_data(data, SplittingParams(test_size, random_state))
        self.assertFalse(train.equals(stats['train']))
        self.assertFalse(test.equals(stats['test']))

        test_size = 0.3
        random_state = 22

        train, test = split_train_test_data(data, SplittingParams(test_size, random_state))
        self.assertFalse(train.equals(stats['train']))
        self.assertFalse(test.equals(stats['test']))
