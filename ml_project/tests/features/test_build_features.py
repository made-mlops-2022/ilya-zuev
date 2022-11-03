import unittest
import logging
import pandas as pd
import numpy as np
from data.make_dataset import read_csv
from features.build_features import (
    make_features,
    build_transformer,
    extract_target
)
from entities.feature_params import FeatureParams


class TestBuildFeatures(unittest.TestCase):
    PATH_TO_TEST_SAMPLE = "ml_project/tests/samples/sample01.csv"

    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def make_features_for_test(self, values, params):
        data = pd.DataFrame.from_dict(values)
        transformer = build_transformer(params)
        transformer.fit(data)
        return make_features(transformer, data)

    def test_categorical_features(self):
        data = read_csv(self.PATH_TO_TEST_SAMPLE)
        params = FeatureParams(
            categorical_features=["sex", "cp", "fbs"],
            numerical_features=[],
            target_col=""
        )
        transformer = build_transformer(params)
        transformer.fit(data)
        features = make_features(transformer, data)
        self.assertEqual(features.shape, (102, 7))

        params = FeatureParams(["col"], [], "")
        features = self.make_features_for_test(
            {"col": np.zeros(100)},
            params
        )
        self.assertEqual(features.shape, (100, 1))

        features = self.make_features_for_test(
            {"col": ["a", "b", "c", "d"]},
            params
        )
        self.assertEqual(features.shape, (4, 4))

        features = self.make_features_for_test(
            {"col": [0, 1, 2, 2]},
            params
        )
        self.assertEqual(features.shape, (4, 3))

        features = self.make_features_for_test(
            {"col": [0, 1, np.nan]},
            params
        )
        self.assertEqual(features.shape, (3, 2))

        features = self.make_features_for_test(
            {"col1": [0, 1, np.nan], "col2": ["a", "b", "c"]},
            FeatureParams(["col1", "col2"], [], "")
        )
        self.assertEqual(features.shape, (3, 5))

    def test_numerical_features(self):
        data = read_csv(self.PATH_TO_TEST_SAMPLE)
        params = FeatureParams(
            categorical_features=[],
            numerical_features=["sex", "cp", "fbs"],
            target_col=""
        )
        transformer = build_transformer(params)
        transformer.fit(data)
        features = make_features(transformer, data)
        self.assertEqual(features.shape, (102, 3))

        params = FeatureParams([], ["col"], "")
        features = self.make_features_for_test(
            {"col": np.zeros(100)},
            params
        )
        self.assertEqual(features.shape, (100, 1))

        features = self.make_features_for_test(
            {"col": [0, 1, 4, 3]},
            params
        )
        self.assertEqual(features.shape, (4, 1))
        self.assertEqual(features[0].mean(), 2)

        features = self.make_features_for_test(
            {"col": [0, 1, np.nan]},
            params
        )
        self.assertEqual(features.shape, (3, 1))
        self.assertEqual(features[0].mean(), 0.5)

    def test_extract_target(self):
        TARGET_COLUMN = "condition"
        data = read_csv(self.PATH_TO_TEST_SAMPLE)
        params = FeatureParams(
            categorical_features=[],
            numerical_features=[],
            target_col=TARGET_COLUMN
        )
        target_sr = extract_target(data, params)
        self.assertTrue(target_sr.equals(data[TARGET_COLUMN]))
