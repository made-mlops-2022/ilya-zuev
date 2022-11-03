import unittest
import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from ml_project.entities.train_params import TrainingParams
from ml_project.models.model_fit_predict import (
    train_model,
    predict_model,
    predict_proba_model,
    evaluate_model
)
from ml_project.data.make_dataset import read_csv


class TestBuildFeatures(unittest.TestCase):
    PATH_TO_TEST_SAMPLE = "ml_project/tests/samples/sample01.csv"

    def setUp(self):
        logging.disable(logging.CRITICAL)

        self.data = read_csv(self.PATH_TO_TEST_SAMPLE)
        self.TARGET_COLUMN = "condition"
        self.TRAIN_COLUMNS = list(self.data.columns)
        self.TRAIN_COLUMNS.remove(self.TARGET_COLUMN)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def test_train_model(self):
        RANDOM_STATE = 42
        params = TrainingParams(
            model_type="LogisticRegression",
            random_state=RANDOM_STATE
        )
        model = train_model(self.data[self.TRAIN_COLUMNS], self.data[self.TARGET_COLUMN], params)
        self.assertIsInstance(model, LogisticRegression)
        self.assertEqual(model.get_params()["random_state"], RANDOM_STATE)

        with self.assertRaises(NotImplementedError) as _:
            params = TrainingParams(
                model_type="",
                random_state=RANDOM_STATE
            )
            model = train_model(
                self.data[self.TRAIN_COLUMNS],
                self.data[self.TARGET_COLUMN],
                params
            )

    def test_predicts(self):
        params = TrainingParams(
            model_type="LogisticRegression",
            random_state=42
        )
        model = train_model(self.data[self.TRAIN_COLUMNS], self.data[self.TARGET_COLUMN], params)
        predicts = predict_model(model, self.data[self.TRAIN_COLUMNS])
        self.assertIsInstance(predicts, np.ndarray)
        self.assertEqual(predicts.shape, (self.data.shape[0],))
        self.assertEqual(len(np.unique(predicts)), 2)

        predict_probas = predict_proba_model(model, self.data[self.TRAIN_COLUMNS])
        self.assertIsInstance(predict_probas, np.ndarray)
        self.assertEqual(predict_probas.shape, (self.data.shape[0], 2))
        self.assertGreater(len(np.unique(predict_probas)), 10)

    def test_evaluate_model(self):
        params = TrainingParams(
            model_type="LogisticRegression",
            random_state=42
        )
        X = pd.DataFrame({"col": [10] * 10 + [100] * 10})
        y = pd.Series([0] * 10 + [1] * 10)
        model = train_model(X, y, params)
        predicts = predict_model(model, X)
        predict_probas = predict_proba_model(model, X)
        metrics = evaluate_model(predicts, predict_probas, y)
        expected_metrics = {
            "accuracy_score": 1,
            "precision_score": 1,
            "recall_score": 1,
            "roc_auc_score": 1
        }

        y = pd.Series([1] * 10 + [0] * 10)
        expected_metrics = {
            "accuracy_score": 0,
            "precision_score": 0,
            "recall_score": 0,
            "roc_auc_score": 0
        }
        metrics = evaluate_model(predicts, predict_probas, y)
        self.assertEqual(metrics, expected_metrics)
