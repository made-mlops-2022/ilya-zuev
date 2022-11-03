import unittest
import logging
import os
import json
from io import StringIO
from train_pipeline import train_pipeline
from entities.train_pipeline_params import read_training_pipeline_params


class TestTrainPipeline(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def test_train_pipeline(self):
        PATH_TO_CONFIG = "ml_project/tests/configs/train_config.yaml"
        params = read_training_pipeline_params(PATH_TO_CONFIG)
        train_pipeline(params)

        self.assertTrue(os.path.isfile(params.output_model_path))
        self.assertTrue(os.path.isfile(params.feature_transformer_path))
        self.assertTrue(os.path.isfile(params.metric_path))

        with open(params.metric_path, "r", encoding="utf-8") as metric_file:
            metrics = json.load(metric_file)

        self.assertTrue(0 < metrics["accuracy_score"] <= 1)
        self.assertTrue(0 < metrics["precision_score"] <= 1)
        self.assertTrue(0 < metrics["recall_score"] <= 1)
        self.assertTrue(0 < metrics["roc_auc_score"] <= 1)
