import unittest
import logging
import os
import json
from io import StringIO
from train_pipeline import train_pipeline
from entities.train_pipeline_params import read_training_pipeline_params
from tests.samples.generated.test_params import read_test_params


class TestTrainPipeline(unittest.TestCase):
    PATH_TO_TEST_CONFIG = "ml_project/tests/samples/generated/test_config.yaml"

    def setUp(self):
        logging.disable(logging.CRITICAL)
        self.test_params = read_test_params(self.PATH_TO_TEST_CONFIG)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def test_train_pipeline(self):
        def check_pipeline(params):
            train_pipeline(params)

            self.assertTrue(os.path.isfile(params.output_model_path))
            self.assertTrue(os.path.isfile(params.feature_transformer_path))
            self.assertTrue(os.path.isfile(params.metric_path))

            with open(params.metric_path, "r", encoding="utf-8") as metric_file:
                metrics = json.load(metric_file)

            self.assertTrue(0 <= metrics["accuracy_score"] <= 1)
            self.assertTrue(0 <= metrics["precision_score"] <= 1)
            self.assertTrue(0 <= metrics["recall_score"] <= 1)
            self.assertTrue(0 <= metrics["roc_auc_score"] <= 1)


        PATH_TO_CONFIG = "ml_project/tests/configs/train_config.yaml"
        params = read_training_pipeline_params(PATH_TO_CONFIG)
        check_pipeline(params)

        PATH_TO_CONFIG = "ml_project/tests/samples/generated/pipelines/configs/train_config.yaml"
        params = read_training_pipeline_params(PATH_TO_CONFIG)
        for i in range(self.test_params.sample_count):
            sample_path = os.path.join(
                self.test_params.output_samples_folder,
                f"{self.test_params.sample_base_name}{i:02d}.csv"
            )
            params.input_data_path = sample_path

            check_pipeline(params)
