import unittest
import logging
import os
import json
import pandas as pd
from train_pipeline import train_pipeline
from predict_pipeline import predict_pipeline
from entities.train_pipeline_params import read_training_pipeline_params
from entities.predict_pipeline_params import read_predict_pipeline_params
from tests.samples.generated.test_params import read_test_params
from data.make_dataset import read_csv


class TestPipeline(unittest.TestCase):
    PATH_TO_TEST_CONFIG = "ml_project/tests/samples/generated/test_config.yaml"

    def setUp(self):
        logging.disable(logging.CRITICAL)
        self.test_params = read_test_params(self.PATH_TO_TEST_CONFIG)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def check_pipeline(self, params):
        raise NotImplementedError

    def check_pipelines_on_genereted_datа(self, params):
        for i in range(self.test_params.sample_count):
            sample_path = os.path.join(
                self.test_params.output_samples_folder,
                f"{self.test_params.sample_base_name}{i:02d}.csv"
            )
            params.input_data_path = sample_path

            self.check_pipeline(params)


class TestTrainPipeline(TestPipeline):
    def check_pipeline(self, params):
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

    def test_train_pipeline(self):
        path_to_config = "ml_project/tests/configs/train_config.yaml"
        params = read_training_pipeline_params(path_to_config)
        self.check_pipeline(params)

        path_to_config = "ml_project/tests/samples/generated/pipelines/configs/train_config.yaml"
        params = read_training_pipeline_params(path_to_config)
        self.check_pipelines_on_genereted_datа(params)


class TestPredictPipeline(TestPipeline):
    def check_pipeline(self, params):
        predict_pipeline(params)

        self.assertTrue(os.path.isfile(params.predicts_path))
        self.assertTrue(os.path.isfile(params.predict_probas_path))

        predicts = pd.read_csv(params.predicts_path, header=None)
        predict_probas = pd.read_csv(params.predict_probas_path, header=None)

        data = read_csv(params.input_data_path)

        self.assertEqual(predicts.shape, (data.shape[0], 1))
        self.assertEqual(predict_probas.shape, (data.shape[0], 2))

        self.assertGreater(predicts.sum()[0], 0)
        self.assertGreater(predict_probas.sum()[0], 0)
        self.assertGreater(predict_probas.sum()[1], 0)

    def test_predict_pipeline(self):
        path_to_config = "ml_project/tests/configs/predict_config.yaml"
        params = read_predict_pipeline_params(path_to_config)
        self.check_pipeline(params)

        path_to_config = "ml_project/tests/samples/generated/pipelines/configs/predict_config.yaml"
        params = read_predict_pipeline_params(path_to_config)
        self.check_pipelines_on_genereted_datа(params)
