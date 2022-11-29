import unittest
import logging
import os
import json
import pandas as pd
from ml_project.train_pipeline import train_pipeline
from ml_project.predict_pipeline import predict_pipeline
from ml_project.entities.train_pipeline_params import read_training_pipeline_params
from ml_project.entities.predict_pipeline_params import read_predict_pipeline_params
from ml_project.tests.samples.generated.test_params import read_test_params
from ml_project.data.make_dataset import read_csv


class TestPipeline(unittest.TestCase):
    PATH_TO_TEST_CONFIG = "ml_project/tests/samples/generated/test_config.yaml"

    def setUp(self):
        logging.disable(logging.CRITICAL)
        self.test_params = read_test_params(self.PATH_TO_TEST_CONFIG)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def check_pipeline(self, list_params):
        raise NotImplementedError

    def check_pipelines_on_generated_data(self, list_params):
        for i in range(self.test_params.sample_count):
            sample_path = os.path.join(
                self.test_params.output_samples_folder,
                f"{self.test_params.sample_base_name}{i:02d}.csv"
            )
            for params in list_params:
                params.input_data_path = sample_path

            self.check_pipeline(list_params)


class TestTrainPipeline(TestPipeline):
    def check_pipeline(self, list_params):
        for params in list_params:
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
        self.check_pipeline([params])

        path_to_config = "ml_project/tests/samples/generated/pipelines/configs/train_config.yaml"
        params = read_training_pipeline_params(path_to_config)
        self.check_pipelines_on_generated_data([params])


class TestPredictPipeline(TestPipeline):
    def check_pipeline(self, list_params):
        train_pipeline(list_params[0])

        predict_pipeline(list_params[1])

        self.assertTrue(os.path.isfile(list_params[1].predicts_path))
        self.assertTrue(os.path.isfile(list_params[1].predict_probas_path))

        predicts = pd.read_csv(list_params[1].predicts_path, header=None)
        predict_probas = pd.read_csv(list_params[1].predict_probas_path, header=None)

        data = read_csv(list_params[1].input_data_path)

        self.assertEqual(predicts.shape, (data.shape[0], 1))
        self.assertEqual(predict_probas.shape, (data.shape[0], 2))

        self.assertGreater(predicts.sum()[0], 0)
        self.assertGreater(predict_probas.sum()[0], 0)
        self.assertGreater(predict_probas.sum()[1], 0)

    def test_predict_pipeline(self):
        path_to_config_base = "ml_project/tests/configs/"
        path_to_train_config = f"{path_to_config_base}/train_config.yaml"
        training_params = read_training_pipeline_params(path_to_train_config)
        path_to_predict_config = f"{path_to_config_base}/predict_config.yaml"
        predict_params = read_predict_pipeline_params(path_to_predict_config)
        self.check_pipeline([training_params, predict_params])

        path_to_config_base = "ml_project/tests/samples/generated/pipelines/configs"
        path_to_train_config = f"{path_to_config_base}/train_config.yaml"
        training_params = read_training_pipeline_params(path_to_train_config)
        path_to_predict_config = f"{path_to_config_base}/predict_config.yaml"
        predict_params = read_predict_pipeline_params(path_to_predict_config)
        self.check_pipelines_on_generated_data([training_params, predict_params])
