import unittest
import os
import logging
import pandas as pd
from predict_pipeline import predict_pipeline
from entities.predict_pipeline_params import read_predict_pipeline_params
from tests.samples.generated.test_params import read_test_params
from data.make_dataset import read_csv


class TestTrainPipeline(unittest.TestCase):
    PATH_TO_TEST_CONFIG = "ml_project/tests/samples/generated/test_config.yaml"

    def setUp(self):
        logging.disable(logging.CRITICAL)
        self.test_params = read_test_params(self.PATH_TO_TEST_CONFIG)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def test_train_pipeline(self):
        def check_pipeline(params):
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

        PATH_TO_CONFIG = "ml_project/tests/configs/predict_config.yaml"
        params = read_predict_pipeline_params(PATH_TO_CONFIG)
        check_pipeline(params)

        PATH_TO_CONFIG = "ml_project/tests/samples/generated/pipelines/configs/predict_config.yaml"
        params = read_predict_pipeline_params(PATH_TO_CONFIG)
        for i in range(self.test_params.sample_count):
            sample_path = os.path.join(
                self.test_params.output_samples_folder,
                f"{self.test_params.sample_base_name}{i:02d}.csv"
            )
            params.input_data_path = sample_path

            check_pipeline(params)
