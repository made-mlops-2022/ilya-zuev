import unittest
import os
import logging
import pandas as pd
from predict_pipeline import predict_pipeline
from entities.predict_pipeline_params import read_predict_pipeline_params
from data.make_dataset import read_csv


class TestTrainPipeline(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def test_train_pipeline(self):
        PATH_TO_CONFIG = "ml_project/tests/configs/predict_config.yaml"
        params = read_predict_pipeline_params(PATH_TO_CONFIG)
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
