import unittest
import logging
import pandas as pd
from ml_project.features.abs_transformer import AbsTransformer


class TestAbsTransformer(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def test_abs_tranformer(self):
        data = pd.DataFrame({"col": [0, -1, 2, -3]})
        transformer = AbsTransformer()

        transformer.fit(data.values)
        transformed_data = transformer.transform(data)

        self.assertEqual(list(transformed_data.values), [0, 1, 2, 3])
