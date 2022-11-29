from dataclasses import dataclass
from marshmallow_dataclass import class_schema
import yaml


@dataclass()
class TestParams:
    output_samples_folder: str
    output_answers_folder: str
    sample_base_name: str
    sample_count: int
    sample_rows_low: int
    sample_rows_high: int
    random_state: int


TestParamsSchema = class_schema(TestParams)


def read_test_params(path: str) -> TestParams:
    with open(path, "r", encoding="utf-8") as input_stream:
        schema = TestParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
