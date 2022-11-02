from dataclasses import dataclass
from marshmallow_dataclass import class_schema
import yaml


@dataclass()
class TrainingPipelineParams:
    input_data_path: str


TrainingPipelineParamsSchema = class_schema(TrainingPipelineParams)


def read_training_pipeline_params(path: str) -> TrainingPipelineParams:
    with open(path, "r", encoding="utf-8") as input_stream:
        schema = TrainingPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
