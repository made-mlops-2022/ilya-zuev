from dataclasses import dataclass
from marshmallow_dataclass import class_schema
import yaml
from .split_params import SplittingParams


@dataclass()
class TrainingPipelineParams:
    input_data_path: str
    splitting_params: SplittingParams


TrainingPipelineParamsSchema = class_schema(TrainingPipelineParams)


def read_training_pipeline_params(path: str) -> TrainingPipelineParams:
    with open(path, "r", encoding="utf-8") as input_stream:
        schema = TrainingPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
