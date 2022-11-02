from dataclasses import dataclass
from marshmallow_dataclass import class_schema
import yaml
from .split_params import SplittingParams
from .feature_params import FeatureParams
from .train_params import TrainingParams


@dataclass()
class TrainingPipelineParams:
    input_data_path: str
    metric_path: str
    output_model_path: str
    feature_transformer_path: str
    splitting_params: SplittingParams
    feature_params: FeatureParams
    training_params: TrainingParams


TrainingPipelineParamsSchema = class_schema(TrainingPipelineParams)


def read_training_pipeline_params(path: str) -> TrainingPipelineParams:
    with open(path, "r", encoding="utf-8") as input_stream:
        schema = TrainingPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
