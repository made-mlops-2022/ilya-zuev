import logging
import sys
import click

from ml_project.entities.predict_pipeline_params import (
    PredictPipelineParams,
    read_predict_pipeline_params,
)
from ml_project.features.build_features import deserialize_transformer, make_features
from ml_project.data.make_dataset import read_csv
from ml_project.models.model_fit_predict import (
    deserialize_model,
    predict_model,
    predict_proba_model,
    serialize_predicts
)


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def predict_pipeline(predict_pipeline_params: PredictPipelineParams):
    logger.info(f"start predict pipeline with params {predict_pipeline_params}")
    data = read_csv(predict_pipeline_params.input_data_path)
    logger.info(f"data.shape is {data.shape}")

    transformer = deserialize_transformer(predict_pipeline_params.feature_transformer_path)

    test_features = make_features(transformer, data)

    logger.info(f"test_features.shape is {test_features.shape}")

    model = deserialize_model(predict_pipeline_params.model_path)

    predicts = predict_model(
        model,
        test_features
    )

    logger.info(f"predicts.shape is {predicts.shape}")

    predict_probas = predict_proba_model(
        model,
        test_features
    )

    logger.info(f"predict_probas.shape is {predict_probas.shape}")

    path_to_predicts = serialize_predicts(
        predicts, predict_pipeline_params.predicts_path
    )
    path_to_predict_probas = serialize_predicts(
        predict_probas, predict_pipeline_params.predict_probas_path
    )

    return path_to_predicts, path_to_predict_probas


@click.command(name="predict_pipeline")
@click.argument("config_path")
def predict_pipeline_command(config_path: str):
    params = read_predict_pipeline_params(config_path)
    predict_pipeline(params)


if __name__ == "__main__":
    predict_pipeline_command()
