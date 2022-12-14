import logging
import sys
import json
import click
import pandas as pd
from ml_project.entities.train_pipeline_params import (
    TrainingPipelineParams,
    read_training_pipeline_params,
)
from ml_project.features.build_features import (
    build_transformer,
    make_features,
    extract_target,
    serialize_transformer
)
from ml_project.models.model_fit_predict import (
    train_model,
    predict_model,
    predict_proba_model,
    evaluate_model,
    serialize_model,
    SklearnClassificationModel
)
from ml_project.data.make_dataset import read_csv, split_train_test_data


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def predict_and_evaluate_model(
    model: SklearnClassificationModel,
    feautres: pd.DataFrame,
    target: pd.Series
):
    predicts = predict_model(
        model,
        feautres
    )

    predict_probas = predict_proba_model(
        model,
        feautres
    )

    metrics = evaluate_model(
        predicts,
        predict_probas,
        target
    )

    return predicts, predict_probas, metrics


def train_pipeline(training_pipeline_params: TrainingPipelineParams):
    logger.info(f"start train pipeline with params {training_pipeline_params}")
    data = read_csv(training_pipeline_params.input_data_path)
    logger.info(f"data.shape is {data.shape}")

    train_df, test_df = split_train_test_data(
        data, training_pipeline_params.splitting_params
    )
    logger.info(f"train_df.shape is {train_df.shape}")
    logger.info(f"test_df.shape is {test_df.shape}")

    transformer = build_transformer(training_pipeline_params.feature_params)
    transformer.fit(train_df)
    path_to_transformer = serialize_transformer(
        transformer, training_pipeline_params.feature_transformer_path
    )

    train_features = make_features(transformer, train_df)
    train_target = extract_target(train_df, training_pipeline_params.feature_params)

    logger.info(f"train_features.shape is {train_features.shape}")

    model = train_model(
        train_features, train_target, training_pipeline_params.training_params
    )

    test_features = make_features(transformer, test_df)
    test_target = extract_target(test_df, training_pipeline_params.feature_params)

    logger.info(f"test_features.shape is {test_features.shape}")

    _, _, metrics = predict_and_evaluate_model(
        model,
        test_features,
        test_target
    )

    logger.info(f"metrics is {metrics}")

    with open(training_pipeline_params.metric_path, "w", encoding="utf-8") as metric_file:
        json.dump(metrics, metric_file)

    path_to_model = serialize_model(model, training_pipeline_params.output_model_path)

    return path_to_model, metrics, path_to_transformer


@click.command(name="train_pipeline")
@click.argument("config_path")
def train_pipeline_command(config_path: str):
    params = read_training_pipeline_params(config_path)
    train_pipeline(params)


if __name__ == "__main__":
    train_pipeline_command()
