import logging
import sys
import click

from data.make_dataset import read_csv, split_train_test_data
from entities.train_pipeline_params import (
    TrainingPipelineParams,
    read_training_pipeline_params,
)
from features.build_features import (
    build_transformer,
    make_features,
    extract_target
)


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


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
    train_features = make_features(transformer, train_df)
    train_target = extract_target(train_df, training_pipeline_params.feature_params)

    logger.info(f"train_features.shape is {train_features.shape}")


@click.command(name="train_pipeline")
@click.argument("config_path")
def train_pipeline_command(config_path: str):
    params = read_training_pipeline_params(config_path)
    train_pipeline(params)


if __name__ == "__main__":
    train_pipeline_command()
