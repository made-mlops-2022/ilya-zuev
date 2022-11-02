import logging
import sys
import click

from data.make_dataset import read_csv
from entities.train_pipeline_params import (
    TrainingPipelineParams,
    read_training_pipeline_params,
)


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def train_pipeline(training_pipeline_params: TrainingPipelineParams):
    logger.info(f"start train pipeline with params {training_pipeline_params}")
    data = read_csv(training_pipeline_params.input_data_path)
    logger.info(f"data.shape is {data.shape}")


@click.command(name="train_pipeline")
@click.argument("config_path")
def train_pipeline_command(config_path: str):
    params = read_training_pipeline_params(config_path)
    train_pipeline(params)


if __name__ == "__main__":
    train_pipeline_command()
