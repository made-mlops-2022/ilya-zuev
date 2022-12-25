import os
from datetime import timedelta, datetime

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 0,
}

with DAG(
        "train_model",
        default_args=default_args,
        schedule_interval="@weekly",
        start_date=datetime.today(),
) as dag:
    PATH_TO_DATA = "/Users/administrator/zuev/prog/made/1_mlops/ilya-zuev/airflow_ml_dags/data/"

    preprocess = DockerOperator(
        image="airflow-preprocess",
        command="--input-dir /data/raw/{{ ds }} --output-dir /data/processed/{{ ds }}",
        task_id="docker-airflow-preprocess",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=PATH_TO_DATA, target="/data", type='bind')]
    )

    train_model = DockerOperator(
        image="airflow-train-model",
        command="--input-dir /data/processed/{{ ds }} --output-dir /data/models/{{ ds }}",
        task_id="docker-airflow-train-model",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=PATH_TO_DATA, target="/data", type='bind')]
    )

    validate = DockerOperator(
        image="airflow-validate",
        command="--data-dir /data/processed/{{ ds }} --model-dir /data/models/{{ ds }} --output-dir /data/scores/{{ ds }}",
        task_id="docker-airflow-validate",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=PATH_TO_DATA, target="/data", type='bind')]
    )

    preprocess >> train_model >> validate
