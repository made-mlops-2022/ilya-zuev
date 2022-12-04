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
        "predict",
        default_args=default_args,
        schedule_interval=None,
        start_date=datetime.today(),
) as dag:
    PATH_TO_DATA = "/Users/administrator/zuev/prog/made/1_mlops/ilya-zuev/airflow_ml_dags/data/"

    predict = DockerOperator(
        image="airflow-predict",
        command="--data-dir /data/raw/{{ ds }} --model-dir /data/models/{{ ds }} --output-dir /data/predicts/{{ ds }}",
        task_id="docker-airflow-predict",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=PATH_TO_DATA, target="/data", type='bind')]
    )
