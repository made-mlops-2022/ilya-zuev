import os
from datetime import timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from docker.types import Mount

default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
        "download_data",
        default_args=default_args,
        schedule_interval="@daily",
        start_date=days_ago(5),
) as dag:
    download = DockerOperator(
        image="airflow-download",
        command="/data/raw/{{ ds }}",
        network_mode="bridge",
        task_id="docker-airflow-download",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source="/Users/administrator/zuev/prog/made/1_mlops/ilya-zuev/airflow_ml_dags/data/", target="/data", type='bind')]
    )
