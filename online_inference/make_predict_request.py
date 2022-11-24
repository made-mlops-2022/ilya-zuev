import requests
import click
import csv


def make_predict_request(path_to_data: str):
    url = "http://127.0.0.1/predict"

    with open(path_to_data) as csvfile:
        csvreader = csv.reader(csvfile, delimiter=",")
        data = [row for row in csvreader]

    test_data = {
        "data": data
    }

    response = requests.post(url, json=test_data)
    print(response.content)


@click.command(name="train_pipeline")
@click.argument("path_to_data")
def make_predict_request_command(path_to_data: str):
    make_predict_request(path_to_data)


if __name__ == "__main__":
    make_predict_request_command()
