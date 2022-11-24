import csv
from fastapi.testclient import TestClient

from .app import app, Data


PATH_TO_DATA = "./online_inference/data/test_data/test.csv"
PATH_TO_ANSWER = "./online_inference/data/test_data/answer.csv"

client = TestClient(app)


def test_predict():
    with open(PATH_TO_DATA) as csvfile:
        csvreader = csv.reader(csvfile, delimiter=",")
        data = [row for row in csvreader]

    with open(PATH_TO_ANSWER) as csvfile:
        csvreader = csv.reader(csvfile, delimiter=",")
        answer = [int(row[0]) for row in csvreader]

    test_data = {
        "data": data
    }

    response = client.post(
        "/predict",
        json=test_data,
    )

    assert response.status_code == 200
    assert response.json() == answer
