FROM python:3.10

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./setup.py /code/setup.py
COPY ./ml_project /code/ml_project

RUN pip install --no-cache-dir .

COPY ./online_inference /code/online_inference

CMD ["uvicorn", "online_inference.app:app", "--host", "0.0.0.0", "--port", "80"]