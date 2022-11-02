# Ilya Zuev. MADE 2022.

Installation:
~~~
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
~~~

Train:
~~~
python3 ml_project/train_pipeline.py ml_project/configs/train_config.yaml
~~~

Predict:
~~~
python3 ml_project/predict_pipeline.py ml_project/configs/predict_config.yaml
~~~