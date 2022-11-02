import pandas as pd
from sklearn.linear_model import LogisticRegression

from entities.train_params import TrainingParams


SklearnRegressionModel = LogisticRegression


def train_model(
    features: pd.DataFrame, target: pd.Series, train_params: TrainingParams
) -> SklearnRegressionModel:
    if train_params.model_type == "LogisticRegression":
        model = LogisticRegression(max_iter=1000, random_state=train_params.random_state)
    else:
        raise NotImplementedError()

    model.fit(features, target)

    return model
