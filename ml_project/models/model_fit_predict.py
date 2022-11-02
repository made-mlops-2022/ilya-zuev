from typing import Dict
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score
)

from entities.train_params import TrainingParams


SklearnClassificationModel = LogisticRegression


def train_model(
    features: pd.DataFrame, target: pd.Series, train_params: TrainingParams
) -> SklearnClassificationModel:
    if train_params.model_type == "LogisticRegression":
        model = LogisticRegression(max_iter=1000, random_state=train_params.random_state)
    else:
        raise NotImplementedError()

    model.fit(features, target)

    return model


def predict_model(
    model: SklearnClassificationModel, features: pd.DataFrame
) -> np.ndarray:
    predicts = model.predict(features)
    return predicts


def predict_proba_model(
    model: SklearnClassificationModel, features: pd.DataFrame
) -> np.ndarray:
    predict_probas = model.predict_proba(features)
    return predict_probas


def evaluate_model(
    predicts: np.ndarray, predict_probas: np.ndarray, target: pd.Series
) -> Dict[str, float]:
    return {
        "accuracy_score": accuracy_score(target, predicts),
        "precision_score": precision_score(target, predicts),
        "recall_score": recall_score(target, predicts),
        "roc_auc_score": roc_auc_score(target, predict_probas[:, 1]),
    }


def serialize_model(model: SklearnClassificationModel, output: str) -> str:
    with open(output, "wb") as f:
        pickle.dump(model, f)
    return output


def deserialize_model(input_file: str) -> SklearnClassificationModel:
    with open(input_file, "rb") as f:
        model = pickle.load(f)
    return model


def serialize_predicts(predicts: np.array, output: str) -> str:
    np.savetxt(output, predicts, delimiter=",")
    return output
