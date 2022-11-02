from dataclasses import dataclass


@dataclass()
class TrainingParams:
    model_type: str
    random_state: int
