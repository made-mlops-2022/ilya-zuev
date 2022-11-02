from dataclasses import dataclass


@dataclass()
class SplittingParams:
    test_size: float
    random_state: int
