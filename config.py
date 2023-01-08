from dataclasses import dataclass
import pandas as pd
import numpy as np
import json

global config
with open('config.json') as f:
    config: dict = json.load(f)


def get_training_samples():
    path = config['data']['training_data']
    data = pd.read_csv(path)
    return np.array(data.values, dtype=np.float64)


def get_test_samples():
    path = config['data']['test_data']
    data = pd.read_csv(path)
    return np.array(data.values, dtype=np.float64)


@dataclass
class Config:
    training_samples: pd.DataFrame = get_training_samples()
    test_samples: pd.DataFrame = get_test_samples()
    learning_rate: float = config['learning']['rate']
    max_iterations: int = config['learning']['max_iterations']
    min_error: float = config['learning']['min_error']
    is_logistic: bool = config['learning']['is_logistic']
