
from dataclasses import dataclass
import pandas as pd
import json

global config
with open('config.json') as f:
    config: dict = json.load(f)


@dataclass
class Config:
    training_samples: pd.DataFrame = pd.read_csv(
        config['data']['training_data']).values
    test_samples: pd.DataFrame = pd.read_csv(
        config['data']['test_data']).values
    learning_rate: float = config['learning']['rate']
    max_iterations: int = config['learning']['max_iterations']
    min_error: float = config['learning']['min_error']
    is_logistic: bool = config['learning']['is_logistic']
