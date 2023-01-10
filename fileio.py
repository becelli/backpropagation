import numpy as np
from PySide6.QtWidgets import QFileDialog


def suggest_numof_hidden(num_features: int, num_classes: int) -> int:
    num_inputs = num_features
    num_outputs = num_classes
    return int(np.sqrt(num_inputs * num_outputs))


def normalize(sample: np.ndarray) -> np.ndarray:
    # get the max and min values for each column
    max_values = np.amax(sample, axis=0)
    min_values = np.amin(sample, axis=0)

    # normalize the values
    normalized = (sample - min_values) / (max_values - min_values)
    return normalized


def read_sample(window) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    testing_file = QFileDialog.getOpenFileName(
        window, "Selecinar amostra", "", "CSV Files (*.csv)")
    filename = testing_file[0]

    try:
        with open(filename) as f:

            # Read csv with header (first row is the header)
            data: np.ndarray = np.genfromtxt(
                filename, delimiter=',', skip_header=1)

            values = data
            np.random.shuffle(values)

            # Get the inputs and classes
            inputs: np.ndarray = data[:, :-1]
            inputs = normalize(inputs)
            classes: np.ndarray = data[:, -1]

            # Get the number of classes and features
            num_classes = np.unique(classes).size
            num_features = inputs.shape[1]

            num_hidden = suggest_numof_hidden(num_features, num_classes)

            return inputs, classes, num_classes, num_features, num_hidden
    except Exception as e:
        print(e)
        raise e
