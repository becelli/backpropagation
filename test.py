import numpy as np


def change_np(array: np.ndarray):
    array -= array + 2 * array


data = np.array([1, 2, 3, 4, 5])
change_np(data)

print(data)