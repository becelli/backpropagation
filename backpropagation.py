import time
import numpy as np

import sys


def choose_curve(is_sigmoid: bool) -> tuple[callable, callable]:
    # Returns the activation function and its derivative
    if is_sigmoid:
        def fx_sigmoid(x: np.ndarray) -> np.ndarray:
            # The same as (1 / (1 + np.exp(-x))) * 2 - 1
            return np.subtract(np.multiply(np.exp(-np.logaddexp(0, -x)), 2), 1)

        def dfx_sigmoid(x: np.ndarray) -> np.ndarray:
            sigmoid = fx_sigmoid(x)
            return 0.5 * (1 + sigmoid) * (1 - sigmoid)

        return fx_sigmoid, dfx_sigmoid

    else:
        def fx_tanh(x: np.ndarray) -> np.ndarray:
            return np.tanh(x)

        def dfx_tanh(x: np.ndarray) -> np.ndarray:
            return 1 - np.tanh(x) ** 2

        return fx_tanh, dfx_tanh


def init_weights(num_classes: int, num_features: int, num_hidden: int) -> tuple[np.ndarray, np.ndarray]:
    # Initialize the weights as small random values
    # number of inputs (class features)
    num_inputs: int = num_features
    # number of output neurons
    num_outputs: int = num_classes

    hidden_weight = np.random.randn(
        num_inputs, num_hidden) * np.sqrt(2 / (num_inputs + num_hidden))

    output_weight = np.random.randn(
        num_hidden, num_outputs) * np.sqrt(2 / (num_outputs + num_hidden))

    return hidden_weight, output_weight


def get_expected_values(is_sigmoid, classes, num_classes) -> np.ndarray:
    # calc the expected output from each neuron of the given layer
    # each layer is described by set of weights of each neuron

    # Logistic functions
    if is_sigmoid:
        matrix = np.full((classes.size, num_classes), -1, dtype=np.float64)
        matrix[np.arange(classes.size), classes.astype(int) - 1] = 1
        return matrix

    # Hyperbolic tangent
    matrix: np.ndarray = np.full((classes.size, num_classes), -1, dtype=int)
    matrix[np.arange(classes.size), classes.astype(int) - 1] = 1
    return matrix


def assign_classes(output: np.ndarray) -> np.ndarray:
    # Get the neuron that has the greater value.
    # This neuron is the one that represents the class
    return np.argmax(output, axis=1) + 1


def forward(
    fx: callable,
    inputs: np.ndarray,
    hidden_weight: np.ndarray,
    output_weight: np.ndarray,


):
    # Calculates the outputs of each layer
    hidden_layer: np.ndarray = fx(np.dot(inputs, hidden_weight))
    output_layer: np.ndarray = fx(np.dot(hidden_layer, output_weight))
    return hidden_layer, output_layer


def backward(
    dfx: callable,
    expected: np.ndarray,
    attained: np.ndarray,
    inputs: np.ndarray,
    hidden_layer: np.ndarray,
    output_weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    # Calculates the errors of each layer

    output_error = expected - attained
    output_grads = np.dot(hidden_layer.T, output_error)

    hidden_errors = np.dot(output_error, output_weight.T) * dfx(
        hidden_layer)

    hidden_grads = np.dot(inputs.T, hidden_errors)

    return hidden_grads, output_grads


def iterate(fx: callable,
            dfx: callable,
            inputs: np.ndarray,
            expected: np.ndarray,
            rate: np.float64,
            hidden_weight: np.ndarray,
            output_weight: np.ndarray,
            ):

    # Update weights on every sample that it processes
    n_samples = inputs.shape[0]
    new_hidden_weight = hidden_weight
    new_output_weight = output_weight
    # batch_size = n_samples

    for i in range(n_samples):
        entry = inputs[i].reshape(1, -1)
        expec = expected[i].reshape(1, -1)
        hidden_layer, output_layer = forward(
            fx, entry, hidden_weight, output_weight)

        hidden_grads, output_grads = backward(
            dfx, expec, output_layer, entry, hidden_layer, output_weight)

        new_hidden_weight += rate * hidden_grads
        new_output_weight += rate * output_grads

    return new_hidden_weight, new_output_weight, hidden_layer, output_layer


def train(
    inputs: np.ndarray,
    classes: np.ndarray,
    num_classes: int,
    num_features: int,
    num_hidden: int,
    rate: np.float64,
    is_sigmoid: bool,
    stop_by_error: bool,
    stop_value,


) -> tuple:

    # initialize the weights
    hidden_weight, output_weight = init_weights(
        num_classes, num_features, num_hidden)

    # get the function for the choosen curve
    fx, dfx = choose_curve(is_sigmoid)
    expected = get_expected_values(is_sigmoid, classes, num_classes)

    # Train the network
    if not stop_by_error:
        for _ in range(stop_value):
            hidden_weight, output_weight, _, _ = iterate(
                fx, dfx, inputs, expected, rate, hidden_weight, output_weight)
    else:
        error = stop_value + 1
        while error > stop_value:
            hidden_weight, output_weight, _, output_layer = iterate(
                fx, dfx, inputs, expected, rate, hidden_weight, output_weight)
            error = net_error(expected, output_layer)

    return hidden_weight, output_weight


def net_error(expected: np.ndarray, output: np.ndarray) -> np.float64:
    error = np.sum(np.power(expected - output, 2)) / 2
    print(error)
    return error


def test(
    inputs: np.ndarray,
    classes: np.ndarray,
    num_classes: int,
    is_sigmoid: bool,
    hidden_weight: np.ndarray,
    output_weight: np.ndarray,
) -> np.ndarray:

    # Get the function for the choosen curve
    fx, _ = choose_curve(is_sigmoid)

    # Test the network
    _, output_values = forward(fx, inputs, hidden_weight, output_weight)

    # get the class assigned to
    assigned_class = assign_classes(output_values)
    # get the expected class
    expected_class = classes.astype(int)

    # Create the confusion matrix using numpy
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for i in range(assigned_class.size):
        confusion_matrix[expected_class[i] - 1, assigned_class[i] - 1] += 1

    return confusion_matrix
