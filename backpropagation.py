import numpy as np
from scipy.special import expit as logistic


def choose_curve(is_logistic: bool) -> tuple[callable, callable]:
    # Returns the activation function and its derivative
    if is_logistic:
        def fx_logistic(x: np.ndarray) -> np.ndarray:
            # return 1.0 / (1.0 + np.exp(-x))
            return logistic(x)

        def dfx_logistic(x: np.ndarray) -> np.ndarray:
            return fx_logistic(x) * (1 - fx_logistic(x))

        return fx_logistic, dfx_logistic

    else:
        def fx_tanh(x: np.ndarray) -> np.ndarray:
            return np.tanh(x)

        def dfx_tanh(x: np.ndarray) -> np.ndarray:
            return 1 - np.tanh(x) ** 2

        return fx_tanh, dfx_tanh


def init_weights(num_classes: int, num_features: int, num_hidden: int) -> tuple[np.ndarray, np.ndarray]:
    # Initialize the weights as small random values
    # number of inputs (class features)
    num_inputs = num_features
    # number of output neurons
    num_outputs = num_classes
    # number of hidden neurons (geometry mean of inputs and outputs)
    # initialize the weights
    hidden_weight = np.random.rand(num_inputs, num_hidden) / 1000
    # Choose between the options -1 and  1
    output_weight = np.random.rand(num_hidden, num_outputs) / 1000

    return hidden_weight, output_weight


def get_expected_values(is_logistic, classes, num_classes) -> np.ndarray:
    # calc the expected output from each neuron of the given layer
    # each layer is described by set of weights of each neuron

    # Logistic functions
    if is_logistic:
        matrix = np.zeros((classes.size, num_classes), dtype=int)
        matrix[np.arange(classes.size), classes.astype(int) - 1] = 1
        return matrix

    # Hyperbolic tangent
    matrix = np.full((classes.size, num_classes), -1, dtype=int)
    matrix[np.arange(classes.size), classes.astype(int) - 1] = 1
    return matrix


def calc_output_error(d_fx, weights, expected, hidden_output, attained):
    # calc the error for the output layer
    errors = np.subtract(expected, attained)
    return np.multiply(errors, d_fx(attained))


def net_error(error_output: np.ndarray) -> float:
    return np.divide(np.square(error_output).sum(), 2)


def assign_classes(output: np.ndarray) -> np.ndarray:
    # Get the neuron that has the greater value.
    return np.argmax(output, axis=1) + 1


def forward(
    fx: callable,
    inputs: np.ndarray,
    hidden_weight: np.ndarray,
    output_weight: np.ndarray,
):
    # Calculates the outputs of each layer
    hidden_layer = fx(np.dot(inputs, hidden_weight))
    output_layer = fx(np.dot(hidden_layer, output_weight))
    return hidden_layer, output_layer


def backward(
    dfx: callable,
    expected: np.ndarray,
    attained: np.ndarray,
    inputs: np.ndarray,
    hidden_layer: np.ndarray,
    hidden_weight: np.ndarray,
    output_weight: np.ndarray,
    rate: np.float64,
):
    # Calculates the errors of each layer

    errors = np.subtract(expected, attained)
    output_delta = np.multiply(errors, dfx(attained))

    hidden_errors = np.dot(output_delta, output_weight.T)
    hidden_delta = np.multiply(hidden_errors, dfx(hidden_layer))

    hidden_weight = np.add(hidden_weight, np.dot(
        inputs.T, hidden_delta) * rate)
    output_weight = np.add(output_weight, np.dot(
        hidden_layer.T, output_delta) * rate)

    return hidden_weight, output_weight


def iterate(fx: callable,
            dfx: callable,
            inputs: np.ndarray,
            expected_values: np.ndarray,
            rate: np.float64,
            hidden_weight: np.ndarray,
            output_weight: np.ndarray
            ):

    new_hidden_layer, new_output_layer = forward(
        fx, inputs, hidden_weight, output_weight)

    new_hidden_weight, new_output_weight = backward(
        dfx,
        expected_values,
        new_output_layer,
        inputs,
        new_hidden_layer,
        hidden_weight,
        output_weight,
        rate,
    )

    return new_hidden_weight, new_output_weight, new_hidden_layer, new_output_layer


def train(
    inputs: np.ndarray,
    classes: np.ndarray,
    num_classes: int,
    num_features: int,
    num_hidden: int,
    rate: np.float64,
    is_logistic: bool,
    stop_by_error: bool,
    stop_value,
) -> tuple:

    # initialize the weights
    hidden_weight, output_weight = init_weights(
        num_classes, num_features, num_hidden)

    # get the function for the choosen curve
    fx, dfx = choose_curve(is_logistic)
    expected = get_expected_values(is_logistic, classes, num_classes)

    # Train the network
    if stop_by_error:
        network_error = np.inf
        while network_error > stop_value:
            hidden_weight, output_weight, hidden_layer, output_layer = iterate(
                fx, dfx, inputs, expected, rate, hidden_weight, output_weight)

            new_error = calc_output_error(
                dfx, output_weight, expected, hidden_layer, output_layer)

            network_error = net_error(new_error)

    else:
        for _ in range(stop_value):
            hidden_weight, output_weight, _, _ = iterate(
                fx, dfx, inputs, expected, rate, hidden_weight, output_weight)

    return hidden_weight, output_weight


def test(
    inputs: np.ndarray,
    classes: np.ndarray,
    num_classes: int,
    is_logistic: bool,
    hidden_weight: np.ndarray,
    output_weight: np.ndarray,
) -> np.ndarray:

    # Get the function for the choosen curve
    fx, _ = choose_curve(is_logistic)

    # Test the network
    _, output_values = forward(fx, inputs, hidden_weight, output_weight)

    # get the class assigned to
    assigned_class = assign_classes(output_values)
    # get the expected class
    expected_class = classes.astype(int)

    # Create the confusion matrix using numpy
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for i in range(assigned_class.size):
        confusion_matrix[expected_class[i] - 1][assigned_class[i] - 1] += 1

    return confusion_matrix
