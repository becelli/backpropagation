import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit as logistic
from config import Config


def choose_curve(is_logistic: bool) -> tuple:
    if is_logistic:
        def fx_logistic(x: np.ndarray) -> np.ndarray:
            return logistic(x)

        def dfx_logistic(x: np.ndarray) -> np.ndarray:
            return logistic(x)*(1 - logistic(x))

        return fx_logistic, dfx_logistic

    else:
        def fx_tanh(x: np.ndarray) -> np.ndarray:
            return np.tanh(x)

        def dfx_tanh(x: np.ndarray) -> np.ndarray:
            return 1 - np.tanh(x)**2

        return fx_tanh, dfx_tanh


# initialize the weights as small random values
def init_weights(num_classes: int, num_features: int) -> tuple:
    # number of inputs (class features)
    num_inputs = num_features
    # number of output neurons
    num_outputs = num_classes
    # number of hidden neurons (geometry mean of inputs and outputs)
    num_hidden = int(np.sqrt(num_inputs*num_outputs))
    # initialize the weights
    hidden_weight = np.random.rand(num_inputs, num_hidden)
    # Choose between the options -1 and  1
    output_weight = np.random.rand(num_hidden, num_outputs)

    return hidden_weight, output_weight

# calc the output from each neuron of the given layer
# each layer is described by set of weights of each neuron


def calc_expected_val(is_logistic, classes, num_classes):
    if is_logistic:
        matrix = np.zeros((classes.size, num_classes), dtype=int)
        matrix[np.arange(classes.size), classes.astype(int)-1] = 1
        return matrix
    else:
        matrix = np.full((classes.size, num_classes), -1, dtype=int)
        matrix[np.arange(classes.size), classes.astype(int)-1] = 1
        return matrix


# calc the error for the output layer

# def calc_output_error(d_fx, weights, expected, hidden_output, attained):
#     return np.multiply(np.subtract(expected, attained), d_fx(np.dot(hidden_output, weights)))
#     #return np.multiply(np.subtract(expected, attained), d_fx(output_output))


# def calc_hidden_error(d_fx, weights_hidden, weights_output, error_output, inputs):
#     return np.multiply(np.dot(error_output, weights_output), d_fx(np.dot(inputs, weights_hidden)))
#     #return np.multiply(np.dot(error_output, weights_output), d_fx(hidden_output))


# def adjust_weights(weights: np.ndarray, learning_rate: float, error: np.ndarray, layer_input: np.ndarray) -> np.ndarray:
#     return weights + learning_rate*np.outer(error, layer_input)


# def net_error(error_output: np.ndarray) -> float:
#     return np.square(error_output).sum()/2


def assign_classes(output: np.ndarray) -> np.ndarray:
    return np.argmax(output, axis=1)+1


def forward(
    fx: callable,
    inputs: np.ndarray,
    hidden_weight: np.ndarray,
    output_weight: np.ndarray,
):
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
):
    errors = np.subtract(expected, attained)
    output_delta = np.multiply(errors, dfx(attained))

    hidden_errors = np.dot(output_delta, output_weight.T)
    hidden_delta = np.multiply(hidden_errors, dfx(hidden_layer))

    hidden_weight = np.add(hidden_weight, np.dot(inputs.T, hidden_delta))
    output_weight = np.add(output_weight, np.dot(hidden_layer.T, output_delta))

    return hidden_weight, output_weight


def train_network(sample, is_logistic: bool, learning_rate: float, max_it: int = 100, min_error: float = 0.001) -> tuple:
    # Get the inputs from the classes
    inputs = sample[:, :-1]
    classes = sample[:, -1]
    # Get the number of classes and features
    num_classes = np.unique(classes).size
    num_features = sample.shape[1] - 1

    # initialize the weights
    hidden_weight, output_weight = init_weights(num_classes, num_features)

    # get the function for the choosen curve
    fx, d_fx = choose_curve(is_logistic)
    expected_values = calc_expected_val(is_logistic, classes, num_classes)

    for _ in range(max_it):

        hidden_layer, output_layer = forward(
            fx, inputs, hidden_weight, output_weight)
        hidden_weight, output_weight = backward(
            d_fx, expected_values, output_layer, inputs, hidden_layer, hidden_weight, output_weight)

        # error_output_layer = calc_output_error(d_fx, output_weight, expected_values, hidden_layer, output_layer)
        # if net_error(error_output_layer) < min_error:
        #     break

    return hidden_weight, output_weight


def test_network(
    sample: np.ndarray,
    is_logistic: bool,
    hidden_weight: np.ndarray,
    output_weight: np.ndarray
) -> np.ndarray:

    inputs = sample[:, :-1]
    classes = sample[:, -1]
    num_classes = np.unique(classes).size

    fx, _ = choose_curve(is_logistic)

    _, output_values = forward(fx, inputs, hidden_weight, output_weight)

    # get the class assigned to
    assigned_class = assign_classes(output_values)
    # get the expected class
    expected_class = classes.astype(int)

    # Create the confusion matrix using numpy
    confusion_matrix = np.zeros((num_classes, num_classes))
    for i in range(assigned_class.size):
        confusion_matrix[expected_class[i]-1][assigned_class[i]-1] += 1

    return confusion_matrix


def normalize(sample: np.ndarray) -> np.ndarray:
    # get the max and min values for each column (ignore the last column)
    max_values = np.amax(sample[:, :-1], axis=0)
    min_values = np.amin(sample[:, :-1], axis=0)

    # normalize the values
    sample[:, :-1] = (sample[:, :-1] - min_values)/(max_values - min_values)
    return sample


def plot_confusion_matrix(matrix: np.ndarray):
    # plot the confusion matrix.
    plt.imshow(matrix, cmap='Blues')
    # Plus, on each cell, add the number of samples
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, int(matrix[i, j]), ha='center', va='center')
    plt.show()


def backpropagation():
    l_rate = Config.learning_rate
    max_it = Config.max_iterations
    min_error = Config.min_error
    is_logistic = Config.is_logistic
    training_sample = Config.training_samples
    testing_sample = Config.test_samples

    training_sample = normalize(training_sample)
    testing_sample = normalize(testing_sample)
    np.random.shuffle(training_sample)

    hidden_weight, output_weight = train_network(
        sample=training_sample,
        is_logistic=is_logistic,
        learning_rate=l_rate,
        max_it=max_it,
        min_error=min_error
    )

    confusion_matrix = test_network(
        sample=testing_sample,
        is_logistic=is_logistic,
        hidden_weight=hidden_weight,
        output_weight=output_weight
    )

    plot_confusion_matrix(confusion_matrix)
