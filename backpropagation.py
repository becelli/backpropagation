import numpy as np
import time
import matplotlib.pyplot as plt
from config import Config


def choose_curve(is_logistic: bool) -> tuple:
    if is_logistic:
        def fx_logistic(x: np.ndarray) -> np.ndarray:
            return 1.0/(1+np.exp(-x))

        def dfx_logistic(x: np.ndarray) -> np.ndarray:
            return np.exp(x)/(1+np.exp(x))**2.0

        return fx_logistic, dfx_logistic

    else:
        def fx_tanh(x: np.ndarray) -> np.ndarray:
            return np.tanh(x)

        def dfx_tanh(x: np.ndarray) -> np.ndarray:
            return (1.0/np.cosh(x))**2.0

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
    weight_hidden = np.full((num_hidden, num_inputs), 0.0055)
    # Choose between the options -1 and  1
    weight_output = np.zeros((num_outputs, num_hidden))
    for i in range(num_outputs):
        for j in range(num_hidden):
            if np.random.rand() > 0.5:
                weight_output[i][j] = 1.5
            else:
                weight_output[i][j] = -1.5

    return weight_hidden, weight_output

# calc the output from each neuron of the given layer
# each layer is described by set of weights of each neuron


def calc_layer(fx, weights_layer, layer_inputs):
    return fx(np.dot(weights_layer, layer_inputs))


def calc_expected_val(is_logistic, classes, num_classes):
    if is_logistic:
        matrix = np.zeros((classes.size, num_classes))
        matrix[np.arange(classes.size), classes.astype(int)-1] = 1
        return matrix
    else:
        matrix = np.full((classes.size, num_classes), -1)
        matrix[np.arange(classes.size), classes.astype(int)-1] = 1
        return matrix


# calc the error for the output layer

def calc_output_error(d_fx, weights, expected, attained, hidden_output):
    return np.multiply(np.subtract(expected, attained), d_fx(np.dot(weights, hidden_output)))


def calc_hidden_error(d_fx, weights_hidden, weights_output, error_output, inputs):
    return np.multiply(np.dot(error_output, weights_output), d_fx(np.dot(inputs, weights_hidden.T)))


def adjust_weights(weights: np.ndarray, learning_rate: float, error: np.ndarray, layer_input: np.ndarray) -> np.ndarray:
    return weights + learning_rate*np.outer(error, layer_input)


def net_error(error_output: np.ndarray) -> float:
    return np.square(error_output).sum()


def assign_class(output):
    return np.argmax(output) + 1


def train_network(sample, is_logistic: bool, learning_rate: float = 0.0001, max_it: int = 100, min_error: float = 0.001) -> tuple:
    # get information from the dataset
    num_classes = np.unique(sample[:, -1]).size
    num_features = sample.shape[1] - 1
    sample_size = sample.shape[0]

    # remove the last column from the inputs list, i.e., remove the class type from inputs
    inputs = np.array(np.delete(sample, sample.shape[1]-1, 1))
    classes = sample[:, -1]

    # initialize the initial weights
    weight_hidden, weight_output = init_weights(num_classes, num_features)
    global t_expected, t_hidden, t_output, t_out_error, t_hidden_error, t_adjust_output, t_adjust_hidden
    t_expected = t_hidden = t_output = t_out_error = t_hidden_error = t_adjust_output = t_adjust_hidden = 0

    # get the function for the choosen curve
    fx, d_fx = choose_curve(is_logistic)

    start = time.perf_counter()
    expected_values = calc_expected_val(
        is_logistic, classes, num_classes)
    t_expected += time.perf_counter() - start

    for j in range(max_it):
        # train the net for sample in dataset
        for i in range(sample_size):

            # calc the output for the hidden layer and the output layer
            start = time.perf_counter()
            hidden_values = calc_layer(fx, weight_hidden, inputs[i])
            t_hidden += time.perf_counter() - start

            start = time.perf_counter()
            output_values = calc_layer(fx, weight_output, hidden_values)
            t_output += time.perf_counter() - start

            # calc the error for each layer
            start = time.perf_counter()
            error_output_layer = calc_output_error(
                d_fx, weight_output, expected_values[i], output_values, hidden_values)

            t_out_error += time.perf_counter() - start

            start = time.perf_counter()
            error_hidden_layer = calc_hidden_error(
                d_fx, weight_hidden, weight_output, error_output_layer, inputs[i])

            t_hidden_error += time.perf_counter() - start

            # adjust weights for each layer
            start = time.perf_counter()

            weight_output = adjust_weights(
                weight_output, learning_rate, error_output_layer, hidden_values)

            t_adjust_output += time.perf_counter() - start

            start = time.perf_counter()

            weight_hidden = adjust_weights(
                weight_hidden, learning_rate, error_hidden_layer, inputs[i])

            t_adjust_hidden += time.perf_counter() - start
        net_error_ = net_error(error_output_layer)
        print("Iteration: ", j, "error: ", round(net_error_, 4))

        if net_error_ < min_error:
            break

    print("Time for...")
    print("Calculating expected values: ", t_expected)
    print("Calculating hidden values: ", t_hidden)
    print("Calculating output values: ", t_output)
    print("Calculating output error: ", t_out_error)
    print("Calculating hidden error: ", t_hidden_error)
    print("Adjusting output weights: ", t_adjust_output)
    print("Adjusting hidden weights: ", t_adjust_hidden)

    # return the trained values for each layer
    return weight_hidden, weight_output


def test_network(sample, is_logistic: bool, weight_hidden: np.ndarray, weight_output: np.ndarray) -> np.ndarray:
    # get information from the dataset
    num_classes = np.unique(sample[:, -1]).size
    confusion_matrix = np.zeros((num_classes, num_classes))

    sample_size = sample.shape[0]
    inputs = np.array(np.delete(sample, sample.shape[1]-1, 1))
    classes = sample[:, -1]

    fx, _ = choose_curve(is_logistic)

    for i in range(0, sample_size):
        # calc the value for the hidden layer and the output layer
        hidden_values = calc_layer(fx, weight_hidden, inputs[i])
        output_values = calc_layer(fx, weight_output, hidden_values)

        assigned_class = assign_class(output_values)
        confusion_matrix[int(classes[i])-1][assigned_class-1] += 1

    return confusion_matrix


def plot_confusion_matrix(matrix: np.ndarray):
    plt.imshow(matrix, cmap='binary')
    plt.colorbar()
    plt.show()


def backpropagation():
    l_rate = Config.learning_rate
    max_it = Config.max_iterations
    min_error = Config.min_error
    # TODO: remove this line

    # np.random.seed(Config.training_samples.shape[1] - 3)
    training_sample = Config.training_samples
    np.random.shuffle(training_sample)
    is_logistic = Config.is_logistic
    weight_hidden, weight_output = train_network(
        sample=training_sample, is_logistic=is_logistic, learning_rate=l_rate, max_it=max_it, min_error=min_error)

    testing_sample = Config.test_samples
    confusion_matrix = test_network(sample=testing_sample, is_logistic=is_logistic,
                                    weight_hidden=weight_hidden, weight_output=weight_output)
    plot_confusion_matrix(confusion_matrix)
