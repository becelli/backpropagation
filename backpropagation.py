import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.special import expit as logistic
from config import Config


def choose_curve(is_logistic: bool) -> tuple:
    if is_logistic:
        def fx_logistic(x: np.ndarray) -> np.ndarray:
            return logistic(x)

        def dfx_logistic(x: np.ndarray) -> np.ndarray:
            return logistic(x) * (1 - logistic(x))

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
        matrix = np.zeros((classes.size, num_classes))
        matrix[np.arange(classes.size), classes.astype(int)-1] = 1
        return matrix
    else:
        matrix = np.full((classes.size, num_classes), -1)
        matrix[np.arange(classes.size), classes.astype(int)-1] = 1
        return matrix


# calc the error for the output layer

def calc_output_error(d_fx, weights, expected, hidden_output, attained):
    return np.multiply(np.subtract(expected, attained), d_fx(np.dot(hidden_output, weights)))
    #return np.multiply(np.subtract(expected, attained), d_fx(output_output))


def calc_hidden_error(d_fx, weights_hidden, weights_output, error_output, inputs):
    return np.multiply(np.dot(error_output, weights_output), d_fx(np.dot(inputs, weights_hidden)))
    #return np.multiply(np.dot(error_output, weights_output), d_fx(hidden_output))


def adjust_weights(weights: np.ndarray, learning_rate: float, error: np.ndarray, layer_input: np.ndarray) -> np.ndarray:
    return weights + learning_rate*np.outer(error, layer_input)


def net_error(error_output: np.ndarray) -> float:
    return np.square(error_output).sum()/2


def assign_class(output):
    return np.argmax(output) + 1

def forward(
    fx: callable,
    inputs: np.ndarray,
    hidden_weight: np.ndarray,
    output_weight: np.ndarray,
):
    
    # print('FORWARD ----\ninputs: ', inputs.shape)
    # print('hidden_weight: ', hidden_weight.shape)
    # print('output_weight: ', output_weight.shape)

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
    
    # print("BACKWARD ----\nexpected: ", expected.shape)
    # print("attained: ", attained.shape)
    # print("inputs: ", inputs.shape)
    # print("hidden_layer: ", hidden_layer.shape)
    # print("hidden_weight: ", hidden_weight.shape)
    # print("output_weight: ", output_weight.shape)


    errors = expected - attained
    output_delta = errors * dfx(attained)
    hidden_errors = np.dot(output_delta, output_weight)
    hidden_delta = hidden_errors * dfx(hidden_layer)
    
    output_weight += hidden_layer.T.dot(output_delta)
    hidden_weight += inputs.T.dot(hidden_delta)


    return hidden_weight, output_weight

    


def train_network(sample, is_logistic: bool, learning_rate: float, max_it: int = 100, min_error: float = 0.001) -> tuple:
    # get information from the dataset
    num_classes = np.unique(sample[:, -1]).size
    num_features = sample.shape[1] - 1
    sample_size = sample.shape[0]

    # remove the last column from the inputs list, i.e., remove the class type from inputs
    inputs = np.array(np.delete(sample, sample.shape[1]-1, 1))
    classes = sample[:, -1]

    # initialize the initial weights
    hidden_weight, output_weight = init_weights(num_classes, num_features)
 
    # get the function for the choosen curve
    fx, d_fx = choose_curve(is_logistic)

    
    expected_values = calc_expected_val(
        is_logistic, classes, num_classes)
    
    

    for _ in range(1000):

        # print('weights: ', hidden_weight, output_weight)
        hidden_layer, output_layer = forward(fx, inputs, hidden_weight, output_weight)
        hidden_weight, output_weight = backward(d_fx, expected_values, output_layer, inputs, hidden_layer, hidden_weight, output_weight)
        
        error_output_layer = calc_output_error(d_fx, output_weight, expected_values, hidden_layer, output_layer)
        # if net_error(error_output_layer) < min_error:
        #     break

    
    return hidden_weight, output_weight


def test_network(sample, is_logistic: bool, hidden_weight: np.ndarray, output_weight: np.ndarray) -> np.ndarray:
    # get information from the dataset
    num_classes = np.unique(sample[:, -1]).size
    confusion_matrix = np.zeros((num_classes, num_classes))

    sample_size = sample.shape[0]
    inputs = np.array(np.delete(sample, sample.shape[1]-1, 1))
    # print(inputs)
    classes = sample[:, -1]
    # print(classes)

    fx, _ = choose_curve(is_logistic)

    
        # # calc the value for the hidden layer and the output layer
        # hidden_values = calc_layer(fx, hidden_weight, inputs[i])
        # output_values = calc_layer(fx, output_weight, hidden_values)

    hidden_values, output_values = forward(fx, inputs, hidden_weight, output_weight)
    
    print(output_values)
    
    for i in range(output_values.shape[0]):
        # get the class assigned to
        assigned_class = assign_class(output_values[i])
        # get the expected class
        expected_class = int(classes[i])
        # increment the confusion matrix
        confusion_matrix[expected_class-1][assigned_class-1] += 1

    return confusion_matrix


def plot_confusion_matrix(matrix: np.ndarray):
    plt.imshow(matrix, cmap='binary')
    plt.colorbar()
    plt.show()


def backpropagation():
    l_rate = Config.learning_rate
    max_it = Config.max_iterations
    min_error = Config.min_error
    np.random.seed(666)
    
    training_sample = Config.training_samples
    # np.random.shuffle(training_sample)
    
    is_logistic = Config.is_logistic
    hidden_weight, output_weight = train_network(
        sample=training_sample, is_logistic=is_logistic, learning_rate=l_rate, max_it=max_it, min_error=min_error)

    testing_sample = Config.test_samples
    confusion_matrix = test_network(sample=testing_sample, is_logistic=is_logistic,
                                    hidden_weight=hidden_weight, output_weight=output_weight)
    plot_confusion_matrix(confusion_matrix)
