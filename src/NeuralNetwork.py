import numpy as np
import pandas as pd 
import warnings
warnings.filterwarnings("ignore")

def cost(x, y):
    return np.sum((x - y)**2)

def dcost(x, y):
    return 2.0*(x - y)

def relu(x):
    return max(0, x)

def drelu(x):
    if x < 0:
        return 0
    else:
        return 1

class Network:
    def __init__(self, layer_sizes, input_layer_values, output_layer_values, af):
        if len(layer_sizes) < 3:
            raise ValueError("No of layers should be at least 3")

        self.layer_sizes = layer_sizes
        self.num_layers = len(self.layer_sizes)
        self.network = [np.array(input_layer_values)]+[np.zeros(shape = (i, 1)).reshape(-1) for i in self.layer_sizes[1:]]
        self.biases = [None] + [np.ones(shape = (i, 1)).reshape(-1) for i in self.layer_sizes[1:]]
        self.weights = [None] + [np.ones(shape = (self.layer_sizes[i+1], self.layer_sizes[i])) for i in range(self.num_layers - 1)]
        self.target = np.array(output_layer_values)
        if af == "relu":
            self.af = np.vectorize(relu)
            self.daf = np.vectorize(drelu)
        self.grad_weights = [None] + [np.zeros(shape = (self.layer_sizes[i+1], self.layer_sizes[i])) for i in range(self.num_layers - 1)]
        self.grad_biases = [None] + [np.zeros(shape = (i, 1)).reshape(-1) for i in self.layer_sizes[1:]]

        self._forward_pass()

    def _forward_pass(self):
        self.network = [self.network[0]] + [self.af(np.dot(self.weights[i], self.network[i-1]) + self.biases[i]) for i in range(1, self.num_layers)]

    def _back_propagate(self, learning_rate, method):
        if method == "gradient descent":
            z = np.dot(self.weights[-1], self.network[-2]) + self.biases[-1]
            cost_deriv = dcost(self.network[-1], self.target)
            daf = self.daf(z)
            self.grad_weights[-1] = np.outer(cost_deriv*daf, self.network[-2])
            self.grad_biases[-1] = cost_deriv * daf
            for i in range(self.num_layers - 2, 1, -1):
                cost_deriv = np.dot(self.weights[i+1].T, cost_deriv*daf)
                z = np.dot(self.weights[i], self.network[i-1]) + self.biases[i]
                daf = self.daf(z)
                self.grad_weights[i] = np.outer(cost_deriv*daf, self.network[i-1])
                self.grad_biases[i] = cost_deriv * daf
            self.weights = [self.weights[0]] + [self.weights[i] - learning_rate*self.grad_weights[i] for i in range(1, len(self.weights))]
            self.biases = [self.biases[0]] + [self.biases[i] - learning_rate*self.grad_biases[i] for i in range(1, len(self.biases))]

    def show_network(self):
        max_size = max(self.layer_sizes)
        df = pd.DataFrame(columns = [f"layer{i}" for i in range(self.num_layers)], index = range(max_size))
        for i in range(self.num_layers):
            if len(self.network[i]) < max_size:
                df[f"layer{i}"] = np.concatenate([self.network[i], np.array([None]*(max_size - self.layer_sizes[i]))])
            else:
                df[f"layer{i}"] = self.network[i]
        print(df)

    def train(self, epochs = 100, learning_rate=0.1, method="gradient descent"):
        for i in range(epochs):
            self._back_propagate(learning_rate, method)
            self._forward_pass()

my_nn = Network([3, 10, 2, 5, 7], [2, 4, 3], [2, 4, 3, 1, 6, 2, 0], "relu")
my_nn.train(epochs = 20, learning_rate = 0.11)
my_nn.show_network()