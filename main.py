import numpy as np


class NeuralNetwork(object):
    def __init__(self, layer_sizes, layer_types):
        self.training_in = None
        self.training_out = None

        self.layers = []
        self.weights = []
        self.layer_types = layer_types

        for i in range(len(layer_sizes)):
            self.layers.append(None)

        np.random.seed(1)
        for i in range(len(layer_sizes) - 1):
            self.weights.append(2 * np.random.random((layer_sizes[i],
                                                      layer_sizes[i+1])))

    def sigmoid(self, x, deriv=False):
        if deriv:
            return x * (1 - x)

        return 1 / (1 + np.exp(-x))

    def relu(self, x, deriv=False):
        if deriv:
            return 1 * (x > 0)

        return np.maximum(x, 0)

    def train_network(self, iterations=100000):
        for j in range(iterations):
            self.layers[0] = self.training_in

            # Define layers
            for i in range(1, len(self.layers)):
                #print(i - 1)
                if "sig" in self.layer_types[i - 1]:
                    func = self.sigmoid
                else:
                    func = self.relu
                self.layers[i] = func(np.dot(self.layers[i - 1],
                                             self.weights[i - 1]))

            # Backpropagate
            errors = [self.training_out - self.layers[-1]]
            deltas = []
            for i in range(len(self.layers) - 1, 0, -1):
                # from len-2, 0 because there is no error in input
                ##layer (which is -1)
                #print("layer:", i)
                if "sig" in self.layer_types[i - 1]:
                    func = self.sigmoid
                else:
                    func = self.relu
                deltas.insert(0, errors[0] * func(self.layers[i], True))
                errors.insert(0, deltas[0].dot(self.weights[i - 1].T))

            for i in range(len(self.weights)):
                self.weights[i] += self.layers[i].T.dot(deltas[i])

            if (j % 10000) == 0:
                print("Error:" + str(np.mean(np.abs(errors[-1]))))

    def train_from_file(self, filepath):
        with open(filepath) as f:
            f = f.read()

        ins = eval(f.split("---")[0])

        outs = eval(f.split("---")[1])

        self.training_in = np.array(ins)
        self.training_out = np.array(outs)

        self.train_network()

    def max_output(self, outputs):
        max_out = 0
        index = 0
        for i in range(len(outputs)):
            if outputs[i] > max_out:
                max_out = outputs[i]
                index = i

        return index

    def interact(self, in_):
        self.layers[0] = in_
        for i in range(1, len(self.layers)):
            if "sig" in self.layer_types[i - 1]:
                func = self.sigmoid
            else:
                func = self.relu
            self.layers[i] = func(np.dot(self.layers[i - 1],
                                         self.weights[i - 1]))

        max_ = self.max_output(self.layers[-1])

        return self.layers[-1]
