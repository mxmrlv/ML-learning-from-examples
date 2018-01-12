
from elements.tensor import layer, features
from elements import neural_network
from elements.activation_functions import unipolar_sigmoid


class Learner(object):
    def __init__(self, dimensions, activation_function):
        """
        Create a learner

        :param dimensions: the dimensions of the neural network. E.g. for input
        (1,2,3,2) -> there is a single feature, the first hidden layer outputs
        to 2 neurons, and the second hidden layer has outputs to 3 neurons.
        While the labels have only 2 categories.
                        ->  h_3,1
                 h_2,1  ->          ->  l_1
        x_1 ->          ->  h_3,2   ->
                 h_2,2  ->          ->  l_2
                        ->  h_3,3

        :param activation_function: the activation function of each neuron
        """
        self._dimensions = dimensions
        self._act_function = activation_function
        self._layers = self._create_layers()
        self._neural_network = neural_network.NeuralNetwork(self._layers)

    def process_batch(self, features_matrix, labels, batch_size=1):
        for row in features_matrix:
            self.process(row, labels)

    def process(self, features, labels=None):
        """

        :param features: the features to flow through the neural network.
        :param labels: passing labels means we are in training mode.
        :return:
        """
        return self._neural_network.process(features)

    def _create_layers(self):
        return [
            layer.Layer(
                self._dimensions[d],
                self._dimensions[d+1],
                act_func=self._act_function
            )
            for d, _ in enumerate(self._dimensions[:-1])
        ]

    def __str__(self):
        return str(self._neural_network)


if __name__ == '__main__':
    l = Learner((2, 3, 5, 2, 3), unipolar_sigmoid)
    f = features.Features([1, 2])

    res = l.process(f)

    print l
    print
    print f
    print


    print res
