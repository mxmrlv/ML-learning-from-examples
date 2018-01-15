
from elements.tensor import layer, features, tensor
from elements import neural_network
from elements.activation_functions import unipolar_sigmoid


class Learner(object):
    def __init__(self, dimensions, labels, activation_function):
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
        :param labels: number of labels
        """
        self._dimensions = dimensions
        self._act_function = activation_function
        self._labels = {l: i for i, l in enumerate(labels)}
        self._layers = self._create_layers()
        self._neural_network = neural_network.NeuralNetwork(self._layers)

    def process_batch(self, features_matrix, labels, batch_size=1):
        for row in features_matrix:
            self.process(row, labels)

    def _translate_label(self, label):
        vector_label = [0] * self._dimensions[-1]
        vector_label[self._labels[label]] = 1
        return tensor.Tensor(vector_label)

    def process(self, features, label):
        """

        :param features: the features to flow through the neural network.
        :param label: passing labels means we are in training mode.
        :return:
        """
        output_vector = self._neural_network.process(features)
        self._neural_network.propagate_deltas(self._translate_label(label))
        self._neural_network.update_layers()
        return output_vector

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
    l = Learner((2, 4, 10, 2, 25, 3, 2), ['dog', 'cat'], unipolar_sigmoid)
    f = features.Features([2, 3])
    print f
    print
    print l

    print

    res = l.process(f, 'dog')


    print res

    print '~' *20
    print l
