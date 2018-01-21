import numpy
from contextlib import contextmanager


from .elements import neural_network, functions


numpy.set_printoptions(precision=2)


class Learner(object):

    class Stats(object):
        def __init__(self):
            self.reset()

        def reset(self):
            self._right = 0
            self._wrong = 0
            self._total = 0
            self._right_dict = {}
            self._wrong_dict = {}

        @property
        def right(self):
            return self.right

        def is_right(self, label):
            self._total += 1
            self._right += 1
            self._right_dict.setdefault(label, 0)
            self._right_dict[label] += 1

        @property
        def wrong(self):
            return self._wrong

        def is_wrong(self, label, output):
            self._total += 1
            self._wrong += 1
            self._wrong_dict.setdefault(label, {}).setdefault(output, 0)
            self._wrong_dict[label][output] += 1

        @property
        def p_right(self):
            return float(self._right) / self._total

        @property
        def p_wrong(self):
            return float(self._wrong) / self._total

        def __str__(self):
            p_right = self.p_right * 100
            return '(right={self._right}):' \
                   '(wrong={self._wrong}):' \
                   '(p_right={p_right:.2f}%)'.format(self=self, p_right=p_right)

        def progress(self, total):
            c_pro = float(self._total) / total * 100
            r_pro = 100 - c_pro
            return '\r|{0}{1}|{2}'.format('>' * int(c_pro / 10),
                                          '-' * int(r_pro / 10), self)

    def __init__(
            self,
            features_d,
            labels,
            hidden_d=(),
            act_func=functions.Sigmoid,
            cost_func=functions.QuadraticCostFunction,
            step=0.003,
    ):
        """
        Create a learner for a problem.

        :param features_d: the dimension of the features
        :param labels: the available labels
        :param hidden_d: number of hidden layers, and number of neurons pair
        layer. e.g. (7, 4) is two hidden layers, with 7 neurons in the first
        hidden layer and 4 in the second.
        :param act_func: the activation function to use. Defaults to sigmoid.
        :param cost_func: the cost function to use. Builtins are 'quad` for
        quadratic, or 'cross' for cross-entropy.
        :param step: the step size.
        """
        # Adding one to the first matrix in order to handle biases.
        self._dimensions = (features_d + 1, ) + hidden_d + (len(labels),)
        self._labels = {l: i for i, l in enumerate(labels)}
        self._step = step
        self._neural_network = neural_network.NeuralNetwork(
            self._dimensions, self._step, act_func, cost_func)
        self._stats = None

    @property
    @contextmanager
    def stats(self):
        self._stats = self.Stats()
        yield self._stats

    def learn(self, features, label):
        # adding bias to features
        features = numpy.concatenate((numpy.ones((1,)), features), 0)

        features = self._neural_network.vectorize(features)

        output_vector = self._neural_network.process(features, True)
        self._neural_network.update_weights(
            self._label_to_tensor(label), features)
        self._keep_track(output_vector, label)

    def test(self, features, label):
        features = numpy.concatenate((numpy.ones((1,)), features), 0)

        features = self._neural_network.vectorize(features.flatten())
        output_vector = self._neural_network.process(features, False)
        self._keep_track(output_vector, label)

    def _keep_track(self, output_vector, label):
        if self._stats:
            if output_vector.argmax() == label:
                self._stats.is_right(label)
            else:
                self._stats.is_wrong(label, output_vector.argmax())

    def _label_to_tensor(self, label):
        vector_label = numpy.zeros((len(self._labels)))
        vector_label[self._labels[label]] = 1.
        return self._neural_network.vectorize(vector_label)

    def __str__(self):
        return str(self._neural_network)
