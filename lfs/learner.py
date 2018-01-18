from contextlib import contextmanager

import numpy

from elements import neural_network


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
        return float(self._right) / self._total * 100

    @property
    def p_wrong(self):
        return float(self._wrong) / self._total * 100

    def __str__(self):
        return '(right={self._right}):' \
               '(wrong={self._wrong}):' \
               '(p_right={self.p_right:.2f}%)'.format(self=self)

    def progress(self, total):
        c_pro = float(self._total) / total * 100
        r_pro = 100 - c_pro
        return '\r|{0}{1}|{2}'.format('>' * int(c_pro / 10),
                                      '-' * int(r_pro / 10), self)


class Learner(object):
    def __init__(self, features_d, labels, step=0.003, hidden_d=()):
        """Create a learner"""
        self._dimensions = (features_d, ) + hidden_d + (len(labels),)
        self._labels = {l: i for i, l in enumerate(labels)}
        self._step = step
        self._neural_network = neural_network.NeuralNetwork(
            self._dimensions, self._step)

        self._stats = None

    @property
    @contextmanager
    def stats(self):
        self._stats = Stats()
        yield self._stats

    def learn(self, features, label):
        features = self._neural_network.vectorize(features.flatten())
        output_vector = self._neural_network.process(features, True)
        self._neural_network.update_layers(self._label_to_tensor(label), features)
        self._keep_track(output_vector, label)

    def test(self, features, label):
        features = self._neural_network.vectorize(features.flatten())
        output_vector = self._neural_network.process(features, False)
        self._keep_track(output_vector, label)

    def _keep_track(self, output_vector, label):
        if self._stats:
            if output_vector.argmax() == label:
                self._stats.is_right(label)
            else:
                self._stats.is_wrong(label, output_vector.argmax())

    def _process(self, features, learning=True):
        """

        :param features: the features to flow through the neural network.
        :param label: passing labels means we are in training mode.
        :param testing: whether we are in learning or testing mode. in testing
        mode, no changes will be made to the neural network
        :return:
        """
        return

    def _label_to_tensor(self, label):
        vector_label = numpy.zeros((len(self._labels)))
        vector_label[self._labels[label]] = 1.
        return self._neural_network.vectorize(vector_label)

    def __str__(self):
        return str(self._neural_network)
