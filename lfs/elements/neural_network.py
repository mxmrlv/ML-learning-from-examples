import numpy

from . import layer, matrix


class NeuralNetwork(object):
    def __init__(self, dimensions, step):
        self._layers = self._create_layers(dimensions)
        self._step = step

    def process(self, features_vector, learning=True):
        return reduce(
            lambda l1, l2: l2.dot(l1, learning),
            self._layers,
            features_vector
        )

    def update_layers(self, label, features_vector):
        self._propagate_deltas(label)
        self._update_layers(features_vector)

    def _propagate_deltas(self, label):
        # calculate the top layer deltas
        self._layers[-1].update_deltas(label=label)

        # calculate the rest of the layers deltas
        if len(self._layers) > 1:
            reduce(lambda n, c: c.update_deltas(next_layer=n),
                   reversed(self._layers))

    def _update_layers(self, features_vector):

        def _update(o1, l2):
            l2.matrix -= self._step * numpy.outer(l2.deltas.vector, o1.vector).transpose()

        return reduce(_update, self._layers, features_vector)

    @staticmethod
    def _create_layers(dimensions):
        return [
            layer.Layer(*dimensions[d:d+2])
            for d in xrange(len(dimensions[:-1]))
        ]

    def __str__(self):
        return ('\n\n'.join(str(l) for l in self._layers) +
                '(step={self._step})'.format(self=self))

    @staticmethod
    def vectorize(ndarray):
        return matrix.Vector(ndarray)