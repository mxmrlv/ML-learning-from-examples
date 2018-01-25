import numpy

from . import (
    layer,
    matrix,
    functions
)


class NeuralNetwork(object):
    def __init__(
        self,
        dimensions,
        step,
        lambda_factor=.95,
        activation=functions.Sigmoid,
        cost_func=functions.QuadraticCostFunction
    ):
        self._layers = self._create_layers(dimensions)
        self._step = step
        self._lambda = lambda_factor
        self._activation = activation()
        self._cost_func = cost_func()

    def process(self, features_vector, learning=True):
        return reduce(
            lambda l1, l2: l2.dot(l1, self._act, learning=learning),
            self._layers,
            features_vector.matrix.transpose()
        )

    def update_weights(self, label, features_vector):
        self._update_deltas(label)
        self._update_layers(features_vector)

    def _update_deltas(self, label):
        # Update top layer deltas
        self._layers[-1].deltas = matrix.Matrix(
            (-self._lambda *
            # The cost function
            self._cost_func(label, self._layers[-1]) *
            # the derivative of the outcome itself
            self._act_der(self._layers[-1].outputs)).transpose()
        )

        # Update hidden layer deltas
        def _update(layer_, next_layer):
            layer_.deltas = matrix.Matrix(
                (self._lambda *
                 next_layer.matrix.dot(next_layer.deltas.matrix) *
                 self._act_der(layer_.outputs.matrix.transpose()))
            )

            return layer_

        if len(self._layers) > 1:
            reduce(
                lambda n, c: _update(layer_=c, next_layer=n),
                reversed(self._layers)
            )

    def _update_layers(self, features_vector):

        def _update(o1, l2):
            l2.matrix -= (
                self._step *
                numpy.outer(o1.matrix.sum(axis=0), l2.deltas.matrix.sum(axis=1))
            )

            return l2.outputs

        return reduce(_update, self._layers, features_vector)

    @staticmethod
    def _wrap_functions(x, func):
        if isinstance(x, matrix.Matrix):
            return numpy.vectorize(func)(x.matrix)
        if isinstance(x, numpy.ndarray):
            return numpy.vectorize(func)(x)

        return func(x)

    def _act(self, x):
        return self._wrap_functions(x, self._activation.call)

    def _act_der(self, o):
        return self._wrap_functions(o, self._activation.der_call)

    @staticmethod
    def _create_layers(dim):
        return [layer.Layer(*dim[d:d + 2]) for d in xrange(len(dim[:-1]))]

    @staticmethod
    def vectorize(ndarray):
        return matrix.Matrix(ndarray)

    def __str__(self):
        return ('\n\n'.join(str(l) for l in self._layers) +
                '(step={self._step})'.format(self=self))
