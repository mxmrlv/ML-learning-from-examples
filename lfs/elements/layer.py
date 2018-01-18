

"""
Tensor wrapper with extended capabilities - such as outputs and deltas.
"""

import numpy

from . import matrix


def unipolar_sigmoid(x):
    return 1 / (1 + numpy.exp(-x))


class Layer(matrix.Matrix):
    def __init__(self, rows, cols, act_func=unipolar_sigmoid, *args, **kwargs):
        super(Layer, self).__init__(numpy.random.rand(rows, cols),
                                    *args,
                                    **kwargs)
        self._act_func = act_func
        self._outputs = None
        self._deltas = None
        self._lambda = 1

    def dot(self, other, learning=True, *args, **kwargs):
        res = self.matrix.transpose().dot(other.vector)
        outputs = matrix.Vector(numpy.array([self._act_func(o) for o in res]))
        if learning:
            self._outputs = outputs
        return outputs

    def update_deltas(self, label=None, next_layer=None):
        if label:
            assert next_layer is None
            # j = i + 1; delta_j = -lambda * o_j * (1 - o_j) * (d_j - o_j)
            deltas = (
                    # The derivative of the actual distance from the label.
                    -(label.vector - self.outputs.vector) *

                    # the derivative of the outcome itself
                    (-self._lambda) *
                    self.outputs.vector *
                    (1 - self.outputs.vector)
            )
        else:
            assert next_layer is not None
            # j = i + 1; delta_i = lambda * o_i * (1 - o_i) * (W_[:, i] * delta_j)
            deltas = numpy.array([
                self._lambda *
                self.outputs.vector[i] *
                (1 - self.outputs.vector[i]) *
                next_layer.matrix[i].dot(next_layer.deltas.vector)

                for i in xrange(len(self.outputs.vector))
            ])

        self._deltas = matrix.Vector(deltas)
        return self

    @property
    def matrix(self):
        return self._matrix

    @matrix.setter
    def matrix(self, new_matrix):
        self._matrix = new_matrix

    @property
    def outputs(self):
        return self._outputs

    @property
    def deltas(self):
        return self._deltas

    def __str__(self):
        return (
            '{self.matrix}\n' 
            '(outputs={self.outputs}, deltas={self.deltas})'
            .format(self=self)
        )
