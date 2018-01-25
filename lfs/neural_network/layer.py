

"""
Matrix wrapper with extended capabilities - such as outputs and deltas.
"""

import numpy

from . import matrix


class Layer(matrix.Matrix):
    def __init__(self, rows, cols, *args, **kwargs):
        super(Layer, self).__init__(numpy.random.rand(rows, cols),
                                    *args,
                                    **kwargs)
        self.outputs = None
        self.deltas = None

    def dot(self, other, act_func, learning=True):
        res = self.T.dot(other).T
        outputs = matrix.Matrix(numpy.array([act_func(o) for o in res]))
        self.outputs = outputs if learning else self.outputs
        return outputs.T

    def __str__(self):
        return (
            '{self.matrix}\n' 
            '(outputs={self.outputs}, deltas={self.deltas})'
            .format(self=self)
        )
