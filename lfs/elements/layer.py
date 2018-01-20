

"""
Tensor wrapper with extended capabilities - such as outputs and deltas.
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
        res = self.matrix.transpose().dot(other.vector)
        outputs = matrix.Vector(numpy.array([act_func(o) for o in res]))
        self.outputs = outputs if learning else self.outputs
        return outputs

    def __str__(self):
        return (
            '{self.matrix}\n' 
            '(outputs={self.outputs}, deltas={self.deltas})'
            .format(self=self)
        )
