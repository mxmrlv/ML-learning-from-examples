

"""
Tensor wrapper with extended capabilities - such as outputs and deltas.
"""

import numpy

from . import tensor


class Layer(tensor.Tensor):
    def __init__(self, rows, cols, act_func, *args, **kwargs):
        super(Layer, self).__init__(numpy.random.rand(rows, cols),
                                    *args,
                                    **kwargs)
        self._act_func = act_func
        self._outputs = None
        self._deltas = None

    def _rev_dot(self, other):
        product = other._tensor.dot(self)

        # Create output values according to the product of the table and
        # the activation functions
        for r in product:
            pass
        vec_func = numpy.vectorize(self._act_func)
        self._outputs = vec_func(product)
        return product

    @property
    def matrix(self):
        return self._tensor

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
