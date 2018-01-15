

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
        self._lambda = 1

    def dot(self, other):
        product = self.tensor.transpose().dot(other.tensor)
        vec_func = numpy.vectorize(self._act_func)
        self._outputs = tensor.Tensor(vec_func(product))
        return self.outputs

    def calculate_deltas(self, label=None, top_layer=None):
        if label:
            deltas = (
                    -self._lambda *
                    (label.tensor - self.outputs.tensor) *
                    self.outputs.tensor *
                    (1 - self.outputs.tensor)
            )
        else:
            assert top_layer is not None
            deltas = [
                self._lambda *
                self.outputs[i] *
                (1 - self.outputs[i]) *
                top_layer.tensor[i].dot(top_layer.deltas.tensor)

                for i in xrange(len(self.outputs.tensor))
            ]

        self._deltas = tensor.Tensor(numpy.array(deltas))
        return self

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
