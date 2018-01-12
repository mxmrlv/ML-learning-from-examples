"""
Basic Tensor extension which creates an array of features
"""

import numpy

from . import tensor


class Features(tensor.Tensor):
    def __init__(self, features, *args, **kwargs):
        super(Features, self).__init__(numpy.array(features), *args, **kwargs)
