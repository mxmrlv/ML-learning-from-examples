import numpy


class Matrix(object):

    def __init__(self, ndarray, *args, **kwargs):
        self.matrix = ndarray

    def __str__(self):
        return str(self.matrix)

    def __getattr__(self, item):
        try:
            return_val = getattr(self.matrix, item)
            if isinstance(return_val, numpy.ndarray):
                return Matrix(return_val)
            else:
                return return_val
        except AttributeError:
            return super(Matrix. self).__getattribute__(item)

    @property
    def T(self):
        return self.transpose()

    def __getitem__(self, item):
        return self.matrix[item]

    def __setitem__(self, key, value):
        self.matrix[key] = value

