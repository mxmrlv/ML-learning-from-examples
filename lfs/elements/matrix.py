

class Matrix(object):

    def __init__(self, ndarray, *args, **kwargs):
        self.matrix = ndarray

    def __str__(self):
        return str(self.matrix)

    def __getitem__(self, item):
        return self.matrix[item]

    def __setitem__(self, key, value):
        self.matrix[key] = value


class Vector(Matrix):
    def __init__(self, ndarray, *args, **kwargs):
        assert len(ndarray.shape) == 1
        super(Vector, self).__init__(ndarray, *args, **kwargs)

    @property
    def vector(self):
        return self.matrix

    def argmax(self):
        return self.vector.argmax()

    def max(self):
        return self.vector.max()
