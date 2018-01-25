

class Matrix(object):

    def __init__(self, ndarray, *args, **kwargs):
        self.matrix = ndarray

    def __str__(self):
        return str(self.matrix)
    
    def __getitem__(self, item):
        return self.matrix[item]

    def __setitem__(self, key, value):
        self.matrix[key] = value

