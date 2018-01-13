

"""
The most basic tensor wrapper
"""


class Tensor(object):

    def __init__(self, tensor, *args, **kwargs):
        self._tensor = tensor

    def __getattr__(self, item):
        try:
            return getattr(self._tensor, item)
        except AttributeError:
            return super(Tensor, self).__getattribute__(item)

    def __str__(self):
        return str(self._tensor)

    def __getitem__(self, item):
        return self._tensor[item]

    def __setitem__(self, key, value):
        self._tensor[key] = value

    def dot(self, other):
        return other.dot(self)

    @property
    def tensor(self):
        return self._tensor
