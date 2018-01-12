
class NeuralNetwork(object):

    def __init__(self, layers):
        self._layers = layers

    def process(self, features_row):
        return reduce(lambda v1, v2: v1.dot(v2), self._layers, features_row)

    def __str__(self):
        return '\n\n'.join(str(l) for l in self._layers)