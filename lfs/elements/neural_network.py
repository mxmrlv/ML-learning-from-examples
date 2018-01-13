
class NeuralNetwork(object):

    def __init__(self, layers):
        self._layers = layers
        self._lambda = 1

    def process(self, features_vector):
        return reduce(lambda v1, v2: v1.dot(v2), self._layers, features_vector)

    def propagate_deltas(self, label):
        self._layers[-1].deltas = -self._lambda * (label - self._layers[-1].outputs) * self._layers[-1].outputs * (1 - self._layers[-1].outputs)

        prev_deltas = self._layers[-1].deltas

        for layer in reversed(self._layers[:-1]):
            layer.deltas = [
                self._lambda *
                layer.outputs[i] *
                (1 - layer.outputs[i]) *
                layer.transpose()[i].dot(prev_deltas)

                for i in xrange(len(layer.outputs))
            ]

            prev_deltas = layer.deltas

    def __str__(self):
        return '\n\n'.join(str(l) for l in self._layers)