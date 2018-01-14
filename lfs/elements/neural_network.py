
import numpy

class NeuralNetwork(object):

    def __init__(self, layers):
        self._layers = layers
        self._step = 0.003

    def process(self, features_vector):
        return reduce(lambda l1, l2: l1.dot(l2),
                      self._layers,
                      features_vector)

    def propagate_deltas(self, label):
        top_deltas = self._layers[-1].calculate_deltas(label=label)

        return reduce(
            lambda upper_deltas, layer: layer.calculate_deltas(
                upper_deltas=upper_deltas
            ),
            reversed(self._layers[:-1]),
            top_deltas
        )

    def update_layers(self):
        for layer in self._layers:
            layer._tensor = \
                numpy.array([v - (self._step * layer.deltas.tensor * layer.outputs.tensor) for v in layer._tensor])



    def __str__(self):
        return '\n\n'.join(str(l) for l in self._layers)