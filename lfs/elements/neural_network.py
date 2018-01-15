
import numpy

import threading

class NeuralNetwork(object):

    def __init__(self, layers):
        self._layers = layers
        self._step = 0.003

    def process(self, features_vector):
        return reduce(lambda l1, l2: l1.dot(l2),
                      self._layers,
                      features_vector)

    def propagate_deltas(self, label):
        top_layer = self._layers[-1].calculate_deltas(label=label)

        return reduce(
            lambda top_layer, layer: layer.calculate_deltas(
                top_layer=top_layer,
            ),
            reversed(self._layers),
            top_layer
        )

    def update_layers(self):
        def _tensor_calculator(layer):
            layer._tensor = \
                numpy.array(
                    [v - (self._step * layer.deltas.tensor * layer.outputs.tensor)
                     for v in layer._tensor]
                )

        threads = [
            threading.Thread(target=_tensor_calculator,
                             kwargs=(dict(layer=layer)))
            for layer in self._layers
        ]

        for thread in threads:
            thread.start()
            thread.join()

    def __str__(self):
        return '\n\n'.join(str(l) for l in self._layers)