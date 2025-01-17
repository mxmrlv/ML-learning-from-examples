import numpy


class CostFunction(object):
    @staticmethod
    def __call__(label, output_layer):
        raise NotImplementedError


class QuadraticCostFunction(object):

    @staticmethod
    def __call__(label, output_layer):
        return label.matrix - output_layer.outputs.matrix


class CrossEntropyCostFunction(object):

    @staticmethod
    def __call__(label, output_layer):
        # diff = numpy.exp(label.vector * output_layer.outputs.vector)
        # inner_der = output_layer.outputs.vector
        # return (diff * inner_der) / (1 + diff)
        return -label.matrix * numpy.log(output_layer.outputs.matrix)


class ActivationFunction(object):

    @staticmethod
    def call(x):
        raise NotImplementedError

    @staticmethod
    def der_call(o):
        raise NotImplementedError


class Sigmoid(ActivationFunction):
    @staticmethod
    def call(x):
        return 1 / (1 + numpy.exp(-x))

    @staticmethod
    def der_call(o):
        return o * (1 - o)
