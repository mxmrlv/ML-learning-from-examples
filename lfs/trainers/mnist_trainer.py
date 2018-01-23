import os

import mnist
import numpy

from ..trainee import Trainee
from .base import TrainerBase

_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '../../samples/mnist')
)
TRAIN_IMAGES = os.path.join(_PATH, 'train-images.idx3-ubyte')
TRAIN_LABELS = os.path.join(_PATH, 'train-labels.idx1-ubyte')
TEST_IMAGES = os.path.join(_PATH, 't10k-images.idx3-ubyte')
TEST_LABELS = os.path.join(_PATH, 't10k-labels.idx1-ubyte')


class MNISTTrainer(TrainerBase):
    def __init__(self, trainee=None, *args, **kwargs):
        features_size = 28**2

        if trainee:
            assert trainee._dimensions[0] == features_size + 1
            assert len(trainee._labels) == 10
        else:
            trainee = Trainee(
                features_size, range(0, 10),
                lambda_factor=.85,
                step=.03,
                hidden_d=(10, 10)
            )

        super(MNISTTrainer, self).__init__(trainee, *args, **kwargs)
        print 'Getting Training set...'
        self._training_sets = self._get_data()
        print 'Getting Test set...'
        self._test_sets = self._get_data(training=False)

    @property
    def training_sets(self):
        return self._training_sets

    @property
    def test_sets(self):
        return self._test_sets

    @staticmethod
    def _get_data(training=True):
        def _normalize_features(features):
            features_matrix = numpy.array([f.flatten() for f in features])
            return (features_matrix - features_matrix.mean()) / 255.

        image_path = TRAIN_IMAGES if training else TEST_IMAGES
        label_path = TRAIN_LABELS if training else TEST_LABELS
        with open(image_path, 'rb') as f:
            features = _normalize_features(mnist.parse_idx(f))
        with open(label_path, 'rb') as f:
            labels = mnist.parse_idx(f)

        return numpy.array(
            [(features[i, :], labels[i]) for i in xrange(len(labels))]
        )
