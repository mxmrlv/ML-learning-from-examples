import os
import numpy
import cPickle

from ..trainee import Trainee
from .base import TrainerBase

_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '../../samples/cifar-10')
)
TRAIN_BATCHES = [
    os.path.join(_PATH, 'data_batch_{0}'.format(i))
    for i in xrange(1, 6)
]
TEST_BATCH = os.path.join(_PATH, 'test_batch')


class CIFAR10TTrainer(TrainerBase):
    def __init__(self, learner=None, *args, **kwargs):
        features_size = 3072
        if learner:
            assert learner._dimensions[0] == features_size + 1
            assert len(learner._labels) == 10
        else:
            learner = Trainee(
                features_size, range(0, 10), step=.03 , hidden_d=(20, 15, 10))
        super(CIFAR10TTrainer, self).__init__(learner, *args, **kwargs)
        self._training_sets = self._get_training_data()
        self._test_sets = self._get_test_data()

    @property
    def training_sets(self):
        return self._training_sets

    @property
    def test_sets(self):
        return self._test_sets

    @staticmethod
    def _normalize_features(features):
        return (features - features.mean()) / 255.

    def _get_training_data(self):
        batches = None
        for batch_path in TRAIN_BATCHES:
            with open(batch_path, 'rb') as f:
                batch = cPickle.load(f)
                features = self._normalize_features(batch['data'])
                labels = batch['labels']

                processed_batches = numpy.array([
                    (features[i, :], labels[i])
                    for i in xrange(len(labels))
                ])

                if batches is None:
                    batches = processed_batches
                else:
                    batches = numpy.vstack((batches, processed_batches))
        return batches

    def _get_test_data(self):
        with open(TEST_BATCH, 'rb') as f:
            batches = cPickle.load(f)
            features = self._normalize_features(batches['data'])
            labels = batches['labels']

        return numpy.array([
            (features[i, :], labels[i]) for i in xrange(len(labels))
        ])