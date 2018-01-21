import os
from random import shuffle

import mnist
import numpy

from .learner import Learner

_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '../samples/mnist')
)
TRAIN_IMAGES = os.path.join(_PATH, 'train-images.idx3-ubyte')
TRAIN_LABELS = os.path.join(_PATH, 'train-labels.idx1-ubyte')
TEST_IMAGES = os.path.join(_PATH, 't10k-images.idx3-ubyte')
TEST_LABELS = os.path.join(_PATH, 't10k-labels.idx1-ubyte')


class TrainerBase(object):
    def __init__(self, learner):
        self._learner = learner

    @property
    def training_sets(self):
        raise NotImplementedError

    @property
    def test_sets(self):
        raise NotImplementedError

    def train(self, p_right=0.9):
        epochs = 0
        current_p_right = 0
        while p_right > current_p_right:
            with self._learner.stats as stats:
                epochs += 1
                shuffle(self.training_sets)
                for features, label in self.training_sets:
                    self._learner.learn(features, label)
                    print stats.progress(len(self.training_sets)),
                print 'Number of epochs: {0}'.format(epochs)
            print 'Assessing with the test set...'
            current_p_right = self.asses(print_state=False).p_right

    def asses(self, print_state=True):
        with self._learner.stats as stats:
            for (features, label) in self.test_sets:
                self._learner.test(features, label)
                if print_state:
                    print stats.progress(len(self.test_sets)),
            return stats


class MNISTTrainer(TrainerBase):
    def __init__(self, learner=None, *args, **kwargs):
        learner = learner or \
                  Learner(28*28, range(0, 10), step=.3, hidden_d=(10, ) * 2)
        super(MNISTTrainer, self).__init__(learner, *args, **kwargs)
        self._training_sets = self._get_data()
        self._test_sets = self._get_data(training=False)

    @property
    def training_sets(self):
        return self._training_sets

    @property
    def test_sets(self):
        return self._test_sets

    @staticmethod
    def _get_data(training=True):
        def _process_features(features):
            features_matrix = numpy.array([f.flatten() for f in features])
            return (features_matrix - features_matrix.mean()) / 255.

        image_path = TRAIN_IMAGES if training else TEST_IMAGES
        label_path = TRAIN_LABELS if training else TEST_LABELS
        with open(image_path, 'rb') as f:
            features = _process_features(mnist.parse_idx(f))
        with open(label_path, 'rb') as f:
            labels = mnist.parse_idx(f)

        return [(features[i, :], labels[i]) for i in xrange(len(labels))]
