import os
import numpy
from random import shuffle

import mnist

from learner import Learner
from lfs.elements import functions

numpy.set_printoptions(precision=2)

_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '../samples/mnist')
)
TRAIN_IMAGES = os.path.join(_PATH, 'train-images.idx3-ubyte')
TRAIN_LABELS = os.path.join(_PATH, 'train-labels.idx1-ubyte')
TEST_IMAGES = os.path.join(_PATH, 't10k-images.idx3-ubyte')
TEST_LABELS = os.path.join(_PATH, 't10k-labels.idx1-ubyte')


def _process_features(features):
    features_matrix = numpy.array([f.flatten() for f in features])
    return (features_matrix - features_matrix.mean()) / 255.


def _get_data(training=True):
    image_path = TRAIN_IMAGES if training else TEST_IMAGES
    label_path = TRAIN_LABELS if training else TEST_LABELS
    with open(image_path, 'rb') as f:
        features = _process_features(mnist.parse_idx(f))
    with open(label_path, 'rb') as f:
        labels = mnist.parse_idx(f)

    return [(features[i, :], labels[i]) for i in xrange(len(labels))]


def _train(learner, p_right=0.9):
    features_mapping = _get_data()
    epochs = 0
    current_p_right = 0
    while p_right > current_p_right:
        with learner.stats as stats:
            epochs += 1
            shuffle(features_mapping)
            for features, label in features_mapping:
                learner.learn(features, label)
                print stats.progress(len(features_mapping)),
            print 'Number of epochs: {0}'.format(epochs)
            current_p_right = stats.p_right


def _asses(learner):
    features = _get_data(training=False)
    with learner.stats as stats:
        for (features, label) in features:
            learner.test(features, label)
            print stats.progress(len(features)),


if __name__ == '__main__':
    l = Learner(features_d=28*28,
                labels=range(0, 10),
                step=.03,
                cost_func=functions.CrossEntropyCostFunction,
                hidden_d=(10, ) * 3)
    _train(l)

    print '~' * 20

    _asses(l)
