import os
import numpy

import mnist

from learner import Learner

numpy.set_printoptions(precision=2)

_PATH = '/home/maxim-pcu/uni/ML-learning-from-samples/samples/mnist'
TRAIN_IMAGES = os.path.join(_PATH, 'train-images.idx3-ubyte')
TRAIN_LABELS = os.path.join(_PATH, 'train-labels.idx1-ubyte')
TEST_IMAGES = os.path.join(_PATH, 't10k-images.idx3-ubyte')
TEST_LABELS = os.path.join(_PATH, 't10k-labels.idx1-ubyte')


def _get_data(training=True):
    image_path = TRAIN_IMAGES if training else TEST_IMAGES
    label_path = TRAIN_LABELS if training else TEST_LABELS
    with open(image_path, 'rb') as f:
        features = mnist.parse_idx(f) / 255.
    with open(label_path, 'rb') as f:
        labels = mnist.parse_idx(f)

    return [(features[i], labels[i]) for i in xrange(len(labels))]


def _train(learner):
    from random import shuffle
    features_list = _get_data()
    for _ in xrange(4):
        with learner.stats as stats:
            shuffle(features_list)
            for features, label in features_list:
                learner.learn(features, label)
                print stats.progress(len(features_list)),
            print


def _asses(learner):
    features_list = _get_data(training=False)
    with learner.stats as stats:
        for (features, label) in features_list:
            learner.test(features, label)
            print stats.progress(len(features_list)),


if __name__ == '__main__':
    l = Learner(features_d=28*28,
                labels=range(0, 10),
                step=0.003)
                # hidden_d=(4, 4))
    _train(l)

    print '~' * 20

    _asses(l)
