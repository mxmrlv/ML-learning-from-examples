from random import shuffle


class TrainerBase(object):
    def __init__(self, learner):
        self._learner = learner

    @property
    def training_sets(self):
        raise NotImplementedError

    @property
    def test_sets(self):
        raise NotImplementedError

    def train(self, p_right=0.95):
        epochs = 0
        current_p_right = 0
        while p_right > current_p_right:
            with self._learner.stats as stats:
                epochs += 1
                shuffle(self.training_sets)
                for features_sets, label_sets in self.training_sets:
                    self._learner.train(features_sets, label_sets)
                    print stats.progress(len(self.training_sets)),
                print 'Epoch: {0}'.format(epochs)

            print 'Assessing with the test set...',
            current_p_right = self.test(print_state=False).p_right
            print 'Success rate is at: {0:.2f}%'.format(current_p_right * 100)

    def test(self, print_state=True):
        with self._learner.stats as stats:
            for (features, label) in self.test_sets:
                self._learner.test(features, label)
                if print_state:
                    print stats.progress(len(self.test_sets)),
            return stats
