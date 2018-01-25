from random import shuffle


class TrainerBase(object):
    def __init__(self, trainee, batch_size=1):
        self._trainee = trainee
        self._batch_size = batch_size

    @property
    def training_sets(self):
        raise NotImplementedError

    @property
    def test_sets(self):
        raise NotImplementedError

    @property
    def _batches(self):
        for item in self.training_sets.reshape(
                self.training_sets.shape[0] // self._batch_size,
                -1,
                self.training_sets.shape[1]
        ):
            yield item[:, 0], item[:, 1]

    def train(self, p_right=0.95):
        epochs = 0
        current_p_right = 0
        while p_right > current_p_right:
            with self._trainee.stats as stats:
                epochs += 1
                shuffle(self.training_sets)
                for features_sets, label_sets in self._batches:
                    self._trainee.train(features_sets, label_sets)
                    print stats.progress(len(self.training_sets)),
                print 'Epoch: {0}'.format(epochs)

            print 'Assessing with the test set...',
            current_p_right = self.test(print_state=False).p_right
            print 'Success rate is at: {0:.2f}%'.format(current_p_right * 100)

    def test(self, print_state=True):
        with self._trainee.stats as stats:
            for (features, label) in self.test_sets:
                self._trainee.test(features, label)
                if print_state:
                    print stats.progress(len(self.test_sets)),
            return stats
