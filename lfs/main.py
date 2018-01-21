from lfs import (
    learner,
    trainer
)


if __name__ == '__main__':
    t = trainer.MNISTTrainer(
        learner.Learner(28 * 28,
                        range(0, 10),
                        step=.3,
                        hidden_d=(10,) * 2)
    )
    t.train()
    print '~' * 20
    t.asses()
