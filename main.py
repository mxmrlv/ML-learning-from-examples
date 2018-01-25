from lfs import trainers
from lfs.trainee import Trainee

if __name__ == '__main__':
    t = trainers.MNISTTrainer(
        trainee=Trainee(28*28,
                        range(0, 10),
                        hidden_d=(10, 10),
                        step=.03,
                        lambda_factor=.95),
        batch_size=10
    )
    t.train(p_right=0.98)
    print '~' * 20
    t.test()
