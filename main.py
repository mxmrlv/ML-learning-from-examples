from lfs import trainers


if __name__ == '__main__':
    t = trainers.MNISTTrainer()
    t.train(p_right=0.98)
    print '~' * 20
    t.test()
