from chainer import Chain, optimizers, datasets, iterators, training
from chainer import functions as F
from chainer import links as L
from chainer.training import extensions


class Config:
    batchsize = 100
    epoch = 20


class MnistCNN(Chain):
    def __init__(self, channel=1, c1=16, c2=32, c3=64, f1=256, f2=512, filter_size1=3, filter_size2=3, filter_size3=3):
        super(MnistCNN, self).__init__(
            conv1=L.Convolution2D(channel, c1, filter_size1),
            conv2=L.Convolution2D(c1, c2, filter_size2),
            conv3=L.Convolution2D(c2, c3, filter_size3),
            l1=L.Linear(f1, f2),
            l2=L.Linear(f2, 10)
        )


    def __call__(self, x):
        x = x.reshape(len(x), 1, 28, 28)
        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(h, 2)
        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(h, 2)
        h = F.relu(self.conv3(h))
        h = F.max_pooling_2d(h, 2)
        h = F.dropout(F.relu(self.l1(h)))
        y = self.l2(h)

        return y


def main():
    model = L.Classifier(MnistCNN())
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    train, test = datasets.get_mnist()
    train_iter = iterators.SerialIterator(train, Config.batchsize)
    test_iter = iterators.SerialIterator(test, Config.batchsize, repeat=False, shuffle=False)

    updater = training.StandardUpdater(
        train_iter, optimizer
    )

    trainer = training.Trainer(updater, (Config.epoch, 'epoch'))
    trainer.extend(extensions.Evaluator(test_iter, model))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], 'epoch', file_name='loss.png'))
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.ProgressBar())

    trainer.run()


if __name__ == '__main__':
     main()