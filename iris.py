import numpy as np
import chainer.functions as F
import chainer.links as L
from sklearn.datasets import load_iris
from chainer import cuda, Variable, Chain, optimizers, iterators, training
from chainer.datasets import tuple_dataset
from chainer.training import extensions


def main():
    iris = load_iris()
    model = L.Classifier(IrisModel())
    optimizer = optimizers.Adam()
    optimizer.setup(model)
    train_data = tuple_dataset.TupleDataset(iris.data.astype(np.float32), iris.target.astype(np.int32))
    train_iter = iterators.SerialIterator(train_data, batch_size=50)
    updater = training.StandardUpdater(train_iter, optimizer)
    trainer = training.Trainer(updater, (10000, 'epoch'), out='result')
    trainer.extend(extensions.ProgressBar())
    trainer.run()

    X = np.array([
        [5.4, 3.6, 1.4, 0.3],
        [5.4, 2.6, 4.0, 1.4],
        [6.8, 3.2, 5.5, 2.1]
    ])
    y = model.predictor(Variable(X.astype(np.float32)))
    print(y)


class IrisModel(Chain):
    def __init__(self):
        super(IrisModel, self).__init__(
            l1 = L.Linear(4, 100),
            l2 = L.Linear(100, 100),
            l3 = L.Linear(100, 3)
        )

    def __call__(self, x):
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        return self.l3(h)


if __name__ == '__main__':
    main()
