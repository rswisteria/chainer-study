import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from chainer import cuda, Variable, Chain, optimizers
import chainer.functions as F
import chainer.links as L


def train(X, y, model, optimizer):
    x = Variable(X.astype(np.float32))
    t = Variable(y.astype(np.int32))

    for i in range(20000):
        optimizer.update(model, x, t)
        if i % 1000 == 0:
            print("Finished %i" % i)


def main():
    iris = load_iris()
    model = L.Classifier(IrisModel())
    optimizer = optimizers.Adam()
    optimizer.setup(model)
    train(iris.data, iris.target, model, optimizer)

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
