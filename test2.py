import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
from chainer import Chain, optimizers, Variable, training
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions


class Config:
    nn_input_dim = 2  # input layer dimensionality
    nn_output_dim = 2  # output layer dimensionality
    num_examples = 200
    nn_hdim = 3
    # Gradient descent parameters (I picked these by hand)
    epsilon = 0.01  # learning rate for gradient descent
    reg_lambda = 0.01  # regularization strength


def generate_data():
    np.random.seed(0)
    X, y = datasets.make_moons(Config.num_examples, noise=0.20)
    return X, y


def visualize(X, y, model):
    # plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    # plt.show()
    plot_decision_boundary(lambda x:predict(model,x), X, y)
    plt.title("Logistic Regression")


def plot_decision_boundary(pred_func, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()


def predict(model, x_data):
    x = Variable(x_data.astype(np.float32))
    y = model.predictor(x)
    return np.argmax(y.data, axis=1)


def train(X, y, model, optimizer):
    x = Variable(X.astype(np.float32))
    t = Variable(y.astype(np.int32))

    for i in range(20000):
        optimizer.update(model, x, t)
        if i % 1000 == 0:
            print("Finished %i" % i)


class MakeMoonModel(Chain):
    def __init__(self):
        super(MakeMoonModel, self).__init__(
            l1=L.Linear(Config.nn_input_dim, Config.nn_hdim),
            l2=L.Linear(Config.nn_hdim, Config.nn_output_dim),
        )

    def __call__(self, x):
        h = F.tanh(self.l1(x))
        return self.l2(h)


def main():
    X, y = generate_data()
    model = L.Classifier(MakeMoonModel())
    optimizer = optimizers.Adam()
    optimizer.setup(model)
    updater = training.StandardUpdater(X, )
    train(X, y, model, optimizer)
    visualize(X, y, model)


if __name__ == "__main__":
    main()