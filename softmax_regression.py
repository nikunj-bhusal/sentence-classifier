import numpy as np


def softmax(z):
    # Numerically stable softmax: subtract the max value of each row to prevent math overflow errors
    z -= np.max(z, axis=1, keepdims=True)
    z = np.exp(z)
    z /= np.sum(z, axis=1, keepdims=True)
    return z


class SoftmaxRegression:
    def __init__(self):
        self.num_of_class = None  # Number of categories to predict
        self.n = None  # Number of data samples
        self.m = None  # Number of features (words/n-grams)
        self.weight = None  # Model weights, shape: (num_classes, features)
        self.learning_rate = None  # How big of a step to take when learning

    def fit(
        self,
        X,
        y,
        learning_rate=0.01,
        epoch=10,
        num_of_class=5,
        print_loss_steps=-1,
        update_strategy="batch",
    ):
        self.n, self.m = X.shape
        self.num_of_class = num_of_class
        self.weight = np.random.randn(self.num_of_class, self.m)
        self.learning_rate = learning_rate

        # Convert labels (y) to one-hot encoding (e.g., class 2 becomes [0, 0, 1, 0, 0])
        y_one_hot = np.zeros((self.n, self.num_of_class))
        for i in range(self.n):
            y_one_hot[i][y[i]] = 1

        loss_history = []

        for e in range(epoch):
            loss = 0
            # Stochastic Gradient Descent: Updates weights one random sample at a time
            if update_strategy == "stochastic":
                rand_index = np.arange(len(X))
                np.random.shuffle(rand_index)
                for index in list(rand_index):
                    Xi = X[index].reshape(1, -1)
                    prob = Xi.dot(self.weight.T)
                    prob = softmax(prob).flatten()
                    loss += -np.log(prob[y[index]])
                    self.weight += (
                        Xi.reshape(1, self.m)
                        .T.dot((y_one_hot[index] - prob).reshape(1, self.num_of_class))
                        .T
                    )

            # Batch Gradient Descent: Updates weights using all samples at once
            if update_strategy == "batch":
                prob = X.dot(self.weight.T)
                prob = softmax(prob)

                for i in range(self.n):
                    loss -= np.log(prob[i][y[i]])

                weight_update = np.zeros_like(self.weight)
                for i in range(self.n):
                    weight_update += (
                        X[i]
                        .reshape(1, self.m)
                        .T.dot((y_one_hot[i] - prob[i]).reshape(1, self.num_of_class))
                        .T
                    )
                self.weight += weight_update * self.learning_rate / self.n

            loss /= self.n
            loss_history.append(loss)
            if print_loss_steps != -1 and e % print_loss_steps == 0:
                print("epoch {} loss {}".format(e, loss))
        return loss_history

    def predict(self, X):
        prob = softmax(X.dot(self.weight.T))
        return prob.argmax(axis=1)

    def score(self, X, y):
        pred = self.predict(X)
        return np.sum(pred.reshape(y.shape) == y) / y.shape[0]
