import numpy as np
from scipy.sparse import issparse, hstack, csr_matrix


def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def compute_class_weights(y_flat, num_classes):
    """
    Inverse-frequency weights so rare classes get proportionally larger gradients.
    Formula: w_c = total / (num_classes * count_c)
    """
    weights = np.zeros(num_classes)
    total = len(y_flat)
    for c in range(num_classes):
        count = np.sum(y_flat == c)
        if count > 0:
            weights[c] = total / (num_classes * count)
    return weights


def _row_slice(X, indices):
    """Slice rows from either a sparse or dense matrix."""
    if issparse(X):
        return X[indices]
    return X[indices]


def _dot(X, W_T):
    """
    X @ W.T — works for both dense numpy arrays and scipy sparse matrices.
    Returns a dense numpy array (softmax always needs dense output).
    """
    if issparse(X):
        return np.asarray(X.dot(W_T))
    return X.dot(W_T)


def _add_bias(X):
    """
    Prepend a column of ones (bias trick) for both sparse and dense inputs.
    For sparse matrices this avoids converting the whole matrix to dense.
    """
    if issparse(X):
        n = X.shape[0]
        bias_col = csr_matrix(np.ones((n, 1), dtype=np.float32))
        return hstack([bias_col, X], format="csr")
    return np.c_[np.ones((X.shape[0], 1), dtype=np.float32), X]


class SoftmaxRegression:
    def __init__(self):
        self.num_of_class = None
        self.weight = None

    def fit(
        self,
        X_raw,
        y,
        learning_rate=0.1,
        epoch=30,
        num_of_class=3,
        print_loss_steps=5,
        update_strategy="mini_batch",
        lambda_reg=0.001,
        batch_size=256,
        lr_decay=0.98,
        use_class_weights=True,
    ):
        X = _add_bias(X_raw)
        self.n = X.shape[0]
        self.m = X.shape[1]
        self.num_of_class = num_of_class

        y_flat = np.array(y).flatten().astype(int)

        if use_class_weights:
            class_weights = compute_class_weights(y_flat, num_of_class)
            print(
                f"  Class weights: { {i: round(w, 2) for i, w in enumerate(class_weights)} }"
            )
        else:
            class_weights = np.ones(num_of_class)

        self.weight = (
            np.random.randn(self.num_of_class, self.m).astype(np.float32) * 0.01
        )

        y_one_hot = np.zeros((self.n, self.num_of_class), dtype=np.float32)
        y_one_hot[np.arange(self.n), y_flat] = 1.0

        current_lr = float(learning_rate)
        loss_history = []

        for e in range(epoch):
            rand_index = np.random.permutation(self.n)
            epoch_loss = 0.0

            if update_strategy == "stochastic":
                for index in rand_index:
                    Xi = _row_slice(X, [index])
                    prob = softmax(_dot(Xi, self.weight.T)).flatten()
                    c = y_flat[index]
                    epoch_loss += class_weights[c] * -np.log(prob[c] + 1e-15)

                    grad = class_weights[c] * (y_one_hot[index] - prob)

                    if issparse(Xi):
                        weight_update = np.outer(grad, Xi.toarray().flatten())
                    else:
                        weight_update = np.outer(grad, Xi.flatten())
                    self.weight += current_lr * weight_update
                    self.weight[:, 1:] *= 1.0 - current_lr * lambda_reg

            elif update_strategy == "mini_batch":
                for start in range(0, self.n, batch_size):
                    idx = rand_index[start : start + batch_size]
                    Xi = _row_slice(X, idx)
                    yi = y_flat[idx]
                    yi_oh = y_one_hot[idx]

                    prob = softmax(_dot(Xi, self.weight.T))

                    sample_weights = class_weights[yi]
                    correct_probs = prob[np.arange(len(idx)), yi]
                    epoch_loss += float(
                        np.sum(sample_weights * -np.log(correct_probs + 1e-15))
                    )

                    grad = (yi_oh - prob) * sample_weights[:, None]
                    grad /= len(idx)

                    if issparse(Xi):
                        weight_update = np.asarray(grad.T.dot(Xi.toarray()))
                    else:
                        weight_update = grad.T.dot(Xi)

                    self.weight += current_lr * weight_update
                    self.weight[:, 1:] *= 1.0 - current_lr * lambda_reg

            current_lr *= lr_decay

            l2_penalty = (lambda_reg / 2.0) * float(np.sum(self.weight[:, 1:] ** 2))
            loss = float(epoch_loss / self.n) + l2_penalty
            loss_history.append(loss)

            if print_loss_steps > 0 and (e % print_loss_steps == 0 or e == epoch - 1):
                print(f"  epoch {e:3d} | loss {loss:.4f} | lr {current_lr:.5f}")

        return loss_history

    def predict(self, X_raw):
        X = _add_bias(X_raw)
        prob = softmax(_dot(X, self.weight.T))
        return prob.argmax(axis=1)

    def predict_proba(self, X_raw):
        X = _add_bias(X_raw)
        return softmax(_dot(X, self.weight.T))

    def score(self, X, y):
        pred = self.predict(X)
        y_flat = np.array(y).flatten()
        return float(np.sum(pred == y_flat) / len(y_flat))
