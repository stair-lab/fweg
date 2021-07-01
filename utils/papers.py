"""
This file contains functions copied or adapted from previous work so
that we can baseline our method with existing techniques.
"""


import numpy as np
import tensorflow as tf
import keras.backend as K
import keras.losses as losses
import sklearn.metrics
from cvxopt import matrix, solvers


def forward_correct_est_T(model, x_train, nc, batch_size):
    """
    `x_train`: the train dataset, either a numpy array or `tf.data.Dataset`
    Estimates T using forward correction. T is a [num_class, num_class] numpy matrix.
    T[i,j] means the probabilty that class i is flipped to class j.

    Paper: https://arxiv.org/pdf/1609.03683.pdf
    """
    T_hat = np.empty((nc, nc))

    # predict probability on the train
    eta_corr = model.predict(x_train, batch_size=batch_size)

    # find a 'perfect example' for each class
    for i in np.arange(nc):
        # ignore the top 3% of probability predictions of this class
        # this is helpful against overfitting where the model is overly
        # confident about the classs that a sample belongs to
        eta_thresh = np.percentile(eta_corr[:, i], 97, interpolation="higher")
        robust_eta = eta_corr[:, i]
        robust_eta[robust_eta >= eta_thresh] = 0.0
        idx_best = np.argmax(robust_eta)
        for j in np.arange(nc):
            T_hat[i, j] = eta_corr[idx_best, j]

    # row normalize T_hat
    row_sums = T_hat.sum(axis=1)
    T_hat /= row_sums[:, np.newaxis]

    return T_hat


def make_forward_correct_loss(T_hat):
    """
    Returns the forward correction loss.

    Paper: https://arxiv.org/pdf/1609.03683.pdf
    """
    P = K.constant(T_hat)

    def loss(y_true, y_pred):
        # y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
        return losses.categorical_crossentropy(y_true, K.dot(y_pred, P))

    return loss


def train_step(model, x, y, x_valid, y_valid, loss_f, optimizer):
    """
    `model`: keras model
    `x`: x batch
    `y`: y labels for batch
    `x_valid`: x_valid batch
    `y_valid`: y labels for valid batch
    `loss_f`: loss function, must be initialized with reduction=tf.keras.losses.Reduction.NONE
    `optimizer`: the optimizer to use

    Train step of "learning to reweight". This essentially implements an eager version
    of https://github.com/uber-research/learning-to-reweight-examples/blob/0b616c99ecf8a1c99925322167694272a966ed00/models/reweight_model.py#L17.
    Follow the non-legacy code path and it will closely resemble below.

    Paper: https://arxiv.org/pdf/1803.09050.pdf
    """
    # do a forward pass and compute loss using epsilon = 0
    epsilon = tf.zeros([len(x), 1], dtype=tf.float32)
    # tape_epsilon is persistent because we make multiple calls to compute gradient
    with tf.GradientTape(persistent=True) as tape_epsilon:
        tape_epsilon.watch(epsilon)

        # tape_preds is not persistent because we only make one gradient call
        with tf.GradientTape(persistent=False) as tape_preds:
            preds = model(x)
            # compute loss using epsilon
            loss = tf.math.reduce_sum(loss_f(y, preds) * epsilon) / len(x)

        # compute model gradients
        grads_train = tape_preds.gradient(loss, model.trainable_variables)

        # compute loss on valid set
        preds_valid = model(x_valid)
        loss_valid = tf.math.reduce_sum(loss_f(y_valid, preds_valid)) / len(x)

    # grad of validation wrt model variables
    grads_valid = tape_epsilon.gradient(loss_valid, model.trainable_variables)
    # this is a trick to computing the impact of epsilon, adapated from original paper Github
    # we can compute the grad of epsilon on grads_train, and use
    # output_gradients=grads_valid to account for the impact of epsilon on the validation loss
    grads_epsilon = tape_epsilon.gradient(
        grads_train, epsilon, output_gradients=grads_valid
    )

    # normalization of grads, as discussed in paper
    ex_weight_plus = tf.maximum(grads_epsilon, 0.0)
    ex_weight_sum = tf.reduce_sum(ex_weight_plus)
    ex_weight_sum += tf.cast(tf.equal(ex_weight_sum, 0.0), dtype=tf.float32)
    # sample weights
    ex_weight_norm = ex_weight_plus / ex_weight_sum

    # do a normal forward pass using the meta-learned sample weights
    with tf.GradientTape(persistent=False) as tape:
        preds = model(x)
        loss = tf.math.reduce_sum(loss_f(y, preds) * ex_weight_norm) / len(x)

    # compute gradient and apply it with optimizer
    grads = tape.gradient(loss, model.trainable_variables)
    grads_and_vars = [
        (grad, var) for grad, var in zip(grads, model.trainable_variables)
    ]
    optimizer.apply_gradients(grads_and_vars)

    return loss


def kernel(ker, X1, X2, gamma):
    """
    Compute the kernel of type `ker` between two samples `X1` and `X2`

    See https://github.com/omidbazgirTTU/KMM/blob/master/KMM.ipynb for original source
    """
    K = None
    if ker == "linear":
        if X2 is not None:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1), np.asarray(X2))
        else:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1))
    elif ker == "rbf":
        if X2 is not None:
            K = sklearn.metrics.pairwise.rbf_kernel(
                np.asarray(X1), np.asarray(X2), gamma
            )
        else:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1), None, gamma)
    else:
        raise ValueError(f"unrecognized kernel {ker}")

    return K


class KMM:
    """
    Kernel Mean Matching class

    See https://github.com/omidbazgirTTU/KMM/blob/master/KMM.ipynb for original source

    Paper: https://papers.nips.cc/paper/2006/file/a2186aa7c086b46ad4e8bf81e2a3a19b-Paper.pdf
    """

    def __init__(self, kernel_type="linear", gamma=1.0, B=1.0, eps=None):
        """
        Initialization function
        :param kernel_type: 'linear' | 'rbf'
        :param gamma: kernel bandwidth for rbf kernel
        :param B: bound for beta
        :param eps: bound for sigma_beta
        """
        self.kernel_type = kernel_type
        self.gamma = gamma
        self.B = B
        self.eps = eps

    def fit(self, Xs, Xt):
        """
        Fit source and target using KMM (compute the coefficients)
        :param Xs: ns * dim
        :param Xt: nt * dim
        :return: Coefficients (Pt / Ps) value vector (Beta in the paper)
        """
        ns = Xs.shape[0]
        nt = Xt.shape[0]
        if self.eps == None:
            self.eps = self.B / np.sqrt(ns)
        K = kernel(self.kernel_type, Xs, None, self.gamma)
        kappa = np.sum(
            kernel(self.kernel_type, Xs, Xt, self.gamma) * float(ns) / float(nt), axis=1
        )

        # cvxopt has a bug with using numpy directly
        # fix: https://stackoverflow.com/questions/12551009/python3-conversion-between-cvxopt-matrix-and-numpy-array
        K = matrix(np.array(K).T.tolist())
        kappa = matrix(np.array(kappa.T).tolist())
        G = matrix(np.r_[np.ones((1, ns)), -np.ones((1, ns)), np.eye(ns), -np.eye(ns)])
        h = matrix(
            np.r_[
                ns * (1 + self.eps),
                ns * (self.eps - 1),
                self.B * np.ones((ns,)),
                np.zeros((ns,)),
            ]
        )

        self.sol = solvers.qp(K, -kappa, G, h)
        self.beta = np.array(self.sol["x"])
        return self.beta
