from copy import deepcopy

import numpy as np
import pandas as pd

# metric must be this or a BlackBoxMetric
SUPPORTED_METRICS = ("Accuracy", "G-mean", "F-measure")


class BlackBoxMetric:
    def __init__(self):
        pass
    
    def eval_metric(self, preds: np.array, y: np.array, nc: int):
        raise NotImplementedError()

    def __str__(self):
        raise NotImplementedError()

    def __repr__(self):
        raise NotImplementedError()


class AdultBBMetric(BlackBoxMetric):
    def __init__(self, male_y: np.array, female_y: np.array):
        """
        This will be G-mean of male and female confusions combined.
        Specifically:
        (TP_m * TN_m * TP_f * TN_f)^0.25
        """
        super().__init__()
        self.male_y = male_y.reshape(-1,1)
        self.female_y = female_y.reshape(-1, 1)

    def eval_metric(self, preds: np.array, y: np.array, nc: int):
        assert len(preds) == len(self.male_y)
        y_one_hot = np.zeros((y.size, nc))
        y_one_hot[np.arange(y.size), y] = 1

        male_and_labels = self.male_y * y_one_hot
        denom = np.sum(self.male_y)
        male_conf = np.matmul(np.transpose(male_and_labels), preds) / denom

        female_and_labels = self.female_y * y_one_hot
        denom = np.sum(self.female_y)
        female_conf = np.matmul(np.transpose(female_and_labels), preds) / denom

        male_score = eval_metric(male_conf, "G-mean")
        female_score = eval_metric(female_conf, "G-mean")
        # this undoes the square root performed by G-mean, then multiplies
        # the male and female scores together. Then, it computes the 4th root
        return ((male_score ** 2) * (female_score ** 2))**(1.0 / 4)

    def __str__(self):
        return "male_female_gmean_blackbox_metric"

    def __repr__(self):
        return "male_female_gmean_blackbox_metric"

def eval_metric(conf_matrix: np.array, metric: str) -> float:
    """
    Args:
        conf_matrix: the overall confusion matrix of a classifier with true labels
          as rows and predictions as columns. Use sklearn.metrics.confusion_matrix
          to create this, and make sure to set the flag ` normalize='all' `
        metric: name of the metric, must be one of ['Accuracy', 'G-mean', 'F-measure']

    Returns:
        The metric value
    """
    assert metric in ["Accuracy", "G-mean", "F-measure"]
    classes = len(conf_matrix)

    if metric == "Accuracy":
        return conf_matrix.trace()

    elif metric == "G-mean":
        row_sums = conf_matrix.sum(axis=1)
        # this will divide each row of conf_matrix by the corresponding row_sum
        conf_matrix = conf_matrix / row_sums[:, np.newaxis]
        return np.power(conf_matrix.diagonal().prod(), 1.0 / classes)

    elif metric == "F-measure":
        row_sums = conf_matrix.sum(axis=1)
        col_sums = conf_matrix.sum(axis=0)
        denom = row_sums + col_sums
        return ((2 * conf_matrix) / denom[:, np.newaxis]).trace() / classes

    else:
        raise ValueError(f"Unrecognized metric {metric}")


def eval_true_metric(conf_matrix: np.array, true_grad: np.array) -> float:
    """
    Args:
        conf_matrix: the overall confusion matrix of a classifier with true labels
          as rows and predictions as columns. Use sklearn.metrics.confusion_matrix
          to create this, and make sure to set the flag ` normalize='all' `
       true_grad: the diagonal weights of the true metric on the validation set

    Returns:
        The metric value
    """
    return np.dot(conf_matrix.diagonal(), true_grad)