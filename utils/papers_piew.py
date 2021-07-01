import numpy as np
from sklearn.metrics import confusion_matrix
import tqdm

def mval(conf_overall, metric, classes):
    """Returns the metric value for a confusion
    Args:
    conf_overall: the overall confusion matrix of a classifier with true labels
      on rows and predictions on columns
    metric: Name of the metric
    classes: number of classes
    Returns:
    Metric value for the given confusion
    """

    # Computing G-mean
    if(isinstance(metric, str)):
        if (metric == 'G-mean'):
            rate_overall = conf_overall.copy()
            row_sums = rate_overall.sum(axis=1)
            rate_overall = rate_overall / row_sums[:, np.newaxis]
            # return 1 - np.sqrt(rate_overall.diagonal().prod())
            return np.sqrt(rate_overall.diagonal().prod())

        # Computing F-measure
        elif (metric == 'F-measure'):
            row_sums = conf_overall.sum(axis=1)
            col_sums = conf_overall.sum(axis=0)
            denom = row_sums + col_sums
            return ((2 * conf_overall) / denom[:, np.newaxis]).trace() / classes

        # Computing Accuracy
        else:
            assert (metric == "Accuracy"), "metric can be G-mean, F-measure, or Accuracy"
            return conf_overall.trace()

    # Making dot product of true grad
    else:
        return np.dot(conf_overall.diagonal(), metric)

def plugin(metric, y_val, y_test, pred_eta_model, classes, step=0.01):
    """
    Post shifts the provided estimate of conditional probability of the positive class
    using a threhsold learned on the validationd data.
    """

    # obtain prediction probabilties on validation data
    pred_prob_val = pred_eta_model['val_full'].copy()
    label_val = y_val

    # obtain prediction probabilties on test data
    pred_prob_test = pred_eta_model['test'].copy()
    label_test = y_test

    weight_ratio_list = []
    mval_val_max_list = []

    total_iterations = int(classes * (classes - 1) * (1.0-0.04)/step)
    pbar = tqdm.tqdm(range(total_iterations))
    
    for c_max in range(classes):

        weight_ratio = np.ones(classes)

        # loop to brute-force search the threshold which minimizes metric on val data
        # this loop finds optimal threhsolds based on CLASSES one-vs-all binary problems
        for c in list(range(c_max)) + list(range(c_max + 1, classes)):

            mval_val_max = np.float('-inf')
            t_max = -1.0

            for t in np.arange(0.02, 1 - 0.02, step):

                weights = np.zeros((y_val.shape[0], classes))

                weights[:, c_max] = t
                weights[:, c] = 1 - t

                pred_val = np.argmax(weights * pred_prob_val, axis=1)

                conf_val = confusion_matrix(label_val, pred_val, normalize='all')

                mval_t = mval(conf_val, metric, classes)

                if (mval_t > mval_val_max):
                    mval_val_max = mval_t
                    t_max = t
                pbar.update(1)

            weight_ratio[c] = (1 - t_max) / t_max
        
        weight_ratio_list.append(weight_ratio)
        mval_val_max_list.append(mval_val_max)

    pbar.close()
        
    # weight_ratio_indi = sorted( range(len(mval_val_max_list)), key = lambda k: -mval_val_max_list[k])[0]
    # # weights are normalized
    # final_weights = weight_ratio_list[weight_ratio_indi] / np.sum(weight_ratio_list[weight_ratio_indi])
    final_weights = weight_ratio_list[np.argmax(mval_val_max_list)]
    final_weights = final_weights / final_weights.sum()

    final_weights_val = np.tile(final_weights, (y_val.shape[0], 1))
    final_weights_test = np.tile(final_weights, (y_test.shape[0], 1))

    # predictions on val and test data
    pred_val = np.argmax(final_weights_val * pred_prob_val, axis=1)
    pred_test = np.argmax(final_weights_test * pred_prob_test, axis=1)

    # confusions on val and test data
    conf_val = confusion_matrix(label_val, pred_val, normalize='all')
    conf_test = confusion_matrix(label_test, pred_test, normalize='all')

    # metric values on val and test data
    mval_val = mval(conf_val, metric, classes)
    mval_test = mval(conf_test, metric, classes)

    return mval_val, mval_test, final_weights
