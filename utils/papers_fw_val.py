import numpy as np
from sklearn.metrics import confusion_matrix

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

def get_gradient_FW_val(y_dat, conf_overall_val, metric, classes):
    """
    Takes a confusion on the validation data and returns the
    mathematical gradient at the metric at that confusion
    Args:
        dat: dataset
        conf_overall: overall confusion for the dataset dat
        metric: metric name
        classes: number of classes
    Returns:
        gradient of the metric at the provided confusion
    """

    # probability of class c
    pi = np.zeros(classes)
    for c in range(classes):
        pi[c] = len(y_dat[y_dat == c]) / len(y_dat)

    # Gradient of G-mean
    if (metric == "G-mean"):
        diagnonal_conf = conf_overall_val.diagonal().copy()
        diagnonal_conf[conf_overall_val.diagonal() == 0] = 0.0001
        numerate = np.sqrt((diagnonal_conf.prod()) / pi.prod())
        grad_free = (numerate / 2) * 1 / (diagnonal_conf)
  # Gradient of F-measure
    elif (metric == "F-measure"):
        row_sums = conf_overall_val.sum(axis=1)
        col_sums = conf_overall_val.sum(axis=0)
        denom = row_sums + col_sums
        diag = conf_overall_val.diagonal()
        grad_free = 2*(denom - 2*diag)/(classes*np.square(denom))
    else:
        assert (metric == "Accuracy"), "metric can be G-mean, F-measure, or Accuracy"
        # Gradient of Accuracy
        grad_free = np.ones(classes)

    return grad_free

def get_opt_confs_FW_val(y_dat, pred_prob, weights, classes):
    """
    Takes data, estimate of the conditional probability of the positive class,
    weights (costs) on the entries of the confusion, construct the Bayes optimal
    classifier corresponding to those weights, and then returns the optimal confusion
    Args:
        dat: data set for which the optimal confusion is required
        eta_model: estimate of the conditional probability of the positive class
        weights: costs for the different confusion entries
        classes: number of classes
    Returns:
        optimal confusion corresponding to Bayes optimal classifier for the given
        weights
    """
    # Finding the Bayes optimal classifier as desribed in the above paper
    final_weights = np.tile(weights, (y_dat.shape[0], 1))
    pred_dat = np.argmax(final_weights * pred_prob, axis=1)

    # Computing confusion for the optimal classifier
    conf = confusion_matrix(y_dat, pred_dat, normalize='all')

    return conf

def FW_val(num_iters, metric, y_val, y_hyper_val, y_test, pred_eta_model, classes, RANDOM_SEED):
    """
    This method implements the Frank-Wolfe method on the validation data 
    making use of the above subroutines.
    Args:
        num_iters: iterations in the FW optimization procedure
        metric: metric name
        val_data: validation data
        test_data: test data
        eta_model: estimate of the conditional probability of the positive class.
          Pass the one learned using the validation data
        classes: number of classes
    Returns:
        mval_val_list: list of metric value on the validation data over the
        iterations
        mval_test_list: list of metric value on the test data over the iterations
        grad_norm_list: list of norm of the gradient over the iterations
    """

    np.random.seed(RANDOM_SEED)

    mval_val_list = []
    mval_hyper_val_list = []
    mval_test_list = []
    grad_norm_list = []

    conf_grps_val_list = []
    conf_grps_hyper_val_list = []
    conf_grps_test_list = []

    # Initializing with a deterministic classfier / confusion selected at random
    init_weights = np.random.uniform(size=classes)
    init_conf_grps_val = get_opt_confs_FW_val(y_val, pred_eta_model['val'], 
                                              init_weights, classes)
    init_conf_grps_hyper_val = get_opt_confs_FW_val(y_hyper_val, pred_eta_model['hyper_val'], 
                                              init_weights, classes)
    init_conf_grps_test = get_opt_confs_FW_val(y_test, pred_eta_model['test'], 
                                               init_weights, classes)

    conf_grps_val_list.append(init_conf_grps_val)
    conf_grps_hyper_val_list.append(init_conf_grps_hyper_val)
    conf_grps_test_list.append(init_conf_grps_test)

    # FW iterations
    for i in range(num_iters):

        # computing the exact gradient at a confusion on validation data
        weights = get_gradient_FW_val(y_val, conf_grps_val_list[-1], 
                                      metric, classes)

        # given the linearized metric via weights find the optimal confusion
        conf_grps_val = get_opt_confs_FW_val(y_val, pred_eta_model['val'], 
                                             weights, classes)
        conf_grps_hyper_val = get_opt_confs_FW_val(y_hyper_val, pred_eta_model['hyper_val'], 
                                             weights, classes)
        conf_grps_test = get_opt_confs_FW_val(y_test, pred_eta_model['test'], 
                                              weights, classes)

        # convex combination used in the FW procedure
        new_conf_grps_val = (1 - (2 / (i + 2))) * (conf_grps_val_list[-1]) + (2 / (i + 2)) * (conf_grps_val)
        new_conf_grps_hyper_val = (1 - (2 / (i + 2))) * (conf_grps_hyper_val_list[-1]) + (2 / (i + 2)) * (conf_grps_hyper_val)
        new_conf_grps_test = (1 - (2 / (i + 2))) * (conf_grps_test_list[-1]) + (2 / (i + 2)) * (conf_grps_test)

        conf_grps_val_list.append(new_conf_grps_val)
        conf_grps_hyper_val_list.append(new_conf_grps_hyper_val)
        conf_grps_test_list.append(new_conf_grps_test)

        # Computing metric value on val and test
        mval_val_list.append(mval(new_conf_grps_val, metric, classes))
        mval_hyper_val_list.append(mval(new_conf_grps_hyper_val, metric, classes))
        mval_test_list.append(mval(new_conf_grps_test, metric, classes))

        grad_norm_list.append(np.linalg.norm(weights))

    return mval_val_list, mval_hyper_val_list, mval_test_list, grad_norm_list
