import pandas as pd
import numpy as np


def get_basis_fns(
    groups,
    eval_groups,
    z_train,
    z_eval_train,
    z_val,
    z_eval_val,
    z_test,
    z_eval_test,
    add_all_grp=0,
):
    """
    Creates a separate dataset whose columns are RBF's or GROUPS or both, and
    rows denote the membership value for a sample for that basis function.

    Args:
        TODO

    Returns:
        TODO
    """

    if len(groups) == 0:
        add_all_grp = 1

    basis_train = pd.DataFrame()
    basis_val = pd.DataFrame()
    basis_test = pd.DataFrame()

    eval_train = pd.DataFrame()
    eval_val = pd.DataFrame()
    eval_test = pd.DataFrame()

    for g_id, g in enumerate(groups):
        basis_train[g] = z_train[g_id]
        basis_val[g] = z_val[g_id]
        basis_test[g] = z_test[g_id]

    for g_id, g in enumerate(eval_groups):
        eval_train[g] = z_eval_train[g_id]
        eval_val[g] = z_eval_val[g_id]
        eval_test[g] = z_eval_test[g_id]

    if add_all_grp:
        basis_train["All"] = np.ones(z_eval_train[0].shape[0])
        basis_val["All"] = np.ones(z_eval_val[0].shape[0])
        basis_test["All"] = np.ones(z_eval_test[0].shape[0])

    grp_id_arr = list(basis_train.columns)
    eval_grp_id_arr = list(eval_train.columns)

    print("Basis functions are ", grp_id_arr)
    print("Evaluation groups are ", eval_grp_id_arr)

    return (
        basis_train,
        eval_train,
        basis_val,
        eval_val,
        basis_test,
        eval_test,
        grp_id_arr,
        eval_grp_id_arr,
    )


def get_confs_frm_scr(y_dat, basis_dat, eval_dat, pred_scr, classes, pseudo_conf=False):
    """
    Given a dataset, basis functions, and predictions for the samples in the dataset,
    this function returns group-wise confusions and overall confusions.
    Can also be used to get pseudo group-wise confusions to be used in non-trivial
    feasible perturbations.

    Args:
        TODO
    """

    label_all = y_dat
    label_one_hot = np.zeros((label_all.size, classes))
    label_one_hot[np.arange(label_all.size), label_all] = 1

    # The following chunk of code computes group-wise confusions using memebrship defined
    # by basis functions

    grp_conf_dict = {}

    for grp_id in list(basis_dat.columns):

        # flag to check whether pseudo-confusions are required or not
        #         if (pseudo_conf):
        #             basis_fn = np.square(basis_dat[grp_id].values)
        #         else:
        basis_fn = basis_dat[grp_id].values

        conf_grp = np.zeros((classes, classes))
        for tl in range(classes):
            for pl in range(classes):
                conf_grp[tl, pl] = np.sum(
                    pred_scr[:, pl] * label_one_hot[:, tl] * basis_fn
                ) / np.sum(basis_fn)

        grp_conf_dict[grp_id] = conf_grp

    eval_grp_conf_dict = {}
    for grp_id in list(eval_dat.columns):

        eval_fn = eval_dat[grp_id].values

        conf_grp = np.zeros((classes, classes))
        for tl in range(classes):
            for pl in range(classes):
                conf_grp[tl, pl] = np.sum(
                    pred_scr[:, pl] * label_one_hot[:, tl] * eval_fn
                ) / np.sum(eval_fn)

        eval_grp_conf_dict[grp_id] = conf_grp

    # The following chunk of code computes overall confusions. Does not require basis functions
    conf_all = np.zeros((classes, classes))
    for tl in range(classes):
        for pl in range(classes):
            conf_all[tl, pl] = np.sum(pred_scr[:, pl] * label_one_hot[:, tl]) / len(
                label_all
            )

    return grp_conf_dict, eval_grp_conf_dict, conf_all


def ce_eta(
    metric,
    y_val,
    basis_val,
    eval_val,
    y_test,
    basis_test,
    eval_test,
    pred_eta_model,
    classes,
):
    """
    Takes the passed model for conditional probability, creates a classifier
    which weigh errors on all classes equally, and returns the metric value
    on train and test data.

    Args:
        metric: metric value we are optimizing
        val_data: validation data
        test_data: test data
        eta_model: any scikit based model for conditional probability
        classes: number of classes

    Returns:
        mval_val: metric value of the classifier on validation data
        mval_test: metric value of the classifier on test data
    """

    # obtain prediction probabilties on validation data
    pred_prob_val = deepcopy(pred_eta_model["val"])
    label_val = y_val

    # obtain prediction probabilties on test data
    pred_prob_test = deepcopy(pred_eta_model["test"])
    label_test = y_test

    # Thresholding the probabilities on 0.5 threhold. Hence equal weights for both errors
    pred_val = np.argmax(pred_prob_val, axis=1)
    pred_test = np.argmax(pred_prob_test, axis=1)

    pred_val_one_hot = np.zeros((pred_val.size, classes))
    pred_val_one_hot[np.arange(pred_val.size), pred_val] = 1

    pred_test_one_hot = np.zeros((pred_test.size, classes))
    pred_test_one_hot[np.arange(pred_test.size), pred_test] = 1

    _, conf_val, _ = get_confs_frm_scr(
        label_val, basis_val, eval_val, pred_val_one_hot, classes
    )
    _, conf_test, _ = get_confs_frm_scr(
        label_test, basis_test, eval_test, pred_test_one_hot, classes
    )

    # Computing confusion of the classifier on validation and test data
    #     conf_val = confusion_matrix(label_val, pred_val, normalize='all')
    #     conf_test = confusion_matrix(label_test, pred_test, normalize='all')

    # Computing the metric value for the obtained confusion
    mval_val, g_val = mval(conf_val, metric, classes)
    mval_test, g_test = mval(conf_test, metric, classes)

    return mval_val, mval_test
