"""
This file implements the main contribition of this paper: FWEG.
"""
import sys
from typing import List, Dict, Tuple, Callable, Union

import numpy as np
import pandas as pd
import tqdm

# custom
from utils.metrics import SUPPORTED_METRICS, BlackBoxMetric, eval_metric, eval_true_metric
from utils.record import format_fweg_extra, Results_Recorder


class FWEG:
    def __init__(
        self,
        metric: str,
        num_iters: int,
        epsilon: float,
        classes: int,
        use_linear_val_metric: bool,
        random_state: int,
    ):
        """
        Args:
            metric: must be in SUPPORTED_METRICS
            group_ids: list of grp_ids (groups)
            num_iters: number of iterations for the algorithm
            epsilon: perturbation parameter (amount of perturbation)
            classes: number of classes
            use_linear_val_metric: True if we should use the gradient of the metric on the validation
                data to evaluate metric improvements instead of the metric itself. This requires knowing
                the closed-from formula of the metric and its gradient w.r.t. diagonal
                confusion entries. In some applications, the metric is black-box and defined
                only on the validation/test data. We would not be able to compute the gradient
                in those cases.
            random_state: internal random state for reproducability
        """
        if isinstance(metric, str):
            assert metric in SUPPORTED_METRICS
        else:
            assert isinstance(metric, BlackBoxMetric)
            assert use_linear_val_metric is False

        self.metric = metric
        self.num_iters = num_iters
        self.epsilon = epsilon
        self.nc = classes
        self.use_linear_val_metric = use_linear_val_metric
        self.random_state = random_state

        # set during FWEG
        self.group_ids = None
        self.ng = None
        self.classifier_weights = None

    def fit(
        self,
        preds_prob_train: np.array,
        y_train: np.array,
        basis_train: pd.DataFrame,
        preds_prob_val: np.array,
        y_val: np.array,
        basis_val: np.array,
        verbose: bool = True,
    ):
        """
        TODO: write method

        Args:
            preds_prob_train: (num_samples, classes) shape probability predictions on train set
            y_train: (num_samples) shape true train labels (integers)
            basis_train: (num_samples, num_groups) shape assignment of train samples to groups
            preds_prob_val: (num_samples, classes) shape probability predictions on valid set
            y_val: (num_samples) shape true valid labels (integers)
            basis_val: (num_samples, num_groups) shape assignment of valid samples to groups
            verbose: True if a progress bar should be rendered
        """
        np.random.seed(self.random_state)

        self.group_ids = list(basis_train.columns)
        self.ng = len(self.group_ids)
        preds_train_list = []
        preds_val_list = []
        mval_val_list = []
        grad_norm_list = []
        cond_list = []

        trivial = True
        # This part removes the singularity error when we hit a trivial classifier.
        while trivial:
            init_w = np.random.uniform(size=self.nc * self.ng)
            init_weights = dict(
                zip(self.group_ids, np.split(init_w, self.ng)),
            )
            # get train preds (using probabilities and weights) and confusions
            init_preds_train = self.get_opt_pred(
                preds_prob_train,
                basis_train,
                init_weights,
            )
            init_conf_grps_train, _ = self.get_confusions(
                init_preds_train,
                y_train,
                basis_train,
            )
            # get valid preds (using probabilities and weights) and confusions
            init_preds_val = self.get_opt_pred(
                preds_prob_val,
                basis_val,
                init_weights,
            )
            _, init_conf_all_val = self.get_confusions(
                init_preds_val,
                y_val,
                basis_val,
            )

            # make sure all groups do not have a trivial classifier
            trivial = False
            for grp_id in self.group_ids:
                pred_basis_train_grp = init_preds_train[basis_train[grp_id] > 0, :]
                pred_basis_val_grp = init_preds_val[basis_val[grp_id] > 0, :]
                if np.all(
                    pred_basis_train_grp == pred_basis_train_grp[0, :], axis=0
                ).all():
                    trivial = True
                    break
                if np.all(pred_basis_val_grp == pred_basis_val_grp[0, :], axis=0).all():
                    trivial = True
                    break
        if verbose:
            print("Initialization complete!", flush=True)

        preds_train_list.append(init_preds_train)
        preds_val_list.append(init_preds_val)

        init_mval = None
        if isinstance(self.metric, str):
            init_mval = eval_metric(init_conf_all_val, self.metric)
        elif isinstance(self.metric, BlackBoxMetric):
            init_mval = self.metric.eval_metric(init_preds_val, y_val, self.nc)
        else:
            raise ValueError(f"unknown metric type {type(self.metric)}")
        mval_val_list.append(init_mval)

        self.classifier_weights = [init_weights]

        prev_preds_train = init_preds_train
        prev_preds_val = init_preds_val
        
        # current conf on val set
        conf_all_val = init_conf_all_val

        # Frank Wolfe with Metric Elicitation starts
        if verbose:
            pbar = tqdm.tqdm(range(self.num_iters))
        for i in range(self.num_iters):
            # get_gradient will set this
            weights = None
            cond_SOE_x = None

            # get local linear approximation
            if self.use_linear_val_metric:
                true_grad = self.get_gradient_FW_val(
                    y_val,
                    conf_all_val,
                )
                weights, cond_SOE_x = self.get_gradient(
                    prev_preds_train,
                    y_train,
                    basis_train,
                    prev_preds_val,
                    y_val,
                    basis_val,
                    init_preds_train,
                    init_preds_val,
                    true_grad=true_grad,
                )
            else:
                weights, cond_SOE_x = self.get_gradient(
                    prev_preds_train,
                    y_train,
                    basis_train,
                    prev_preds_val,
                    y_val,
                    basis_val,
                    init_preds_train,
                    init_preds_val,
                )

            cond_list.append(cond_SOE_x)
            self.classifier_weights.append(weights)

            # get optimal predictions and confusion on train for the local linear approximation
            new_preds_train = self.get_opt_pred(
                preds_prob_train,
                basis_train,
                weights,
            )

            # get optimal predictions and confusion on val for the local linear approximation
            new_preds_val = self.get_opt_pred(
                preds_prob_val,
                basis_val,
                weights,
            )

            # Randomize the new classifier with the previous iterates
            new_preds_train = (1 - (2 / (i + 2))) * (prev_preds_train) + (
                2 / (i + 2)
            ) * (new_preds_train)
            new_preds_val = (1 - (2 / (i + 2))) * (prev_preds_val) + (2 / (i + 2)) * (
                new_preds_val
            )

            preds_train_list.append(new_preds_train)
            preds_val_list.append(new_preds_val)

            new_mval = None
            if isinstance(self.metric, str):
                _, conf_all_val = self.get_confusions(
                    new_preds_val,
                    y_val,
                    basis_val,
                )
                new_mval = eval_metric(conf_all_val, self.metric)
            elif isinstance(self.metric, BlackBoxMetric):
                new_mval = self.metric.eval_metric(new_preds_val, y_val, self.nc)
            else:
                raise ValueError(f"unknown metric type {type(self.metric)}")
            mval_val_list.append(new_mval)

            grad_norm_list.append(
                np.linalg.norm(np.concatenate(list(weights.values())))
            )

            if verbose:
                pbar.update(1)
                # update description with current metric value
                metric_str = self.metric if isinstance(self.metric, str) else ""
                pbar.set_description(
                    f"Val {metric_str}: {np.round(new_mval, decimals=3)}"
                )
            
            # use in the next iteration
            prev_preds_train = new_preds_train
            prev_preds_val = new_preds_val

        if verbose:
            pbar.close()

        return mval_val_list, grad_norm_list, cond_list

    def predict(
        self,
        preds_prob_test: np.array,
        y_test: np.array,
        basis_test: np.array,
        deterministic=False,
        metric=None,
    ) -> List[np.array]:
        """
        Apply learned weights to a test set.

        Args:
            preds_prob_test: (num_samples, classes) shape probability predictions on test set
            y_test: (num_samples) shape true train labels (integers). See `deterministic` for why we need this.
            basis_test: (num_samples, num_groups) shape assignment of test samples to groups
            deterministic: whether to use a deterministic version of the randomized classifer. Otherwise,
                we use the combination of randomized classifiers which means we combine their confusions
                (this is what the fit procedure does; we need y_test to make these confusions).
                The deterministic version is easier to port to "real" use cases where
                y_test cannot be provided as an argument

        Returns:
            If deterministic, return is a list of predictions at each iteration of FWEG.
            If non-deterministic, return is a list of confusion matrices at each iteration.
        """
        if metric is None:
            metric = self.metric
        if isinstance(metric, BlackBoxMetric):
            # caller must provide a metric interface filled-in for the current samples
            assert isinstance(metric, BlackBoxMetric)
        elif isinstance(metric, str):
            assert self.metric == metric
        else:
            raise ValueError(f"Unknown metric {metric}")

        if deterministic:
            # list of predictions
            preds_test_list = []

            # get optimal test preds using probabilities and weights
            cur_preds_test = self.get_opt_pred(
                preds_prob_test,
                basis_test,
                self.classifier_weights[0],  # init_weights
            )

            # use classifier weights up to stop_idx (+1 because stop_idx is the last index to use)
            # start from 1 because 0th is init_weights
            for i in range(1, len(self.classifier_weights)):
                # get optimal preds on test for the metric's local linear approx (classifier_weights[i])
                new_preds_test = self.get_opt_pred(
                    preds_prob_test,
                    basis_test,
                    self.classifier_weights[i],
                )

                # Randomize the new classifier with the previous iterates
                cur_preds_test = (1 - (2 / (i + 2))) * (cur_preds_test) + (
                    2 / (i + 2)
                ) * (new_preds_test)

                preds_test_list.append(np.argmax(cur_preds_test, axis=1))

            return preds_test_list

        else:
            # list of predictions and scores
            preds_test_list = []
            mval_test_list = []

            # get optimal test preds using probabilities and weights
            init_preds_test = self.get_opt_pred(
                preds_prob_test,
                basis_test,
                self.classifier_weights[0],  # init_weights
            )

            init_mval = None
            if isinstance(metric, str):
                _, init_conf_all_test = self.get_confusions(
                    init_preds_test,
                    y_test,
                    basis_test,
                )
                init_mval = eval_metric(init_conf_all_test, metric)
            elif isinstance(metric, BlackBoxMetric):
                init_mval = metric.eval_metric(init_preds_test, y_test, self.nc)
            else:
                raise ValueError(f"unknown metric type {type(metric)}")
            preds_test_list.append(init_preds_test)
            mval_test_list.append(init_mval)

            for i in range(0, len(self.classifier_weights) - 1):
                # get optimal preds on test for the metric's local linear approx (classifier_weights[i])
                new_preds_test = self.get_opt_pred(
                    preds_prob_test,
                    basis_test,
                    self.classifier_weights[
                        i + 1  # +1 because 0th index we already used
                    ],
                )

                # Randomize the predictions with the previous iterates
                new_preds_test = (1 - (2 / (i + 2))) * (preds_test_list[-1]) + (
                    2 / (i + 2)
                ) * (new_preds_test)

                new_mval = None
                if isinstance(metric, str):
                    _, new_conf_all_test = self.get_confusions(
                        new_preds_test,
                        y_test,
                        basis_test,
                    )
                    new_mval = eval_metric(new_conf_all_test, metric)
                elif isinstance(metric, BlackBoxMetric):
                    new_mval = metric.eval_metric(new_preds_test, y_test, self.nc)
                else:
                    raise ValueError(f"unknown metric type {type(metric)}")
                preds_test_list.append(new_preds_test)
                mval_test_list.append(new_mval)

            return preds_test_list, mval_test_list


    def get_opt_pred(
        self,
        preds: np.array,
        basis: pd.DataFrame,
        weights: Dict[str, np.array],
    ) -> np.array:
        """
        Given weights on confusion matrices for different groups, this function finds the
        optimal classifier's predictions on samples.

        Args:
            preds: (num_samples, classes) shape probability predictions
            basis: (num_samples, num_groups) shape assignment of samples to groups
            weights: dict mapping each group (str) to a np.array of (linear) metric weights on the
                diagonal of the confusion matrix. For example, if these were all 1 the metric would be accuracy.
        Returns:
            preds: One-hot optimal classifier predictions.
        """

        # Since a sample can belong to multiple groups, we aggregate weights corresponding to
        # each sample based on it groups membership
        weights_all = np.zeros((len(preds), self.nc))
        for grp_id in self.group_ids:
            # this is an array of shape (num_samples, 1) representing alignment to basis `grp_id`
            basis_mtx = basis[grp_id].values.reshape(-1, 1)
            # this is an array of shape (1, num_classes) representing the diagonal weights for each class for `grp_id`
            weights_mtx = weights[grp_id].reshape(1, -1)
            # this matrix multiply creates an array of shape (num_samples, num_classes)
            # result: each sample has its basis multiplied by the weights
            weights_all += basis_mtx @ weights_mtx

        # Creating a classifier based on given weights which determines predictions for samples
        pred = np.argmax(weights_all * preds, axis=1)

        # Predictions are one-hot
        pred_one_hot = np.zeros((pred.size, self.nc))
        pred_one_hot[np.arange(pred.size), pred] = 1

        return pred_one_hot

    def get_confusions(
        self,
        preds: np.array,
        y: np.array,
        basis: pd.DataFrame,
    ) -> np.array:
        """
        Given a dataset, basis functions, and predictions for the samples in the dataset,
        this function returns group-wise confusions and overall confusions.
        Can also be used to get pseudo group-wise confusions to be used in non-trivial
        feasible perturbations.
        Args:
            preds: (num_samples, classes) shape probability predictions
            y: (num_samples) shape true labels
            basis: (num_samples, num_groups) shape assignment of samples to groups
        Returns:
            grp_conf_dict: Groups-wise confusions for the given dataset and predictions
            conf_all: Overall confusions for the given dataset and predictions
        """
        y_one_hot = np.zeros((y.size, self.nc))
        y_one_hot[np.arange(y.size), y] = 1

        # This computes group-wise confusions using membership defined by basis functions
        grp_conf_dict = {}
        for grp_id in self.group_ids:
            basis_and_labels = basis[grp_id].values.reshape(-1, 1) * y_one_hot
            denom = np.sum(basis[grp_id].values)
            conf_grp = np.matmul(np.transpose(basis_and_labels), preds) / denom
            grp_conf_dict[grp_id] = conf_grp

        conf_all = np.matmul(np.transpose(y_one_hot), preds) / len(y_one_hot)
        return grp_conf_dict, conf_all

    def get_gradient_FW_val(
        self, y_val: np.array, conf_overall_val: np.array
    ) -> np.array:
        """
        Takes a confusion on the validation data and returns the mathematical gradient
        at the metric on that confusion with respect to the diagonals of the confusion
        matrix.

        Args:
            y_val: (num_samples) shape validation labels
            conf_overall_val: current overall confusions on val

        Returns:
            gradient of the metric at the provided confusion
        """
        assert self.metric in ["G-mean", "F-measure", "Accuracy"]

        # probability of class c
        pi = np.zeros(self.nc)
        for c in range(self.nc):
            pi[c] = len(y_val[y_val == c]) / len(y_val)

        if self.metric == "G-mean":
            diagnonal_conf = conf_overall_val.diagonal().copy()
            diagnonal_conf[conf_overall_val.diagonal() == 0] = 0.0001
            numerate = np.sqrt((diagnonal_conf.prod()) / pi.prod())
            grad_free = (numerate / 2) * 1 / (diagnonal_conf)
        elif self.metric == "F-measure":
            row_sums = conf_overall_val.sum(axis=1)
            col_sums = conf_overall_val.sum(axis=0)
            denom = row_sums + col_sums
            diag = conf_overall_val.diagonal()
            grad_free = 2 * (denom - 2 * diag) / (self.nc * np.square(denom))
        elif self.metric == "Accuracy":
            grad_free = np.ones(self.nc)
        else:
            raise ValueError(f"Unrecognized metric {self.metric}")

        return grad_free

    def get_gradient(
        self,
        preds_train: np.array,
        y_train: np.array,
        basis_train: pd.DataFrame,
        preds_val: np.array,
        y_val: np.array,
        basis_val: pd.DataFrame,
        init_pred_train: np.array,
        init_pred_val: np.array,
        true_grad: np.array = None,
    ) -> Tuple[Dict[str, np.array], float]:
        """
        This function gets the local-linear approximation of the metric on the
        training data, i.e. weights on diagonal of confusions for each group defined by basis
        functions, by making use of perturbations defined above.
        Args:
            preds_train: current classifier's predictions on training data where
                perturbation needs to be done
            y_train: training labels
            basis_train: dataset containing the group membership of each training sample
            preds_val: current classifier's predictions on validation data where
                perturbation needs to be done
            y_val: validation labels
            basis_val: dataset containing the group membership of each validation sample
            init_pred_train: initial classifier's predictions on training data. This is
                to avoid singluar matrix error. The initial classifier is chosen such that
                its confusions are linearly independent of trivial confusions
            init_pred_val: initial classifier's predictions on validation data. This is
                to avoid singluar matrix error. The initial classifier is chosen such that
                its confusions are linearly independent of trivial confusions
            true_grad: provide true grad on current confusion. Only non-None when use_linear_val_metric=True.

        Returns:
            grad_grp: A dictionary of diagonal metric weights for each group. These define
                the local linear approximation of the metric on the training data.
            cond_no: condtion number of Delta X + lambda*I
        """
        if self.use_linear_val_metric:
            assert true_grad is not None
        else:
            assert true_grad is None

        # SOE = system of equations
        SOE_x = np.zeros((self.ng * self.nc, self.ng * self.nc))
        SOE_y = np.zeros(self.ng * self.nc)

        # Computing current confusions using current predictions
        conf_grps_train, _ = self.get_confusions(
            preds_train,
            y_train,
            basis_train,
        )

        _, conf_grps_val_overall = self.get_confusions(
            preds_val,
            y_val,
            basis_val,
        )

        base_mval = None
        if isinstance(self.metric, str):
            if self.use_linear_val_metric:
                # use the linearized metric on val, aka its gradient
                base_mval = eval_true_metric(conf_grps_val_overall, true_grad)
            else:
                # use the complete metric on val
                base_mval = eval_metric(conf_grps_val_overall, self.metric)
        elif isinstance(self.metric, BlackBoxMetric):
            base_mval = self.metric.eval_metric(preds_val, y_val, self.nc)
        else:
            raise ValueError(f"unknown metric type {type(self.metric)}")

        # for each group and each class we perturb with one of the following:
        eqn_num = 0
        for grp_id in self.group_ids:
            for j in range(self.nc):
                (
                    perturb_conf_grp_train,
                    # perturb_conf_all_val,
                    new_mval,
                ) = self.perturb_conf_trivial(
                    preds_train,
                    y_train,
                    basis_train,
                    preds_val,
                    y_val,
                    basis_val,
                    init_pred_train,
                    init_pred_val,
                    grp_id,
                    j,
                    true_grad=true_grad
                )

                # Constructing Delta X matrix and Delta y vector
                delta_x = np.zeros(self.ng * self.nc)
                for u_ind, u in enumerate(self.group_ids):
                    perturb_conf_train_mat = (
                        perturb_conf_grp_train[u] - conf_grps_train[u]
                    )
                    delta_x[
                        self.nc * u_ind : self.nc * (u_ind + 1)
                    ] = perturb_conf_train_mat.diagonal()

                delta_y = delta_y = new_mval - base_mval
                SOE_x[eqn_num, :] = delta_x
                SOE_y[eqn_num] = delta_y

                eqn_num += 1

        # We prefer to use closed form inverse for local linear approximation
        # Computing condition number for checking purposes
        grad = np.matmul(
            np.linalg.inv(SOE_x - ((0.0001 ** 2) * np.eye(SOE_x.shape[0]))), SOE_y
        )
        cond_no = np.linalg.cond(SOE_x - ((0.0001 ** 2) * np.eye(SOE_x.shape[0])))

        # Rearraging the gradient as weights for each grp
        grad_grp = {}
        for grp_id_ind, grp_id in enumerate(self.group_ids):
            grad_grp[grp_id] = grad[self.nc * grp_id_ind : self.nc * (grp_id_ind + 1)]

        return grad_grp, cond_no

    def perturb_conf_trivial(
        self,
        preds_train: np.array,
        y_train: np.array,
        basis_train: pd.DataFrame,
        preds_val: np.array,
        y_val: np.array,
        basis_val: pd.DataFrame,
        init_pred_train: np.array,
        init_pred_val: np.array,
        grp: str,
        c: int,
        true_grad = None,
    ):
        """
        This functions perturbs the classifier predictions for samples belonging to
        a group with one of the trivial classifiers i.e. classifiers which predicts
        only one class on all samples. At the end we realize the pertubation in the
        group-wise and overall confusion space.

        Args:
            See get_gradient. New args:
            grp: string defining the group to perturb
            c: integer defining class to perturb

        Returns:
            perturb_conf_grp_train: group-wise confusions for training data
                for the classifier h' defined above
            perturb_conf_all_val: metric value on val after perturbation
        """
        if self.use_linear_val_metric is True:
            assert true_grad is not None
        epsilon = self.epsilon

        # Defining group memebership of samples based on provided grp
        basis_fn_train = basis_train[grp].values.reshape(-1, 1)
        basis_fn_val = basis_val[grp].values.reshape(-1, 1)

        # This defined the perturbations based on trivial classifiers on training data
        before_sum_train = preds_train[:, c].sum()
        trivial_conf_cls = np.zeros(preds_train.shape)
        trivial_conf_cls[:, c] = 1
        # this is a "real" perturbation, as the predictions don't exceed 1
        # perturb_pred_train = preds_train + basis_fn_train * epsilon * (
        #     trivial_conf_cls - preds_train
        # )
        coeff = basis_fn_train * epsilon
        perturb_pred_train = (1.0 - coeff) * preds_train + coeff * trivial_conf_cls
        after_sum_train = perturb_pred_train[:, c].sum()

        # This defined the perturbations based on trivial classifiers on validation data
        before_sum_val = preds_val[:, c].sum()
        trivial_conf_cls = np.zeros(preds_val.shape)
        trivial_conf_cls[:, c] = 1
        # this is a "real" perturbation, as the predictions don't exceed 1
        # perturb_pred_val = preds_val + basis_fn_val * epsilon * (
        #     trivial_conf_cls - preds_val
        # )
        coeff = basis_fn_val * epsilon
        perturb_pred_val = (1.0 - coeff) * preds_val + coeff * trivial_conf_cls
        after_sum_val = perturb_pred_val[:, c].sum()

        # This is condition when no perturbation happened, that means one was already
        # at a trivial classifier. When this happens we perturb with the initial classifier
        # TODO: handle this more elegantly
        if before_sum_train == after_sum_train or before_sum_val == after_sum_val:
            print("UNEXPECTED - already had a trivial classifier")
            perturb_pred_train = preds_train + basis_fn_train * epsilon * (
                init_pred_train - preds_train
            )
            perturb_pred_val = preds_val + basis_fn_val * epsilon * (
                init_pred_val - preds_val
            )

        # Computing group-wise confusion of h' on training data
        perturb_conf_grp_train, _ = self.get_confusions(
            perturb_pred_train,
            y_train,
            basis_train,
        )

        new_mval = None
        if isinstance(self.metric, str):
            # Computing group-wise confusion of h' on validation data
            _, perturb_conf_all_val = self.get_confusions(
                perturb_pred_val,
                y_val,
                basis_val,
            )
            if self.use_linear_val_metric:
                new_mval = eval_true_metric(perturb_conf_all_val, true_grad)
            else:
                new_mval = eval_metric(perturb_conf_all_val, self.metric)
        elif isinstance(self.metric, BlackBoxMetric):
            new_mval = self.metric.eval_metric(perturb_pred_val, y_val, self.nc)
        else:
            raise ValueError(f"unknown metric type {type(self.metric)}")

        return perturb_conf_grp_train, new_mval



class FWEG_Hyperparameter_Search:
    def __init__(
        self,
        recorder: Results_Recorder,
        classes: int,
        num_iters: int,
        metric: str,
        basis_fn_generator: Callable[
            [int, bool], Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
        ],
        groups_list: List[Union[List[str], int]],
        groups_descr_list: List[str],
        add_all_list: List[bool],
        epsilon_list: List[float],
        use_linear_val_metric_list: List[bool],
        random_seed: int,
        use_convergence: bool = False
    ):
        """
        Args:
            recorder: instance of our structured results logger
            classes: number of classes
            num_iters: number of iterations to run FWEG
            metric: must be in SUPPORTED_METRICS
            basis_fn_generator: a callable that takes (num_groups, add_all) as arguments and outputs
                four basis DataFrames: (basis_train, basis_val, basis_hyper_val, basis_test). It must
                takes named arguments groups=<>, add_all=<>.
            groups_list: list of number of groups to try
            groups_descr_list: list of strings explaining what each group means
            add_all_list: list of add_all values to try
            epsilon_list: list of epsilons to try
            use_linear_val_metric: list of use_linear_val to try
            random_seed: seed for reproducability
            use_convergence: if True, this picks the argmax of the latter half of hyper_val scores
                instead of all of them. For some metrics that take a while to converge, this helps
                pick FWEG runs that are more stable and do not have spikey performance.
        """
        if isinstance(metric, str):
            assert metric in SUPPORTED_METRICS
        else:
            assert isinstance(metric, BlackBoxMetric)
            assert use_linear_val_metric_list == [False]
        assert len(groups_list) == len(groups_descr_list)

        self.recorder = recorder
        self.classes = classes
        self.num_iters = num_iters
        self.metric = metric
        self.basis_fn_generator = basis_fn_generator
        self.groups_list = groups_list
        self.groups_descr_list = groups_descr_list
        self.add_all_list = add_all_list
        self.epsilon_list = epsilon_list
        self.use_linear_val_metric_list = use_linear_val_metric_list
        self.random_seed = random_seed
        self.use_convergence = use_convergence

    def search(
        self,
        preds_train: np.array,
        y_train: np.array,
        preds_val: np.array,
        y_val: np.array,
        preds_hyper_val: np.array,
        y_hyper_val: np.array,
        preds_test: np.array,
        y_test: np.array,
        metric_hyper_val=None,
        metric_test=None
    ) -> Tuple[int, bool, float, bool]:
        """
        Run FWEG over all hyperparameters. See FWEG.fit() for an explanation of all
        of these arguments

        Returns [best_groups, best_add_all, best_epsilon, best_use_linear_val_metric]
        """
        if isinstance(self.metric, BlackBoxMetric):
            assert isinstance(metric_hyper_val, BlackBoxMetric)
            assert isinstance(metric_test, BlackBoxMetric)

        best_hyper_val_score = 0
        best_test_score = 0
        best_groups = None
        best_add_all = None
        best_epsilon = None
        best_use_linear_val_metric = None

        total_params = (
            len(self.groups_list)
            * len(self.add_all_list)
            * len(self.epsilon_list)
            * len(self.use_linear_val_metric_list)
        )
        pbar = tqdm.tqdm(range(total_params))

        for groups, groups_descr in zip(self.groups_list, self.groups_descr_list):
            for add_all in self.add_all_list:
                # two cases
                # 1. number of groups is 1, in which case add_all should be False because the single group will contain everyone
                # 2. number of groups is [], in which case add_all should be True to make the single group that containers everyone
                # this could be written much better :(
                if (type(groups) == int and groups == 1 and add_all is True) or (type(groups) == list and len(groups) == 0 and add_all is False):
                    # skip this because 1 group just means add_all is already True
                    # this skips all combinations of epsilon_list and use_linear_val_metric_list
                    # so we update the pbar by that amount
                    pbar.update(
                        len(self.epsilon_list) * len(self.use_linear_val_metric_list)
                    )
                    continue

                (
                    basis_train,
                    basis_val,
                    basis_hyper_val,
                    basis_test,
                ) = self.basis_fn_generator(groups=groups, add_all=add_all)

                for epsilon in self.epsilon_list:
                    for use_linear_val_metric in self.use_linear_val_metric_list:
                        if use_linear_val_metric is True and self.metric == "Accuracy":
                            # skip this because the accuracy metric is already linearized on validation
                            # side. Computing the linearized metric value is the same as computing
                            # the original metric value.
                            pbar.update(1)
                            continue

                        # # main body:
                        # # 1. instantiate FWEG
                        # # 2. run on val
                        # # 3. choose best iteration on hyper_val
                        # # 4. apply to test
                        fweg = FWEG(
                            self.metric,
                            self.num_iters,
                            epsilon,
                            self.classes,
                            use_linear_val_metric,
                            self.random_seed,
                        )
                        mval_val_list, _, _ = fweg.fit(
                            preds_train,
                            y_train,
                            basis_train,
                            preds_val,
                            y_val,
                            basis_val,
                            verbose=False,  # disable pbar to not interfere with current pbar
                        )

                        # apply to hyper val set
                        preds_hyper_val_list, mval_hyper_val_list = fweg.predict(
                            preds_hyper_val,
                            y_hyper_val,
                            basis_hyper_val,
                            deterministic=False,
                            metric=metric_hyper_val,
                        )
                        best_idx = None
                        if self.use_convergence:
                            # use the latter half of scores to find the argmax
                            # this can find FWEG's that had better convergences
                            start = self.num_iters // 2
                            best_idx = start + np.argmax(mval_hyper_val_list[start:])
                        else:
                            # just do a naive argmax
                            best_idx = np.argmax(mval_hyper_val_list)
                        val_score = mval_val_list[best_idx]
                        hyper_val_score = mval_hyper_val_list[best_idx]
                        # apply to test set
                        preds_test_list, mval_test_list = fweg.predict(
                            preds_test,
                            y_test,
                            basis_test,
                            deterministic=False,
                            metric=metric_test,
                        )
                        test_score = mval_test_list[best_idx]

                        fweg_params = format_fweg_extra(
                            self.num_iters,
                            groups,
                            groups_descr,
                            add_all,
                            epsilon,
                            use_linear_val_metric,
                        )
                        
                        self.recorder.save(
                            self.random_seed,
                            self.metric,
                            "fweg",
                            val_score,
                            hyper_val_score,
                            test_score,
                            fweg_params,
                        )

                        if hyper_val_score > best_hyper_val_score:
                            best_hyper_val_score = hyper_val_score
                            best_test_score = test_score
                            best_groups = groups
                            best_add_all = add_all
                            best_epsilon = epsilon
                            best_use_linear_val_metric = use_linear_val_metric

                        msg = f"best hyper val: {np.round(best_hyper_val_score, decimals=4)}, test: {np.round(best_test_score, decimals=4)}"
                        pbar.set_description(msg)
                        pbar.update(1)

        pbar.close()
        return best_groups, best_add_all, best_epsilon, best_use_linear_val_metric
