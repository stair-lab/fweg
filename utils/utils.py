import os
import json
from typing import Dict

import numpy as np
import pandas as pd

import tensorflow as tf
from keras.layers import Input, Activation, Dense, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import add
from keras.regularizers import l2
import tqdm

# TODO: decide whether or not to use this
import sklearn


def random_search(preds, y_true, classes, random_seed, tries=500):
    np.random.seed(random_seed)
    best_w = np.ones(classes) / classes
    y_hat = np.argmax(preds * best_w, axis=1)
    cf = sklearn.metrics.confusion_matrix(y_true, y_hat, normalize="all")
    best_score = cf.diagonal().sum()  # accuracy only right now
    for _ in range(tries):
        w = np.random.uniform(size=classes)
        y_hat = np.argmax(preds * w, axis=1)
        cf = sklearn.metrics.confusion_matrix(y_true, y_hat, normalize="all")
        score = cf.diagonal().sum()  # TODO: use eval_metric here
        if score > best_score:
            best_w = w
            best_score = score
    return best_w, best_score


def get_savepath(
    folder: str,
    base_name: str,
    ext: str,
    **kwargs,
) -> str:
    """
    Creates a consistent savepath. Examples explain the usage easiest:

    get_savepath("folder_test", "model_weight", ".h5") -> "folder_test/model_weight.h5"
    get_savepath("folder_test", "model_weight", ".h5", rs=7) -> "folder_test/model_weight_rs=7.h5"
    get_savepath("folder_test", "model_weight", ".h5", rs=7, abcd="true") -> "folder_test/model_weight_abcd=true_rs=7.h5"

    The kwargs are created by adding kwarg_name=<some string that will go into the savepath>
    They are sorted by kwarg_name so the order that they are passed does not matter.
    """
    assert ext[0] == "."  # must start with .
    savepath = os.path.join(folder, base_name)
    order = sorted(kwargs.keys())
    if len(order) == 0:
        # there are no keys
        return savepath + ext
    kwargs_str = ""
    for k in order:
        kwargs_str += f"_{k}={kwargs[k]}"
    return savepath + kwargs_str + ext


def load_config(config_path) -> Dict:
    """
    Load config JSON at `config_path` as dict
    """
    fp = open(config_path, "r")
    data = json.load(fp)
    fp.close()
    return data


def save_config(config_path, data):
    """
    Save config JSON at `config_path` from dict `data`
    """
    fp = open(config_path, "w")
    json.dump(data, fp)
    fp.close()


def compute_preds(model, x, batch_size=None):
    """
    Compute the predictions for `model` on `x`
    `x` types:
        - tf.data.Dataset. Return will be (predictions, labels) as numpy arrays
        - np.array. Return will be the predictions as a numpy array.
    """
    if issubclass(x.__class__, tf.data.Dataset):
        pred_list = []
        label_list = []
        for imgs, labels in tqdm.tqdm(x):
            preds = model.predict(imgs)
            pred_list.append(preds)
            # this turns the one-hot vector into [label1, label2, ...]
            l = np.argwhere(labels.numpy())[:, 1]
            label_list += l.tolist()  # list concatenate
        return np.vstack(pred_list), np.array(label_list)

    elif type(x) == np.ndarray:
        preds = model.predict(x=x, batch_size=batch_size)
        return preds

    else:
        raise ValueError(f"unexpected x type: {type(x)}")


def load_sorted_filenames(processed_f, subset):
    """
    `subset`: see load_sorted_df

    Loads the filenames of the subset alphabetically. This is the order that tensorflow does
    in tf.keras.image_dataset_from_directory if shuffle=False.
    """
    labels = os.listdir(os.path.join(processed_f, subset))
    labels = sorted(labels)
    filenames = list()
    for l in labels:
        filenames += sorted(os.listdir(os.path.join(processed_f, subset, l)))
    return filenames


def load_sorted_df(processed_f, subset):
    """
    `subset`: one of {train, hyper_train, val_full, test}

    Loads the saved dataframe of image metadata in the order that tensorflow loads images if shuffle=False
    in in tf.keras.image_dataset_from_directory.

    Does so by first calling load_sorted_filenames. Each image is named {df_idx}.{extension}. Then we just
    order the saved dataframe by the order that tensorflow reads images.
    """
    assert subset in ["train", "hyper_train", "val", "hyper_val", "test"]
    filenames = load_sorted_filenames(processed_f, subset)
    idxs = [int(s.split(".")[0]) for s in filenames]
    df_subset = pd.read_csv(os.path.join(processed_f, f"{subset}.csv"))
    return df_subset.loc[idxs]


def make_resnet(
    depth, random_state, input_shape=(32, 32, 3), nc=10, final_activation="softmax"
):
    """
    Create a resnet as defined by the forward correction paper. Copied from https://github.com/giorgiop/loss-correction

    Paper: https://arxiv.org/pdf/1609.03683.pdf
    """

    # how many layers this is going to create?
    # 2 + 6 * depth

    tf.random.set_seed(random_state)

    decay = 1e-4
    input = Input(shape=input_shape)

    # 1 conv + BN + relu
    filters = 16
    b = Conv2D(
        filters=filters,
        kernel_size=(3, 3),
        kernel_initializer="he_normal",
        padding="same",
        kernel_regularizer=l2(decay),
        bias_regularizer=l2(0),
    )(input)
    b = BatchNormalization(axis=3)(b)
    b = Activation("relu")(b)

    # 1 res, no striding
    b = residual(filters, decay, first=True)(b)  # 2 layers inside
    for _ in np.arange(1, depth):  # start from 1 => 2 * depth in total
        b = residual(filters, decay)(b)

    filters *= 2

    # 2 res, with striding
    b = residual(filters, decay, more_filters=True)(b)
    for _ in np.arange(1, depth):
        b = residual(filters, decay)(b)

    filters *= 2

    # 3 res, with striding
    b = residual(filters, decay, more_filters=True)(b)
    for _ in np.arange(1, depth):
        b = residual(filters, decay)(b)

    b = BatchNormalization(axis=3)(b)
    b = Activation("relu")(b)

    b = AveragePooling2D(pool_size=(8, 8), strides=(1, 1), padding="valid")(b)

    out = Flatten()(b)
    dense = Dense(
        units=nc,
        kernel_initializer="he_normal",
        activation=final_activation,
        kernel_regularizer=l2(decay),
        bias_regularizer=l2(0),
    )(out)

    return tf.keras.models.Model(inputs=input, outputs=dense)


def residual(filters, decay, more_filters=False, first=False):
    """
    Helper for resnet. Creates residual layers. Copied from https://github.com/giorgiop/loss-correction

    Paper: https://arxiv.org/pdf/1609.03683.pdf
    """

    def f(input):

        if more_filters and not first:
            stride = 2
        else:
            stride = 1

        if not first:
            b = BatchNormalization(3)(input)
            b = Activation("relu")(b)
        else:
            b = input

        b = Conv2D(
            filters=filters,
            kernel_size=(3, 3),
            strides=(stride, stride),
            kernel_initializer="he_normal",
            padding="same",
            kernel_regularizer=l2(decay),
            bias_regularizer=l2(0),
        )(b)
        b = BatchNormalization(axis=3)(b)
        b = Activation("relu")(b)
        res = Conv2D(
            filters=filters,
            kernel_size=(3, 3),
            kernel_initializer="he_normal",
            padding="same",
            kernel_regularizer=l2(decay),
            bias_regularizer=l2(0),
        )(b)

        # check and match number of filter for the shortcut
        input_shape = tf.keras.backend.int_shape(input)
        residual_shape = tf.keras.backend.int_shape(res)
        if not input_shape[3] == residual_shape[3]:

            stride_width = int(round(input_shape[1] / residual_shape[1]))
            stride_height = int(round(input_shape[2] / residual_shape[2]))

            input = Conv2D(
                filters=residual_shape[3],
                kernel_size=(1, 1),
                strides=(stride_width, stride_height),
                kernel_initializer="he_normal",
                padding="valid",
                kernel_regularizer=l2(decay),
            )(input)

        return add([input, res])

    return f
