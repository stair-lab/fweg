import tensorflow as tf
import numpy as np

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.keras.layers.preprocessing import image_preprocessing
from tensorflow.python.keras.preprocessing import dataset_utils
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.util.tf_export import keras_export

WHITELIST_FORMATS = (".bmp", ".gif", ".jpeg", ".jpg", ".png")


def image_dataset_from_directory(
    directory,
    labels="inferred",
    label_mode="int",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(256, 256),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
):
    """
    This is copy-pasted from https://stackoverflow.com/questions/62166588/how-to-obtain-filenames-during-prediction-while-using-tf-keras-preprocessing-ima
    We use it so we can know the image order when doing tf.keras.image_dataset_from_directory.
    """

    if labels != "inferred":
        if not isinstance(labels, (list, tuple)):
            raise ValueError(
                "`labels` argument should be a list/tuple of integer labels, of "
                "the same size as the number of image files in the target "
                "directory. If you wish to infer the labels from the subdirectory "
                'names in the target directory, pass `labels="inferred"`. '
                "If you wish to get a dataset that only contains images "
                "(no labels), pass `label_mode=None`."
            )
        if class_names:
            raise ValueError(
                "You can only pass `class_names` if the labels are "
                "inferred from the subdirectory names in the target "
                'directory (`labels="inferred"`).'
            )
    if label_mode not in {"int", "categorical", "binary", None}:
        raise ValueError(
            '`label_mode` argument must be one of "int", "categorical", "binary", '
            "or None. Received: %s" % (label_mode,)
        )
    if color_mode == "rgb":
        num_channels = 3
    elif color_mode == "rgba":
        num_channels = 4
    elif color_mode == "grayscale":
        num_channels = 1
    else:
        raise ValueError(
            '`color_mode` must be one of {"rbg", "rgba", "grayscale"}. '
            "Received: %s" % (color_mode,)
        )
    interpolation = image_preprocessing.get_interpolation(interpolation)
    dataset_utils.check_validation_split_arg(validation_split, subset, shuffle, seed)

    if seed is None:
        seed = np.random.randint(1e6)
    image_paths, labels, class_names = dataset_utils.index_directory(
        directory,
        labels,
        formats=WHITELIST_FORMATS,
        class_names=class_names,
        shuffle=shuffle,
        seed=seed,
        follow_links=follow_links,
    )

    if label_mode == "binary" and len(class_names) != 2:
        raise ValueError(
            'When passing `label_mode="binary", there must exactly 2 classes. '
            "Found the following classes: %s" % (class_names,)
        )

    image_paths, labels = dataset_utils.get_training_or_validation_split(
        image_paths, labels, validation_split, subset
    )

    dataset = paths_and_labels_to_dataset(
        image_paths=image_paths,
        image_size=image_size,
        num_channels=num_channels,
        labels=labels,
        label_mode=label_mode,
        num_classes=len(class_names),
        interpolation=interpolation,
    )
    if shuffle:
        # Shuffle locally at each iteration
        dataset = dataset.shuffle(buffer_size=batch_size * 8, seed=seed)
    dataset = dataset.batch(batch_size)
    # Users may need to reference `class_names`.
    dataset.class_names = class_names
    return dataset, image_paths


def paths_and_labels_to_dataset(
    image_paths,
    image_size,
    num_channels,
    labels,
    label_mode,
    num_classes,
    interpolation,
):
    """Constructs a dataset of images and labels."""
    path_ds = dataset_ops.Dataset.from_tensor_slices(image_paths)
    img_ds = path_ds.map(
        lambda x: path_to_image(x, image_size, num_channels, interpolation)
    )
    if label_mode:
        label_ds = dataset_utils.labels_to_dataset(labels, label_mode, num_classes)
        img_ds = dataset_ops.Dataset.zip((img_ds, label_ds))
    return img_ds


def path_to_image(path, image_size, num_channels, interpolation):
    img = io_ops.read_file(path)
    img = image_ops.decode_image(img, channels=num_channels, expand_animations=False)
    img = image_ops.resize_images_v2(img, image_size, method=interpolation)
    img.set_shape((image_size[0], image_size[1], num_channels))
    return img


class ShowValidAndTest(tf.keras.callbacks.Callback):
    """
    Prints both validation and test errors
    """

    def __init__(
        self,
        x_valid,
        x_test,
        batch_size=None,
        epoch_freq=1,
    ):
        super(ShowValidAndTest).__init__()
        if issubclass(x_valid.__class__, tf.data.Dataset):
            assert batch_size is None, print(
                "Batch size is provided in the tf.data.Dataset itself"
            )
            self.is_tf_ds = True
            self.x_valid = x_valid
            self.x_test = x_test
            self.epoch_freq = epoch_freq

        elif type(x_valid) == tuple:
            self.is_tf_ds = False
            (x_valid, y_valid) = x_valid
            (x_test, y_test) = x_test
            self.x_valid = x_valid
            self.y_valid = np.argwhere(y_valid)[:, 1]
            self.x_test = x_test
            self.y_test = np.argwhere(y_test)[:, 1]
            self.batch_size = batch_size
            self.epoch_freq = epoch_freq

        else:
            raise ValueError(f"unrecognized input of type {type(x_valid)}")

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.epoch_freq == 0:
            valid_acc = 0
            test_acc = 0

            if self.is_tf_ds:
                # handle tf.data.Dataset case
                correct = 0
                total = 0
                for imgs, labels in self.x_valid:
                    preds = self.model(imgs)
                    p = np.argmax(preds, axis=1)
                    # this turns the one-hot vector into [label1, label2, ...]
                    l = np.argwhere(labels)[:, 1]
                    correct += (p == l).sum()
                    total += len(imgs)
                valid_acc = correct / total

                correct = 0
                total = 0
                for imgs, labels in self.x_test:
                    preds = self.model(imgs)
                    p = np.argmax(preds, axis=1)
                    # this turns the one-hot vector into [label1, label2, ...]
                    l = np.argwhere(labels)[:, 1]
                    correct += (p == l).sum()
                    total += len(imgs)
                test_acc = correct / total

            else:
                # handle numpy tuple case
                # self.model.evaulate does not work
                valid_pred = self.model.predict(
                    x=self.x_valid,
                    verbose=0,
                    batch_size=self.batch_size,
                )
                test_pred = self.model.predict(
                    x=self.x_test,
                    verbose=0,
                    batch_size=self.batch_size,
                )

                valid_pred = np.argmax(valid_pred, axis=1)
                test_pred = np.argmax(test_pred, axis=1)

                valid_acc = (valid_pred == self.y_valid).mean()
                test_acc = (test_pred == self.y_test).mean()

            print("\n")
            print("valid:", valid_acc)
            print("test:", test_acc)
            print("\n")
