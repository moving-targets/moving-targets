"""Utility functions to handle classes and probabilities"""

from typing import Union

import numpy as np


def get_classes(prob: np.ndarray, multi_label: bool) -> np.ndarray:
    """Gets the output classes given the output probabilities per class.

    :param prob:
        A vector/matrix of output probabilities.

    :param multi_label:
        Whether the classification task must handle multiple labels or not.

    :return:
        The respective output classes.
    """
    # strategy varies depending on binary/multilabel vs. multiclass classification
    if multi_label or prob.squeeze().ndim == 1:
        return prob.round().astype(int)
    else:
        return prob.argmax(axis=1)


def get_onehot(vector: np.ndarray, classes: Union[None, int] = None, onehot_binary: bool = False) -> np.ndarray:
    """Gets the one hot encoded representation of the given classes.

    :param vector:
        A vector of class values.

    :param classes:
        Either an integer representing the number of categories or None, in which case the labels are inferred.

    :param onehot_binary:
        Whether or not to onehot encode a vector of binary variables.

    :return:
        The vector (in case the number of classes is 2) or matrix of binary variables.
    """
    assert np.issubdtype(vector.dtype, np.integer), "Please provide a vector of integer numbers only"
    assert np.min(vector) >= 0, "Please provide a vector of non-negative class identifiers"
    # retrieves the correct number of classes depending on the input and the number of samples
    classes = (np.max(vector) + 1) if classes is None else classes
    samples = len(vector)
    # if the task is binary there is nothing to encode, otherwise creates a SxC matrix and places the ones accordingly.
    if classes == 2 and not onehot_binary:
        return vector
    else:
        onehot = np.zeros((samples, classes))
        onehot[np.arange(samples), vector] = 1
        return onehot
