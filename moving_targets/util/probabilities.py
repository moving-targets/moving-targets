"""Utility functions to handle classes and probabilities"""

from typing import Optional

import numpy as np


def count_classes(vector: np.ndarray) -> int:
    """Counts the number of classes in a vector of class targets in the range {0, ..., C - 1}.

    :param vector:
        A vector of class values.

    :return:
        The number of classes.
    """
    return int(np.max(vector)) + 1


def get_classes(prob: np.ndarray, labelling: bool = False) -> np.ndarray:
    """Gets the output classes given the output probabilities per class.

    :param prob:
        An array of probabilities.

    :param labelling:
        Whether the task that generated these probabilities is a labelling or a classification task. This defines the
        strategy to get the discrete values from probabilities in case the given array is bi-dimensional (indeed, for
        one-dimensional vectors this parameter is ignored since the two tasks will be the same), which consists in
        rounding or taking the the argmax over each row, respectively.

    :return:
        The respective output classes/labels.
    """
    return prob.round().astype(int) if prob.squeeze().ndim == 1 or labelling else prob.argmax(axis=1)


def get_onehot(vector: np.ndarray, classes: Optional[int] = None, onehot_binary: bool = False) -> np.ndarray:
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
    vector = vector.squeeze()
    assert vector.ndim == 1, "Pleas provide a be one-dimensional vector"
    assert np.issubdtype(vector.dtype, np.integer), "Please provide a vector of integer numbers only"
    assert np.min(vector) >= 0, "Please provide a vector of non-negative class identifiers"
    # retrieves the correct number of samples and classes depending on the input
    samples = len(vector)
    classes = classes or count_classes(vector=vector)
    # if the task is binary there is nothing to encode, otherwise creates a SxC matrix and places the ones accordingly.
    if classes == 2 and not onehot_binary:
        return vector
    else:
        onehot = np.zeros((samples, classes), dtype=int)
        onehot[np.arange(samples), vector] = 1
        return onehot
