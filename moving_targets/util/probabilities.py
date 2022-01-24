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


def get_discrete(prob: np.ndarray, task: str = 'auto') -> np.ndarray:
    """Gets the output classes given the output probabilities per class.

    :param prob:
        A vector/matrix of output probabilities.

    :param task:
        The kind of task that generated these probabilities, either 'classification' or 'labelling'.

        For 'classification' tasks, returns classes as the argmax over each row, while for 'labelling' tasks returns
        labels via rounding. If 'auto' is passed, it tries to automatically infer the kind of task depending on whether
        the probabilities over each row sum up to one or not (notice that, if the given vector of probabilities is
        one-dimensional, this parameter is ignored since the two tasks will be the same).

    :return:
        The respective output classes/labels.
    """
    # handle task depending on input dimension and given parameter
    if prob.squeeze().ndim == 1:
        rounding = True
    elif task == 'auto':
        rounding = not np.allclose(prob.sum(axis=1), 1.0)
    elif task == 'classification':
        rounding = False
    elif task == 'labelling':
        rounding = True
    else:
        raise AssertionError(f"'task' should be either 'classification', 'labelling', or 'auto', but is {task}")
    # return classes/labels
    return prob.round().astype(int) if rounding else prob.argmax(axis=1)


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
