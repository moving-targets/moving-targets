from collections import Callable

import numpy as np
from sklearn.preprocessing import LabelBinarizer

from moving_targets.util.probabilities import get_classes, get_onehot
from test.abstract import AbstractTest


class TestProbabilities(AbstractTest):
    BINARY_PROBABILITIES = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    BINARY_CLASSES = BINARY_PROBABILITIES.round().astype(int)
    BINARY_ONEHOT = np.transpose([1 - BINARY_CLASSES, BINARY_CLASSES])
    """1d vector of binary probabilities, 1d vector of binary classes, and 2d vector of binary classes, respectively."""

    MULTICLASS_PROBABILITIES = np.array([
        [0.7, 0.2, 0.1],
        [0.5, 0.4, 0.1],
        [0.4, 0.3, 0.3],
        [0.2, 0.6, 0.2],
        [0.3, 0.5, 0.2],
        [0.0, 1.0, 0.0],
        [0.1, 0.5, 0.4],
        [0.1, 0.4, 0.5],
        [0.4, 0.1, 0.5],
        [0.2, 0.2, 0.6]
    ])
    MULTICLASS_CLASSES = MULTICLASS_PROBABILITIES.argmax(axis=1)
    MULTICLASS_ONEHOT = LabelBinarizer().fit_transform(MULTICLASS_CLASSES)
    """2d vector of probabilities, 1d vector of classes, and 2d vector of onehot classes, respectively."""

    CATEGORIES_CLASSES = MULTICLASS_CLASSES + 1
    CATEGORIES_ONEHOT = np.array([
        [0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0],
    ])
    CATEGORIES = 5
    """1d vector of classes (classes in [1, 2, 3]), 2d vector of onehot classes (5 possible values), and number of
    categories, respectively."""

    def _test(self, inp: np.ndarray, ref: np.ndarray, fn: Callable):
        inp = fn(inp)
        self.assertEqual(inp.shape, ref.shape)
        self.assertTrue(np.all(inp == ref))

    def test_classes_binary_1d(self):
        self._test(inp=self.BINARY_PROBABILITIES, ref=self.BINARY_CLASSES, fn=get_classes)

    def test_classes_binary_2d(self):
        inp = np.transpose([1 - self.BINARY_PROBABILITIES, self.BINARY_PROBABILITIES])
        self._test(inp=inp, ref=self.BINARY_CLASSES, fn=get_classes)

    def test_classes_multiclass(self):
        self._test(inp=self.MULTICLASS_PROBABILITIES, ref=self.MULTICLASS_CLASSES, fn=get_classes)

    def test_onehot_binary_1d(self):
        self._test(inp=self.BINARY_CLASSES, ref=self.BINARY_CLASSES, fn=get_onehot)

    def test_onehot_binary_2d(self):
        inp = np.transpose([1 - self.BINARY_CLASSES, self.BINARY_CLASSES])
        self._test(inp=inp, ref=self.BINARY_ONEHOT, fn=get_onehot)

    def test_onehot_multiclass(self):
        self._test(inp=self.MULTICLASS_CLASSES, ref=self.MULTICLASS_ONEHOT, fn=get_onehot)

    def test_categories_auto(self):
        try:
            self._test(inp=self.CATEGORIES_CLASSES, ref=self.CATEGORIES_ONEHOT, fn=get_onehot)
        except AssertionError as exception:
            self.assertTrue(str(exception).startswith('Tuples differ: (10, 4) != (10, 5)'))

    def test_categories_integer(self):
        fn = lambda classes: get_onehot(vector=classes, classes=self.CATEGORIES)
        self._test(inp=self.CATEGORIES_CLASSES, ref=self.CATEGORIES_ONEHOT, fn=fn)
