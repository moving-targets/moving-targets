from typing import Callable

import numpy as np
from sklearn.preprocessing import LabelBinarizer

from moving_targets.util.probabilities import get_classes, get_onehot
from test.test_abstract import TestAbstract


class TestProbabilities(TestAbstract):
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

    MULTILABEL_PROBABILITIES = np.array([
        [0.7, 0.4, 0.6],
        [0.5, 0.9, 0.1],
        [0.2, 0.3, 0.3],
        [0.4, 0.6, 0.2],
        [0.3, 0.5, 0.5],
        [0.0, 1.0, 0.0],
        [0.1, 0.1, 0.8],
        [0.1, 0.9, 0.5],
        [0.7, 0.9, 0.6],
        [0.5, 0.4, 0.6]
    ])
    MULTILABEL_ONEHOT = MULTILABEL_PROBABILITIES.round().astype(int)
    """2d vector of label probabilities, and 2d vector of onehot labels, respectively."""

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
        """Checks that the given function, when the given input is supplied, behaves correctly."""
        inp = fn(inp)
        self.assertEqual(inp.shape, ref.shape)
        self.assertTrue(np.all(inp == ref), msg=f'Input:\n{inp}\nReference:\n{ref}')

    def test_classes_binary_1d(self):
        self._test(inp=self.BINARY_PROBABILITIES, ref=self.BINARY_CLASSES, fn=get_classes)

    def test_classes_binary_2d(self):
        inp = np.transpose([1 - self.BINARY_PROBABILITIES, self.BINARY_PROBABILITIES])
        self._test(inp=inp, ref=self.BINARY_CLASSES, fn=get_classes)

    def test_classes_multiclass(self):
        self._test(inp=self.MULTICLASS_PROBABILITIES, ref=self.MULTICLASS_CLASSES, fn=get_classes)

    def test_classes_multilabel(self):
        self._test(inp=self.MULTILABEL_PROBABILITIES,
                   ref=self.MULTILABEL_ONEHOT,
                   fn=lambda prob: get_classes(prob=prob, labelling=True))

    def test_onehot_binary_1d(self):
        self._test(inp=self.BINARY_CLASSES, ref=self.BINARY_CLASSES, fn=get_onehot)

    def test_onehot_binary_2d(self):
        self._test(inp=self.BINARY_CLASSES,
                   ref=self.BINARY_ONEHOT,
                   fn=lambda vector: get_onehot(vector=vector, onehot_binary=True))

    def test_onehot_multiclass(self):
        self._test(inp=self.MULTICLASS_CLASSES, ref=self.MULTICLASS_ONEHOT, fn=get_onehot)

    def test_categories_auto(self):
        with self.assertRaises(AssertionError) as context:
            self._test(inp=self.CATEGORIES_CLASSES, ref=self.CATEGORIES_ONEHOT, fn=get_onehot)
        self.assertTrue(str(context.exception).startswith('Tuples differ: (10, 4) != (10, 5)'))

    def test_categories_integer(self):
        self._test(inp=self.CATEGORIES_CLASSES,
                   ref=self.CATEGORIES_ONEHOT,
                   fn=lambda vector: get_onehot(vector=vector, classes=self.CATEGORIES))
