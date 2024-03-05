import os
import sys
import unittest


class TestAbstract(unittest.TestCase):
    SEED: int = 42
    """The chosen random seed."""

    NUM_TESTS: int = 10
    """The number of tests carried out."""

    NUM_SAMPLES: int = 10
    """The number of data points."""

    NUM_FEATURES: int = 5
    """The number of input features."""

    NUM_CLASSES: int = 3
    """The number of class labels for multi-class classification tests."""

    PLACES: int = 3
    """The number of digits passed to the `assertAlmostEqual()` method."""

    @staticmethod
    def get_relative_path(*path: str) -> str:
        # paths may have different positions based on the environment in which the function is called
        # we look for the path of the project, i.e., the one that ends with the project name
        root = None
        for p in sys.path:
            if p.endswith('Moving Targets'):
                root = p
                break
        return os.sep.join((root,) + path)
