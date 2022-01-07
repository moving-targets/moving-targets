import os
import sys
import unittest


class AbstractTest(unittest.TestCase):
    SEED: int = 0
    """The chosen random seed."""

    NUM_TESTS: int = 5
    """The number of tests carried out for the same loss."""

    NUM_SAMPLES: int = 20
    """The number of data points."""

    NUM_FEATURES: int = 5
    """The number of input features."""

    NUM_CLASSES: int = 3
    """The number of class labels for multi-class classification tests."""

    PLACES: int = 3
    """The number of digits passed to the `assertAlmostEqual()` method."""

    @staticmethod
    def get_relative_path(*path: str) -> str:
        # when running in debug mode, the root path is in third position
        root = sys.path[1] if 'Moving Targets' in sys.path[1] else sys.path[2]
        return os.sep.join((root,) + path)
