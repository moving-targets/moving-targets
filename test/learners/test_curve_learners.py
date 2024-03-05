import random

import numpy as np
from scipy.optimize import curve_fit

from moving_targets.learners import ScipyCurveFit
from test.learners.test_learners import TestLearners


class TestCurveLearners(TestLearners):
    @staticmethod
    def _curve(x, a, b):
        return a * x.sum(axis=1) + b

    @classmethod
    def _random_state(cls):
        random.seed(cls.SEED)
        np.random.seed(cls.SEED)

    def _reference(self, learner, x, y, sample_weight: np.ndarray) -> np.ndarray:
        if learner == 'scipy_curve_fit':
            a, b = curve_fit(f=self._curve, xdata=x, ydata=y)[0]
            return self._curve(x=x, a=a, b=b)
        else:
            raise AssertionError(f"Unknown reference learner '{learner}'")

    def test_scipy_curve_fit(self):
        self._test(mt_learner=lambda: ScipyCurveFit(curve=self._curve),
                   ref_learner=lambda: 'scipy_curve_fit',
                   classification=False)
