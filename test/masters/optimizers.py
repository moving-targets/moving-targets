from typing import Callable, Dict

import numpy as np

from moving_targets.masters.losses import Loss, SAE, SSE, MAE, MSE, HammingDistance, CrossEntropy, \
    ReversedCrossEntropy, SymmetricCrossEntropy
from moving_targets.masters.optimizers import Optimizer, ConstantSlope, ExponentialSlope, BetaBoundedSatisfiability, \
    BetaClassSatisfiability
from test.abstract import AbstractTest


class TestOptimizers(AbstractTest):
    class Dummy:
        """Dummy class with custom fields."""

        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    def _test_policy(self, optimizer: Optimizer, policy: Callable):
        # test base case then test optimizer wrapping
        for test, opt in dict(base=optimizer, wrapped=Optimizer(base=optimizer)).items():
            macs = self.Dummy(iteration=0)
            for macs.iteration in np.arange(1, self.NUM_TESTS):
                mt_value = opt(macs=macs, x=np.zeros(1), y=np.zeros(1), p=np.zeros(1))
                ref_value = policy(it=macs.iteration)
                msg = f'Error in the {test} case at iteration {macs.iteration}.'
                self.assertAlmostEqual(mt_value, ref_value, places=self.PLACES, msg=msg)

    def _test_satisfiability(self, optimizer: Optimizer, y: np.ndarray, p: np.ndarray, expected: Dict[Loss, float]):
        for test, opt in dict(base=optimizer, wrapped=Optimizer(base=optimizer)).items():
            for loss, ref_value in expected.items():
                master = self.Dummy(p_loss=loss)
                mt_value = opt(macs=self.Dummy(master=master), x=np.zeros(1), y=y, p=p)
                msg = f'Error in the {test} case using loss {loss}.'
                self.assertAlmostEqual(mt_value, ref_value, places=self.PLACES, msg=msg)

    def test_constant(self):
        self._test_policy(optimizer=Optimizer(base=1), policy=lambda it: 1)

    def test_constant_slope(self):
        self._test_policy(optimizer=ConstantSlope(base=1), policy=lambda it: 1 / it)

    def test_exponential_slope(self):
        self._test_policy(optimizer=ExponentialSlope(base=1, slope=2), policy=lambda it: 1 / (2 ** it))

    def test_beta_class_satisfiability(self):
        def ce(a, b):
            b = b.clip(min=1e-15, max=1 - 1e-15)
            return np.mean(a * np.log(b) + (1 - a) * np.log(1 - b))

        p = 0.1 * np.arange(11)
        y = p.round().astype('int')
        self._test_satisfiability(optimizer=BetaClassSatisfiability(base=1), y=y, p=p, expected={
            HammingDistance(): 1,
            CrossEntropy(): 1 - ce(y, p),
            ReversedCrossEntropy(): 1 - ce(p, y),
            SymmetricCrossEntropy(): 1 - ce(y, p) - ce(p, y),
            SAE(binary=True): 1 + np.abs(y - p).sum(),
            SSE(binary=True): 1 + np.square(y - p).sum(),
            MAE(binary=True): 1 + np.abs(y - p).mean(),
            MSE(binary=True): 1 + np.square(y - p).mean()
        })

    def test_beta_bounded_satisfiability(self):
        p = np.arange(-10, 10)
        y = np.arange(-10, 10).clip(min=-5, max=5)
        self._test_satisfiability(optimizer=BetaBoundedSatisfiability(base=1, lb=-5, ub=5), y=y, p=p, expected={
            SAE(): 1 + np.abs(y - p).sum(),
            SSE(): 1 + np.square(y - p).sum(),
            MAE(): 1 + np.abs(y - p).mean(),
            MSE(): 1 + np.square(y - p).mean(),
        })
