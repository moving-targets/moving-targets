from typing import Callable

import numpy as np

from moving_targets.masters import optimizers
from test.test_abstract import TestAbstract


class TestOptimizers(TestAbstract):

    def _test(self, policy: Callable, optimizer: str, **opt_args):
        """Checks whether the optimizer, if instantiated with the given parameters, behaves as the reference policy."""
        dummy = np.zeros(1)
        history = []
        optimizer = optimizers.aliases[optimizer](**opt_args)
        for i in np.arange(self.NUM_TESTS):
            ref_value = policy(it=i)
            step_value = optimizer(x=dummy, y=dummy, p=dummy)
            self.assertAlmostEqual(step_value, ref_value, places=self.PLACES)
            get_value = optimizer.value
            self.assertAlmostEqual(get_value, ref_value, places=self.PLACES)
            history.append(ref_value)
        msg = f"Reference List: {history}\nActual List: {optimizer.history}"
        self.assertTrue(np.allclose(history, optimizer.history, atol=10 ** -self.PLACES), msg=msg)

    def test_constant_default(self):
        self._test(policy=lambda it: 1, optimizer='constant')

    def test_constant_base(self):
        self._test(policy=lambda it: 2, optimizer='constant', base=2)

    def test_harmonic_default(self):
        self._test(policy=lambda it: 1 / (it + 1), optimizer='harmonic')

    def test_harmonic_base(self):
        self._test(policy=lambda it: 2 / (it + 1), optimizer='harmonic', base=2)

    def test_geometric_default(self):
        self._test(policy=lambda it: 1 / (2 ** it), optimizer='geometric')

    def test_geometric_base(self):
        self._test(policy=lambda it: 2 / (2 ** it), optimizer='geometric', base=2)

    def test_geometric_slope(self):
        self._test(policy=lambda it: 1 / (3 ** it), optimizer='geometric', slope=3)
