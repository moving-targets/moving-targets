from typing import Callable

import numpy as np

from moving_targets.metrics import Metric
from test.test_abstract import TestAbstract


class TestMetrics(TestAbstract):
    def _test(self, data_generator: Callable, mt_metric: Metric, ref_metric: Callable):
        """Performs the tests on the given data and checks the correctness of the metric wrt to a reference metric."""
        rng = np.random.default_rng(self.SEED)
        for i in range(self.NUM_TESTS):
            x, y, p = data_generator(rng)
            mt_value, ref_value = mt_metric(x, y, p), ref_metric(x, y, p)
            self.assertAlmostEqual(ref_value, mt_value, places=self.PLACES, msg=f'iteration: {i}')
