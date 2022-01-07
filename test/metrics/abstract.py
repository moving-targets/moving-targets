from typing import Callable

import numpy as np

from moving_targets.metrics import Metric
from test.abstract import AbstractTest


class TestMetrics(AbstractTest):
    def _test(self, data_generator: Callable, mt_metric: Metric, ref_metric: Callable):
        """Performs the tests on the given data and checks the correctness of the metric wrt to a reference metric.

        :param data_generator:
            The data generator, a function of type f() -> (<x>, <y>, <p>).

        :param mt_metric:
            The `Metric` instance to be tested.

        :param ref_metric:
            A callable function of type f(<x>, <y>, <p>) -> <value> that serves as ground truth.
        """
        np.random.seed(self.SEED)
        for i in range(self.NUM_TESTS):
            x, y, p = data_generator()
            mt_value, ref_value = mt_metric(x, y, p), ref_metric(x, y, p)
            self.assertAlmostEqual(ref_value, mt_value, places=self.PLACES, msg=f'iteration: {i}')
