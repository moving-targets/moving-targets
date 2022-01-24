from typing import List, Tuple

import numpy as np

from moving_targets.masters.backends import NumpyBackend
from test.abstract import AbstractTest


# Numpy Backend is used to test the correctness of the operations
class TestOperations(AbstractTest):

    def _test(self, name: str, sizes: List[Tuple]):
        np.random.seed(self.SEED)
        backend = NumpyBackend()
        operation = name.split('_')[0]
        for i in range(self.NUM_TESTS):
            args = [np.random.random(s) for s in sizes]
            mt_operation = getattr(backend, operation)
            mt_value = mt_operation(*args)
            ref_operation = getattr(np, operation)
            ref_value = ref_operation(*args)
            msg = f'Error at iteration {i} in operation {name}, {mt_value} != {ref_value}'
            self.assertTrue(np.allclose(mt_value, ref_value, atol=1.0 ** -self.PLACES), msg=msg)

    def test_sum(self):
        self._test(name='sum_1d', sizes=[(self.NUM_SAMPLES,)])
        self._test(name='sum_2d', sizes=[(self.NUM_SAMPLES, self.NUM_FEATURES)])
        self._test(name='sum_3d', sizes=[(self.NUM_SAMPLES, self.NUM_FEATURES, self.NUM_CLASSES)])

    def test_mean(self):
        self._test(name='mean_1d', sizes=[(self.NUM_SAMPLES,)])
        self._test(name='mean_2d', sizes=[(self.NUM_SAMPLES, self.NUM_FEATURES)])
        self._test(name='mean_3d', sizes=[(self.NUM_SAMPLES, self.NUM_FEATURES, self.NUM_CLASSES)])

    def test_std(self):
        self._test(name='std_1d', sizes=[(self.NUM_SAMPLES,)])
        self._test(name='std_2d', sizes=[(self.NUM_SAMPLES, self.NUM_FEATURES)])
        self._test(name='std_3d', sizes=[(self.NUM_SAMPLES, self.NUM_FEATURES, self.NUM_CLASSES)])

    def test_square(self):
        self._test(name='square_1d', sizes=[(self.NUM_SAMPLES,)])
        self._test(name='square_2d', sizes=[(self.NUM_SAMPLES, self.NUM_FEATURES)])
        self._test(name='square_3d', sizes=[(self.NUM_SAMPLES, self.NUM_FEATURES, self.NUM_CLASSES)])

    def test_abs(self):
        self._test(name='abs_1d', sizes=[(self.NUM_SAMPLES,)])
        self._test(name='abs_2d', sizes=[(self.NUM_SAMPLES, self.NUM_FEATURES)])
        self._test(name='abs_3d', sizes=[(self.NUM_SAMPLES, self.NUM_FEATURES, self.NUM_CLASSES)])

    def test_log(self):
        self._test(name='log_1d', sizes=[(self.NUM_SAMPLES,)])
        self._test(name='log_2d', sizes=[(self.NUM_SAMPLES, self.NUM_FEATURES)])
        self._test(name='log_3d', sizes=[(self.NUM_SAMPLES, self.NUM_FEATURES, self.NUM_CLASSES)])

    def test_add(self):
        self._test(name='add_1d', sizes=[(self.NUM_SAMPLES,)] * 2)
        self._test(name='add_2d', sizes=[(self.NUM_SAMPLES, self.NUM_FEATURES)] * 2)
        self._test(name='add_3d', sizes=[(self.NUM_SAMPLES, self.NUM_FEATURES, self.NUM_CLASSES)] * 2)

    def test_multiply(self):
        self._test(name='multiply_1d', sizes=[(self.NUM_SAMPLES,)] * 2)
        self._test(name='multiply_2d', sizes=[(self.NUM_SAMPLES, self.NUM_FEATURES)] * 2)
        self._test(name='multiply_3d', sizes=[(self.NUM_SAMPLES, self.NUM_FEATURES, self.NUM_CLASSES)] * 2)

    def test_divide(self):
        self._test(name='divide_1d', sizes=[(self.NUM_SAMPLES,)] * 2)
        self._test(name='divide_2d', sizes=[(self.NUM_SAMPLES, self.NUM_FEATURES)] * 2)
        self._test(name='divide_3d', sizes=[(self.NUM_SAMPLES, self.NUM_FEATURES, self.NUM_CLASSES)] * 2)

    def test_dot(self):
        self._test(name='dot_row_col', sizes=[(self.NUM_SAMPLES,), (self.NUM_SAMPLES,)])
        self._test(name='dot_row_mat', sizes=[(self.NUM_SAMPLES,), (self.NUM_SAMPLES, self.NUM_FEATURES)])
        self._test(name='dot_mat_col', sizes=[(self.NUM_SAMPLES, self.NUM_FEATURES), (self.NUM_FEATURES,)])
        self._test(name='dot_mat_mat', sizes=[(self.NUM_SAMPLES, self.NUM_FEATURES),
                                              (self.NUM_FEATURES, self.NUM_SAMPLES)])
