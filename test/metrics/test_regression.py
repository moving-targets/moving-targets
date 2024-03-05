from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from moving_targets.metrics import MSE, MAE, R2
from test.metrics.test_metrics import TestMetrics


class TestRegressionMetrics(TestMetrics):
    @staticmethod
    def _data_generator(rng):
        y = rng.normal(size=TestRegressionMetrics.NUM_SAMPLES)
        p = rng.normal(size=TestRegressionMetrics.NUM_SAMPLES)
        return [], y, p

    def test_mae(self):
        self._test(data_generator=self._data_generator,
                   mt_metric=MSE(),
                   ref_metric=lambda x, y, p: mean_squared_error(y, p))

    def test_mse(self):
        self._test(data_generator=self._data_generator,
                   mt_metric=MAE(),
                   ref_metric=lambda x, y, p: mean_absolute_error(y, p))

    def test_r2(self):
        self._test(data_generator=self._data_generator,
                   mt_metric=R2(),
                   ref_metric=lambda x, y, p: r2_score(y, p))
